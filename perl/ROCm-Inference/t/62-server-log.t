use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use Time::HiRes qw(sleep time);

# bin/rocm-inference-server's log (--log-level debug --log-format json --debug-stream), as is and with --parallel 2:
# a request's lines in order (start, the prefill's progress and end, done), the streamed pieces, and a client that
# leaves during a long prefill (a plain JSON request and a streamed one): its prefill is cancelled, and the server
# goes on serving
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
require ROCm::Inference;
plan skip_all => "Minijinja $ROCm::Inference::MINIJINJA_VERSION is not installed"
    unless ROCm::Inference::_require_minijinja();

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/llm.sock";
my $J = JSON::PP->new->utf8->canonical;

# a request as user $user (X-Remote-User labels its log lines); its response read by http_read
sub http_send {
    my ($user, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = $J->encode($data);
    print $c "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nX-Remote-User: $user\r\n"
             . "Content-Type: application/json\r\nContent-Length: " . length($body) . "\r\n\r\n$body";
    return $c;
}

sub http_read {
    my ($c) = @_;
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c> // '', 2;
    my ($status) = ($head // '') =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest);
}

# the log's JSON lines of a user (and the access lines, which carry the user too)
my $log = "$dir/server.log";
sub lines_of {
    my ($user) = @_;
    open my $f, '<', $log or return ();
    return grep { ($_->{user} // '') eq $user } map { my $l = eval { $J->decode($_) }; $l ? $l : () } <$f>;
}

# waits up to $secs for a line of $user that $test accepts
sub wait_line {
    my ($user, $test, $secs) = @_;
    my $t0 = time;
    while (time - $t0 < $secs) {
        my @l = grep { $test->($_) } lines_of($user);
        return $l[0] if @l;
        sleep 0.05;
    }
    return;
}

# a long prompt (~8k tokens: several prefill steps, a few seconds); the same each time but for its first line, so
# that no request reuses another's prefix
my $filler = join ' ', map { "Item $_ of the inventory list is a box of " . (7 * $_ % 97) . " screws." } 1 .. 600;
sub long_prompt { "Request $_[0]. Read the list, then say how many items there are.\n$filler" }

my %q = (temperature => 0, chat_template_kwargs => { enable_thinking => JSON::PP::false });

for my $parallel (1, 2) {
    note("--parallel $parallel");
    my $p = "p$parallel";
    open my $err, '>', $log or die;
    close $err;
    my $pid = fork // die "fork: $!";
    if (!$pid) {
        open STDERR, '>>', $log or die;
        open STDOUT, '>', '/dev/null' or die;
        exec $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--model', $path, '--listen', $sock,
            '--ctx', 16384, '--name', 'test-model', '--parallel', $parallel, '--log-level', 'debug',
            '--log-format', 'json', '--debug-stream';
        die "exec: $!";
    }
    for (1 .. 600) { last if -S $sock; sleep 0.1 }
    ok(-S $sock, 'the server listens') or BAIL_OUT('no server');

    # a plain request: its lines in order
    my ($st, $body) = http_read(http_send("$p-plain", { %q, max_tokens => 16,
        messages => [ { role => 'user', content => long_prompt("$p-plain") } ] }));
    is($st, 200, 'a long prompt: 200');
    my @l = lines_of("$p-plain");
    my @ev = map { $_->{event} // '' } @l;
    my $at = sub { my ($e) = @_; my ($i) = grep { $ev[$_] eq $e } 0 .. $#ev; $i // -1 };
    ok($at->('start') >= 0, 'a start line');
    my @prog = grep { ($_->{event} // '') eq 'prefill_progress' } @l;
    ok(@prog >= 2, scalar(@prog) . ' prefill progress lines');
    my @done_tok = map { $_->{msg} =~ m{(\d+)/\d+ tok} ? $1 : -1 } @prog;
    ok(!grep({ $done_tok[$_] <= $done_tok[ $_ - 1 ] } 1 .. $#done_tok), "progress goes up: @done_tok");
    my @pend = grep { ($_->{event} // '') eq 'prefill_end' } @l;
    is(scalar @pend, 1, 'one prefill_end line (with --parallel: for all of its chunks)');
    like($pend[0]{msg} // '', qr/^prefill done: (\d+) tok/, 'prefill done: its tokens');
    my ($done) = grep { ($_->{event} // '') eq 'done' } @l;
    like($done->{msg} // '', qr/prompt \d+ tok \(reused \d+, prefilled \d+\), pp [\d.]+ tok\/s; \d+ tok out/,
         "the done line: prompt, reuse and rates: " . ($done->{msg} // ''));
    ok($at->('start') < $at->('prefill_progress') && $at->('prefill_progress') < $at->('prefill_end')
       && $at->('prefill_end') < $at->('done'), 'in order: start, progress, prefill end, done');
    is((grep { $_->{event} eq 'access' } @l)[0]{msg} =~ / 200 / ? 1 : 0, 1, 'the access line: 200');

    # a streamed request: --debug-stream logs each piece sent
    ($st, $body) = http_read(http_send("$p-stream", { %q, max_tokens => 16, stream => JSON::PP::true,
        messages => [ { role => 'user', content => 'Name three primary colors.' } ] }));
    my $sent = join '', map { /^data: (\{.*)/ ? $J->decode($1)->{choices}[0]{delta}{content} // '' : '' } split /\n\n/, $body;
    my $logged = join '', map { $J->decode($_->{msg}) } grep { ($_->{kind} // '') eq 'stream' && $_->{event} eq 'content' }
        lines_of("$p-stream");
    ok(length $sent, "streamed: $sent");
    is($logged, $sent, '--debug-stream: the pieces logged are those sent');

    # clients that leave during the prefill: a plain JSON request (nothing written to it: the socket tells), and a
    # streamed one
    for my $kind ('plain', 'stream') {
        my $u = "$p-gone-$kind";
        my $c = http_send($u, { %q, max_tokens => 16, ($kind eq 'stream' ? (stream => JSON::PP::true) : ()),
            messages => [ { role => 'user', content => long_prompt($u) } ] });
        my $first = wait_line($u, sub { ($_[0]{event} // '') eq 'prefill_progress' }, 60);
        ok($first, "$kind: the prefill runs");
        close $c;
        my $gone = wait_line($u, sub { ($_[0]{msg} // '') =~ /client gone during the prefill/ }, 60);
        ok($gone, "$kind: its prefill is cancelled: " . ($gone ? $gone->{msg} : 'no line'));
        my @after = lines_of($u);
        ok(!grep({ ($_->{event} // '') eq 'done' } @after), "$kind: no done line");
        my ($pe) = grep { ($_->{event} // '') eq 'prefill_end' } @after;
        ok(!$pe, "$kind: the prefill did not end");
    }

    # and the server goes on
    ($st, $body) = http_read(http_send("$p-after", { %q, max_tokens => 24,
        messages => [ { role => 'user', content => 'What is 7 times 8? Answer with the number.' } ] }));
    is($st, 200, 'after the cancels: 200');
    like($J->decode($body)->{choices}[0]{message}{content}, qr/56/, 'and a right answer');

    kill TERM => $pid;
    waitpid $pid, 0;
    ok(!-e $sock, 'SIGTERM: the server stops and removes its socket');
}

done_testing;
