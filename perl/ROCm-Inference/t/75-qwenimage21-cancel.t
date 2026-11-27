use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use Time::HiRes qw(sleep time);

# bin/rocm-inference-server --qwen-image DIR: a client that leaves while its image is made cancels the image (it stops after its
# current step, as a failed job), so the next image is not held up by the rest of the steps. Checked: the image of 40 steps is
# started; the client closes after a few seconds; the next image is refused with 429 only until the cancel takes (within a few
# seconds, where the 40 steps would take about 20 more). Needs the ComfyUI models directory (ROCM_INFERENCE_TEST_QWENIMAGE21=DIR, as
# t/73: diffusion_models/, text_encoders/ with tokenizer.json, vae/).
my $dirq = $ENV{ROCM_INFERENCE_TEST_QWENIMAGE21};
plan skip_all => 'set ROCM_INFERENCE_TEST_QWENIMAGE21 (the --qwen-image directory) to run' unless $dirq && -d $dirq;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/qic.sock";
my $J = JSON::PP->new->utf8->canonical;

sub send_req {  # the connection, the request sent
    my ($method, $target, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " . length($body) . "\r\n\r\n$body";
    return $c;
}
sub read_res {  # (status, body)
    my ($c) = @_;
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = ($head // '') =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest);
}
sub http { read_res(send_req(@_)) }

my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--qwen-image', $dirq, '--listen', $sock, '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server with --qwen-image listens') or BAIL_OUT('no server');

# the model loads with the first image (a small one, so the timing below is the sampler's)
my ($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'warm up', size => '256x256', steps => 1 });
is($st, 200, 'a warm-up image (the model loads)') or diag substr($body // '', 0, 300);

# a long image, and its client leaves after a few seconds
my $long = send_req(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'a red cube on a wooden table, soft studio light',
                                                       size => '512x512', steps => 40, seed => 3 });
sleep 3;
($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'a blue ball', size => '256x256', steps => 1 });
is($st, 429, 'while it is made: 429') or diag substr($body // '', 0, 200);
close $long;
my $left = time;

my $free;
for (1 .. 120) {  # the next image is accepted once the cancelled one has stopped
    ($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'a blue ball', size => '256x256', steps => 1,
                                                           seed => 4 });
    if ($st == 200) { $free = time - $left; last }
    sleep 0.5;
}
ok(defined $free, 'the next image is accepted') or diag substr($body // '', 0, 200);
ok(defined $free && $free < 10, sprintf('the cancel takes effect at once (%.1f s after the client left; 8 steps take about 7 s here, so the 40 steps would take about 30 s)',
                                        $free // -1));
my $b64 = eval { $J->decode($body)->{data}[0]{b64_json} };
ok(defined $b64, 'and that image is made') or diag substr($body // '', 0, 200);

kill TERM => $pid;
close $srv;
done_testing;
