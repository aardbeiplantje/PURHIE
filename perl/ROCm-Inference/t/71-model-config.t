use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use Time::HiRes qw(sleep);

# bin/rocm-inference-server: each model's own settings (parallel, ctx, kv_type): the sidecar NAME.json beside the model
# (checked at each load) overrides the host rules, which override the global flags. A sidecar's unknown key fails the
# load with its file name. The model list shows each chat model's parallel. Needs any chat GGUF:
# ROCM_INFERENCE_TEST_MODEL.
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
require ROCm::Inference;
plan skip_all => "the Minijinja module is not installed"
    unless ROCm::Inference::_require_minijinja();

my $dir = tempdir(CLEANUP => 1);
my $models = "$dir/models";
mkdir $models or die;
symlink $path, "$models/plain.gguf" or die;
symlink $path, "$models/one.gguf" or die;
symlink $path, "$models/bad.gguf" or die;
{ open my $fh, '>', "$models/one.json" or die; print $fh qq({"parallel": 1}\n) }
{ open my $fh, '>', "$models/bad.json" or die; print $fh qq({"para": 2}\n) }
my $sock = "$dir/cfg.sock";
my $J = JSON::PP->new->utf8->canonical;

sub http {
    my ($method, $url, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $url HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
             . length($body) . "\r\n\r\n$body";
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = $head =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest);
}

my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--models-dir', $models,
    '--listen', $sock, '--ctx', 2048, '--parallel', 2, '--log-level', 'info', '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server listens') or BAIL_OUT('no server');

my (undef, $body) = http(GET => '/v1/models');
my %m = map { $_->{id} => $_ } @{ $J->decode($body)->{data} };
is($m{plain}{parallel}, 2, 'no sidecar: the global --parallel 2');
is($m{one}{parallel}, 1, 'sidecar {"parallel": 1} overrides the global flag');
is($m{plain}{context_length}, 2048, 'the context from the global --ctx');

my ($st, $r) = http(POST => '/v1/chat/completions', { model => 'one', max_tokens => 4,
    chat_template_kwargs => { enable_thinking => JSON::PP::false },
    messages => [ { role => 'user', content => 'Say hi.' } ] });
is($st, 200, 'a chat with a model whose sidecar sets parallel 1');

($st, $r) = http(POST => '/v1/chat/completions', { model => 'bad', max_tokens => 4,
    messages => [ { role => 'user', content => 'Say hi.' } ] });
is($st, 500, 'a sidecar with an unknown key: the load fails');
like($r, qr/bad\.json: unknown key 'para'/, 'and the error names the file and the key');

kill TERM => $pid;
close $srv;
ok(!-e $sock, 'SIGTERM: the server stops');

done_testing;
