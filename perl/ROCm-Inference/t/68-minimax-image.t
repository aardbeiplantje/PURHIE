use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use MIME::Base64 qw(decode_base64);
use Time::HiRes qw(sleep);

# bin/rocm-inference-server --minimax DIR: /v1/images/generations with model "minimax-h3-image" generates an image (a PNG) with
# the MiniMax-H3-Image pipeline; an unknown model is refused (400) without running anything. Needs the four GGUFs
# (ROCM_INFERENCE_TEST_MINIMAX=DIR: text-encoder-bf16.gguf, dit-bf16.gguf, pdd-heads.gguf, vae-decoder-f32.gguf; rocm-convert
# of /hf/MiniMax-H3-Image) and about six minutes for the one 256 x 256 request (the released canvas is 2048 x 2048).
my $dirm = $ENV{ROCM_INFERENCE_TEST_MINIMAX};
plan skip_all => 'set ROCM_INFERENCE_TEST_MINIMAX to the directory of the four GGUFs to run'
    unless $dirm && -d $dirm && -f "$dirm/dit-bf16.gguf";
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/img.sock";
my $J = JSON::PP->new->utf8->canonical;

sub http {
    my ($method, $target, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
             . length($body) . "\r\n\r\n$body";
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = $head =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest);
}

my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--minimax', $dirm,
    '--listen', $sock, '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server with --minimax listens') or BAIL_OUT('no server');

my ($st, $body) = http(POST => '/v1/images/generations', { model => 'no-such-model', prompt => 'a fox', size => '256x256' });
is($st, 400, 'an unknown model is refused') or diag $body;

($st, $body) = http(POST => '/v1/images/generations', { model => 'minimax-h3-image', prompt => 'a photo of a red fox in the snow',
                                                        size => '256x256', seed => 0, response_format => 'b64_json' });
is($st, 200, 'an image from the MiniMax pipeline') or diag substr($body, 0, 300);
my $png = eval { decode_base64($J->decode($body)->{data}[0]{b64_json}) } // '';
is(substr($png, 0, 8), "\x89PNG\r\n\x1a\n", 'the image is a PNG') or diag length($png);

kill TERM => $pid;
close $srv;
done_testing;
