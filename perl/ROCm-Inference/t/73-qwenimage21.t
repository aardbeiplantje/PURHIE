use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use MIME::Base64 qw(decode_base64);
use Time::HiRes qw(sleep time);

# bin/rocm-inference-server --qwen-image DIR: qwen-image-2.1 on /v1/images/generations (text to image, an RGBA PNG). Checked: the
# listing (kind image, unloaded until its first request), the refusals (no prompt, a size that is not a multiple of 32, steps out of
# range), an image made while the server answers other requests (the answer waits in a pull response; /health answers meanwhile; a
# second image is refused with 429), the PNG (RGBA, the size asked), the b64_json answer, the model loaded afterwards. Needs the ComfyUI
# models directory with the Qwen-Image 2.1 files (ROCM_INFERENCE_TEST_QWENIMAGE21=DIR, as --qwen-image takes it: diffusion_models/,
# text_encoders/ with tokenizer.json, vae/); about two minutes (the model's registration, two small images).
my $dirq = $ENV{ROCM_INFERENCE_TEST_QWENIMAGE21};
plan skip_all => 'set ROCM_INFERENCE_TEST_QWENIMAGE21 (the --qwen-image directory) to run' unless $dirq && -d $dirq;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/qi.sock";
my $J = JSON::PP->new->utf8->canonical;

sub send_req {  # the connection, the request sent
    my ($method, $target, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " . length($body) . "\r\n\r\n$body";
    return $c;
}
sub read_res {  # (status, body, head)
    my ($c) = @_;
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = ($head // '') =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest, $head);
}
sub http { read_res(send_req(@_)) }
sub png_info {  # (width, height, colour type) of a PNG's IHDR, or ()
    my ($png) = @_;
    return () unless defined $png && length $png > 33 && substr($png, 0, 8) eq "\x89PNG\r\n\x1a\n" && substr($png, 12, 4) eq 'IHDR';
    return (unpack('N', substr($png, 16, 4)), unpack('N', substr($png, 20, 4)), ord substr($png, 25, 1));
}

my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--qwen-image', $dirq, '--listen', $sock, '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server with --qwen-image listens') or BAIL_OUT('no server');

my ($st, $body) = http(GET => '/v1/models?kind=image');
my ($m) = grep { $_->{id} eq 'qwen-image-2.1' } eval { @{ $J->decode($body)->{data} } };
ok($m, 'qwen-image-2.1 is listed') or diag $body;
is($m && $m->{status}{value}, 'unloaded', 'unloaded until its first request');
ok($m && $m->{description} && $m->{usage} && $m->{parameters}{steps}, 'its purpose, usage and parameters');

($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1' });
is($st, 400, 'no prompt: refused') or diag $body;
($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'x', size => '300x300' });
is($st, 400, 'a size that is not a multiple of 32: refused') or diag $body;
like($body // '', qr/multiples of 32/, 'with the reason');
($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'x', steps => 0 });
is($st, 400, 'steps 0: refused') or diag $body;

# an image while the server serves the others
my $t0 = time;
my $c = send_req(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'a red cube on a wooden table, soft studio light',
                                                     size => '256x256', steps => 4, seed => 1, response_format => 'png' });
sleep 1;
my $th = time;
($st, $body) = http(GET => '/health');
is($st, 200, 'the server answers while the image is made');
ok(time - $th < 3, sprintf('  at once (%.2f s)', time - $th));
($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'a blue ball', size => '256x256', steps => 1 });
is($st, 429, 'a second image meanwhile: 429') or diag "$st: " . ($body // '');
my ($ist, $img, $head) = read_res($c);
is($ist, 200, sprintf('the image (%.0f s with the load)', time - $t0)) or diag substr($img // '', 0, 300);
like($head // '', qr{^Content-Type: image/png}mi, 'as image/png');
my ($w, $h, $ct) = png_info($img);
is_deeply([$w, $h, $ct], [256, 256, 6], 'a PNG of 256 x 256, RGBA (colour type 6)');

# the b64_json answer, the model now loaded
($st, $body) = http(POST => '/v1/images/generations', { model => 'qwen-image-2.1', prompt => 'a blue ball', size => '256x256', steps => 2, seed => 2 });
is($st, 200, 'a second image after the first') or diag substr($body // '', 0, 300);
my $b64 = eval { $J->decode($body)->{data}[0]{b64_json} };
my ($w2, $h2) = png_info(defined $b64 ? decode_base64($b64) : undef);
is_deeply([$w2, $h2], [256, 256], 'b64_json: the PNG in data[0].b64_json');
($st, $body) = http(GET => '/v1/models/qwen-image-2.1');
is(eval { $J->decode($body)->{status}{value} }, 'loaded', 'the model is loaded now');

kill TERM => $pid;
close $srv;
done_testing;
