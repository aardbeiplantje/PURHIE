use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use MIME::Base64 qw(encode_base64);
use Time::HiRes qw(sleep);

# bin/rocm-inference-server with --mmproj: a chat with an image (image_url, a data: URL) answers about it; the same
# server without --mmproj refuses the image (400). Needs a Qwen3.5 model with its vision tower (rocm-convert --mmproj)
# and an image: ROCM_INFERENCE_TEST_MODEL, ROCM_INFERENCE_TEST_MMPROJ, ROCM_INFERENCE_TEST_IMAGE (a PNG or JPEG file)
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
my $mmproj = $ENV{ROCM_INFERENCE_TEST_MMPROJ};
my $image = $ENV{ROCM_INFERENCE_TEST_IMAGE};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL, ROCM_INFERENCE_TEST_MMPROJ and ROCM_INFERENCE_TEST_IMAGE to run'
    unless $path && -f $path && $mmproj && -f $mmproj && $image && -f $image;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
require ROCm::Inference;
plan skip_all => "the Minijinja module is not installed"
    unless ROCm::Inference::_require_minijinja();

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/llm.sock";
my $J = JSON::PP->new->utf8->canonical;
my $bytes = do { open my $fh, '<:raw', $image or die "$image: $!"; local $/; <$fh> };
my $mime = $image =~ /\.jpe?g\z/i ? 'jpeg' : 'png';
my $url = "data:image/$mime;base64," . encode_base64($bytes, '');

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

sub start_server {
    my (@extra) = @_;
    my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--model', $path,
        '--listen', $sock, '--ctx', 4096, '--name', 'test-model', '--nolog', @extra;
    die "cannot start the server: $!" unless $pid;
    for (1 .. 600) { last if -S $sock; sleep 0.1 }
    return ($pid, $srv);
}

sub stop_server {
    my ($pid, $srv) = @_;
    kill TERM => $pid;
    close $srv;
}

my @image_msg = ({ role => 'user', content => [
    { type => 'image_url', image_url => { url => $url } },
    { type => 'text', text => 'What shapes and colours are in this image? Answer in one short sentence.' } ] });
my %q = (temperature => 0, max_tokens => 40, chat_template_kwargs => { enable_thinking => JSON::PP::false });

{
    my @srv = start_server('--mmproj', $mmproj);
    ok(-S $sock, 'the server with --mmproj listens') or BAIL_OUT('no server');
    my ($st, $body) = http(POST => '/v1/chat/completions', { %q, messages => \@image_msg });
    is($st, 200, 'an image chat') or diag $body;
    my $reply = eval { $J->decode($body)->{choices}[0]{message}{content} } // '';
    like($reply, qr/red/i, "the answer names the red shape: $reply");
    like($reply, qr/blue/i, 'and the blue one');
    # the conversation's next turn: the image stays in the history (its pads prefilled again, with the image)
    ($st, $body) = http(POST => '/v1/chat/completions', { %q, messages => [ @image_msg,
        { role => 'assistant', content => $reply }, { role => 'user', content => 'And how many shapes are there?' } ] });
    is($st, 200, 'a second turn with the image in the history') or diag $body;
    like(eval { $J->decode($body)->{choices}[0]{message}{content} } // '', qr/two|2/i, 'its answer');
    ($st, $body) = http(POST => '/v1/chat/completions', { %q, messages => [ { role => 'user', content => 'Name three primary colors.' } ] });
    is($st, 200, 'a text chat after the image chat');
    like(eval { $J->decode($body)->{choices}[0]{message}{content} } // '', qr/red|blue|yellow/i, 'its answer');
    ($st, $body) = http(POST => '/v1/chat/completions', { %q, messages => [ { role => 'user', content => [
        { type => 'image_url', image_url => { url => 'https://example.com/a.png' } } ] } ] });
    is($st, 400, 'a URL (not a data: URL) is refused');
    stop_server(@srv);
    ok(!-e $sock, 'SIGTERM: the server stops');
}

{
    my @srv = start_server();
    ok(-S $sock, 'the server without --mmproj listens') or BAIL_OUT('no server');
    my ($st, $body) = http(POST => '/v1/chat/completions', { %q, messages => \@image_msg });
    is($st, 400, 'an image without a vision tower is refused') or diag $body;
    like($J->decode($body)->{error}{message} // $body, qr/mmproj/, 'the error says what to start the server with');
    stop_server(@srv);
}

done_testing;
