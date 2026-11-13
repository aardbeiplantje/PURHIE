use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use Time::HiRes qw(sleep);
use ROCm::Inference::Multipart;

# the multipart parser: the parts of a body, the file's bytes as sent (no model needed)
{
    my $b = 'XyZ';
    my $body = join '', "--$b\r\n", qq(Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n),
               "--$b\r\n", qq(Content-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n),
               "RIFF\r\n\0binary", "\r\n--$b--\r\n";
    my $p = ROCm::Inference::Multipart::parse($body, qq(multipart/form-data; boundary="$b"));
    is($p->{model}{data}, 'whisper-1', 'a field');
    ok(!exists $p->{model}{filename}, 'a field has no filename');
    is($p->{file}{filename}, 'a.wav', 'a file part keeps its filename');
    is($p->{file}{type}, 'audio/wav', 'and its type');
    is($p->{file}{data}, "RIFF\r\n\0binary", 'the bytes as sent, CRLF and NUL included');
    ok(!eval { ROCm::Inference::Multipart::parse($body, 'application/json'); 1 }, 'not multipart: refused');
}

# the server's transcription, with a speech model (ROCM_INFERENCE_TEST_WHISPER: a Whisper GGUF; ROCM_INFERENCE_TEST_WAV:
# a WAV of the same speech, 16 kHz; its transcription is ROCM_INFERENCE_TEST_TEXT)
my $gguf = $ENV{ROCM_INFERENCE_TEST_WHISPER};
my $wav = $ENV{ROCM_INFERENCE_TEST_WAV};
plan skip_all => 'set ROCM_INFERENCE_TEST_WHISPER and ROCM_INFERENCE_TEST_WAV to run the server test'
    unless $gguf && -f $gguf && $wav && -f $wav;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
require ROCm::Inference;
plan skip_all => 'the Minijinja module is not installed' unless ROCm::Inference::_require_minijinja();

my $want = $ENV{ROCM_INFERENCE_TEST_TEXT} // 'Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.';
my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/llm.sock";
my $J = JSON::PP->new->utf8->canonical;
my $audio = do { open my $fh, '<:raw', $wav or die "$wav: $!"; local $/; <$fh> };

sub http {
    my ($method, $target, $ctype, $body) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: $ctype\r\nContent-Length: " . length($body) . "\r\n\r\n$body";
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = $head =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest // '', $head);
}

sub multipart {
    my ($b, @parts) = @_;
    my $body = '';
    for my $p (@parts) {
        my ($name, $data, $filename) = @$p;
        $body .= "--$b\r\nContent-Disposition: form-data; name=\"$name\"" . (defined $filename ? "; filename=\"$filename\"" : '')
               . "\r\n\r\n$data\r\n";
    }
    return $body . "--$b--\r\n";
}

my $b = 'AUDIOBOUNDARY';
my $ct = "multipart/form-data; boundary=$b";
my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--whisper', $gguf,
    '--listen', $sock, '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server with --whisper listens') or BAIL_OUT('no server');

my ($st, $body) = http(POST => '/v1/audio/transcriptions', $ct, multipart($b, [file => $audio, 'clip.wav'], [model => 'whisper-1'], [language => 'en']));
is($st, 200, 'a transcription') or diag $body;
is($J->decode($body)->{text}, $want, 'the text of the clip');

($st, $body) = http(POST => '/v1/audio/transcriptions', $ct, multipart($b, [file => $audio, 'clip.wav'], [response_format => 'text']));
is($st, 200, 'response_format text');
is($body, $want, 'as plain text');

($st, $body) = http(POST => '/v1/audio/transcriptions', $ct, multipart($b, [model => 'whisper-1']));
is($st, 400, 'no file: refused');
like($J->decode($body)->{error}{message}, qr/file/, 'and the error names the file');

($st, $body) = http(POST => '/v1/audio/transcriptions', $ct, multipart($b, [file => 'not audio at all', 'x.wav']));
is($st, 400, 'not a WAV: refused') or diag $body;

($st, $body) = http(POST => '/v1/audio/transcriptions', $ct, multipart($b, [file => $audio, 'clip.wav'], [language => 'nl-NL']));
is($st, 400, 'a language that is not a code: refused');

($st, $body) = http(POST => '/v1/audio/transcriptions', $ct, multipart($b, [file => $audio, 'clip.wav'], [response_format => 'srt']));
is($st, 400, 'srt is not offered');

kill TERM => $pid;
close $srv;
ok(!-e $sock, 'SIGTERM: the server stops');
done_testing;
