use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use File::Basename qw(dirname basename);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use Digest::MD5 qw(md5_hex);
use Time::HiRes qw(sleep);

# bin/rocm-inference-server's speech in the formats through ffmpeg (--ffmpeg): mp3, opus and aac. A stub program stands in
# for ffmpeg: it checks the arguments the server gives it for the format (exit 2 when they are not those), reads the input
# and prints "stub FORMAT BYTES MD5". Its failing and empty modes (STUB_MODE) stand in for an encoder that fails. Then the
# server with no ffmpeg found, and, with ROCM_INFERENCE_TEST_FFMPEG set to a real ffmpeg, its three encoders' output.
# The speech models: ROCM_INFERENCE_TEST_TTS (talker), ROCM_INFERENCE_TEST_TTS_CODEC (codec) and
# ROCM_INFERENCE_TEST_TTS_VOICE (a voice directory), as t/65-speech.t.
my ($talker, $codec, $voice_dir) = @ENV{qw(ROCM_INFERENCE_TEST_TTS ROCM_INFERENCE_TEST_TTS_CODEC ROCM_INFERENCE_TEST_TTS_VOICE)};
plan skip_all => 'set ROCM_INFERENCE_TEST_TTS, ROCM_INFERENCE_TEST_TTS_CODEC and ROCM_INFERENCE_TEST_TTS_VOICE to run'
    unless $talker && -f $talker && $codec && -f $codec && $voice_dir && -d $voice_dir;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
require ROCm::Inference;
plan skip_all => "the Minijinja module is not installed"
    unless ROCm::Inference::_require_minijinja();

(my $vd = $voice_dir) =~ s{/+\z}{};
my $voices = dirname($vd);
my $voice = basename($vd);
my $text = 'Hello, this is a short test of the speech synthesis.';
my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/llm.sock";
my $J = JSON::PP->new->utf8->canonical;
my %CTYPE = (mp3 => 'audio/mpeg', opus => 'audio/ogg', aac => 'audio/aac');

# the stub: the arguments of each format (after the common ones), the input's bytes, the failing and the empty modes
my $stub_src = <<'STUB';
#!PERLPATH
use strict;
use warnings;
use Digest::MD5 qw(md5_hex);
my %WANT = (  # the encoder, the bitrate and the container of each format (after the input)
    mp3  => [qw(libmp3lame 64k mp3)],
    opus => [qw(libopus 32k ogg)],
    aac  => [qw(aac 64k adts)],
);
my @args = @ARGV;
my @common = qw(-hide_banner -loglevel error -f s16le -ar 24000 -ac 1 -i);
my $fmt;
if (@args >= 11 && join(' ', @args[0 .. 9]) eq join(' ', @common)) {
    my $tail = join ' ', @args[11 .. $#args];
    for my $f (sort keys %WANT) {
        my ($enc, $rate, $box) = @{ $WANT{$f} };
        $fmt = $f if $tail eq join(' ', '-c:a', $enc, '-b:a', $rate, '-f', $box, 'pipe:1');
    }
}
if (!defined $fmt || $args[10] eq '') {
    print STDERR "stub: unexpected arguments: @args\n";
    exit 2;
}
my $mode = $ENV{STUB_MODE} // '';
if ($mode eq 'fail') {
    print STDERR "stub: this encoder fails\n";
    exit 1;
}
exit 0 if $mode eq 'empty';
open my $fh, '<:raw', $args[10] or do { print STDERR "stub: $args[10]: $!\n"; exit 2 };
my $data = do { local $/; <$fh> };
binmode STDOUT;
print "stub $fmt " . length($data) . ' ' . md5_hex($data) . "\n";
exit 0;
STUB
(my $stub_text = $stub_src) =~ s/PERLPATH/$^X/;
my $stub = "$dir/ffmpeg-stub";
open my $sf, '>', $stub or die "$stub: $!";
print $sf $stub_text;
close $sf;
chmod 0755, $stub or die "chmod $stub: $!";

sub http {
    my ($method, $target, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    binmode $c;
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $target HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
             . length($body) . "\r\n\r\n$body";
    local $/;
    my $raw = <$c> // '';
    my ($head, $rest) = split /\r\n\r\n/, $raw, 2;
    $head //= '';
    my ($status) = $head =~ m{^HTTP/1\.1 (\d+)};
    my ($type) = $head =~ /^Content-Type:[ \t]*([^\r\n]*)/mi;
    return ($status // 0, $rest // '', $type // '');
}

sub speech {
    my (%req) = @_;
    return http(POST => '/v1/audio/speech', { input => $text, voice => $voice, seed => 7, %req });
}

# a server with the speech models and the given ffmpeg options; its environment as set by the caller
sub start_server {
    my (@extra) = @_;
    unlink $sock;
    my $pid = open my $fh, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server',
        '--tts', $talker, '--tts-codec', $codec, '--voices', $voices, '--listen', $sock, '--nolog', @extra;
    die "cannot start the server: $!" unless $pid;
    for (1 .. 1800) {
        last if -S $sock || !kill 0, $pid;
        sleep 0.1;
    }
    return ($pid, $fh);
}

sub stop_server {
    my ($pid, $fh) = @_;
    kill TERM => $pid;
    close $fh;
}

# a. and b.: the server with the stub
my ($wav_samples, $wav_md5);
{
    my ($pid, $fh) = start_server('--ffmpeg', $stub);
    ok(-S $sock, 'the server with --ffmpeg listens') or BAIL_OUT('no server');

    my ($st, $wav, $type) = speech(response_format => 'wav');
    is($st, 200, 'wav: 200');
    like($type, qr{\Aaudio/wav}, 'wav: its Content-Type');
    is(substr($wav, 0, 4), 'RIFF', 'wav: a RIFF file');
    $wav_samples = (length($wav) - 44) / 2;
    ok($wav_samples > 0 && $wav_samples == int $wav_samples, "wav: $wav_samples samples");
    $wav_md5 = md5_hex(substr($wav, 44));

    for my $fmt (qw(mp3 opus aac)) {
        my ($st, $body, $type) = speech(response_format => $fmt);
        is($st, 200, "$fmt: 200") or diag $body;
        is($type, $CTYPE{$fmt}, "$fmt: Content-Type $CTYPE{$fmt}");
        my ($name, $bytes, $md5) = $body =~ /\A(stub) (\w+) (\d+) ([0-9a-f]{32})\n\z/
            ? ($2, $3, $4) : ();
        is($name, $fmt, "$fmt: the stub's format") or diag $body;
        is($bytes, 2 * $wav_samples, "$fmt: the stub's input is 2 bytes per sample of the wav");
        is($md5, $wav_md5, "$fmt: the input is the wav's samples");
    }

    my ($st3, $body3) = speech(response_format => 'mp3', stream => JSON::PP::true);
    is($st3, 400, 'mp3 with stream: refused');

    stop_server($pid, $fh);
}

# e.: the stub fails (exit 1), and the stub's empty output: both 500
for my $mode (qw(fail empty)) {
    local $ENV{STUB_MODE} = $mode;
    my ($pid, $fh) = start_server('--ffmpeg', $stub);
    ok(-S $sock, "the server with the stub in mode $mode listens") or BAIL_OUT('no server');
    my ($st, $body) = speech(response_format => 'mp3');
    is($st, 500, "mp3 when ffmpeg $mode: 500");
    stop_server($pid, $fh);
}

# d.: no ffmpeg (a path that is not one, and no ffmpeg on the PATH): mp3 is refused, wav still works
{
    local $ENV{PATH} = $dir;
    local $ENV{ROCM_FFMPEG};
    delete $ENV{ROCM_FFMPEG};
    my ($pid, $fh) = start_server('--ffmpeg', "$dir/no-such-ffmpeg");
    ok(-S $sock, 'the server without an ffmpeg listens') or BAIL_OUT('no server');
    my ($st, $body) = speech(response_format => 'mp3');
    is($st, 400, 'mp3 without ffmpeg: 400');
    like($J->decode($body)->{error}{message}, qr/mp3.*opus.*aac.*ffmpeg/, 'the message names the three formats and ffmpeg');
    my ($st2, $wav) = speech(response_format => 'wav');
    is($st2, 200, 'wav without ffmpeg: 200');
    is((length($wav) - 44) / 2, $wav_samples, 'the same samples as with the stub');
    stop_server($pid, $fh);
}

# f.: a real ffmpeg (ROCM_INFERENCE_TEST_FFMPEG): the magic bytes of each format
SKIP: {
    my $real = $ENV{ROCM_INFERENCE_TEST_FFMPEG};
    skip 'set ROCM_INFERENCE_TEST_FFMPEG to a real ffmpeg to test the encoders', 6
        unless $real && -x $real;
    my ($pid, $fh) = start_server('--ffmpeg', $real);
    ok(-S $sock, 'the server with a real ffmpeg listens') or BAIL_OUT('no server');
    my ($st, $mp3) = speech(response_format => 'mp3');
    is($st, 200, 'real ffmpeg: mp3 200');
    ok(substr($mp3, 0, 3) eq 'ID3' || (length($mp3) > 1 && ord(substr($mp3, 0, 1)) == 0xFF
        && (ord(substr($mp3, 1, 1)) & 0xE0) == 0xE0), 'mp3: ID3 or an MPEG frame sync');
    ($st, my $opus) = speech(response_format => 'opus');
    is($st, 200, 'real ffmpeg: opus 200');
    is(substr($opus, 0, 4), 'OggS', 'opus: an Ogg page');
    ($st, my $aac) = speech(response_format => 'aac');
    is($st, 200, 'real ffmpeg: aac 200');
    ok(substr($aac, 0, 2) eq "\xFF\xF1" || substr($aac, 0, 2) eq "\xFF\xF9", 'aac: an ADTS sync');
    stop_server($pid, $fh);
}

done_testing;
