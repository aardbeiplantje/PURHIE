# Speech (Qwen3-TTS, phase T4 of PLAN.md): the WAV writer always runs; the synthesis runs when the environment names the
# talker, the codec and a voice directory (ROCM_INFERENCE_TEST_TTS=talker.gguf, ROCM_INFERENCE_TEST_TTS_CODEC=codec.gguf,
# ROCM_INFERENCE_TEST_TTS_VOICE=dir; ROCM_INFERENCE_TEST_WAV=clip.wav for a voice from a clip).
use strict;
use warnings;
use Test::More;
use ROCm::Inference;

my $samples = pack 'f<*', 0.0, 0.5, -0.5, 2.0;
my $wav = ROCm::Inference::speech_wav16($samples, 24000);
is(length $wav, 44 + 8, 'a WAV of four samples: 44 bytes of header, 8 of data');
is(substr($wav, 0, 4), 'RIFF', 'RIFF');
is(substr($wav, 8, 4), 'WAVE', 'WAVE');
is(unpack('V', substr($wav, 24, 4)), 24000, 'sample rate');
is(unpack('v', substr($wav, 34, 2)), 16, '16 bits');
is_deeply([unpack 's<*', substr($wav, 44)], [0, 16384, -16384, 32767], 'samples scaled and clipped');

SKIP: {
    my ($talker, $codec, $voice) = @ENV{qw(ROCM_INFERENCE_TEST_TTS ROCM_INFERENCE_TEST_TTS_CODEC ROCM_INFERENCE_TEST_TTS_VOICE)};
    skip 'set ROCM_INFERENCE_TEST_TTS, ROCM_INFERENCE_TEST_TTS_CODEC and ROCM_INFERENCE_TEST_TTS_VOICE', 3
        unless $talker && $codec && $voice;
    my $s = ROCm::Inference::speech_load($talker, $codec);
    my $v = ROCm::Inference::speech_voice($voice);
    my ($out, $frames) = $s->speak('Hello, this is a short test of the speech synthesis.', voice => $v, seed => 0);
    ok($frames > 0, "talker frames: $frames");
    my ($again) = $s->speak('Hello, this is a short test of the speech synthesis.', voice => $v, seed => 0);
    is($again, $out, 'the same seed, the same samples');
    ok(length($out) > 0 && length($out) % 4 == 0, 'float32 samples');

    # a voice from a clip (T7): the clip's WAV (ROCM_INFERENCE_TEST_WAV, 0.5 to 60 s), written as a voice directory and read back
    SKIP: {
        my $clip = $ENV{ROCM_INFERENCE_TEST_WAV};
        skip 'set ROCM_INFERENCE_TEST_WAV for a voice from a clip', 4 unless $clip && -f $clip;
        open my $fh, '<:raw', $clip or die "$clip: $!";
        local $/;
        my $wav = <$fh>;
        close $fh;
        my $samples = ROCm::Inference::speech_wav_samples($wav, 24000);
        my ($codes, $frames, $spk) = $s->make_voice($samples);
        ok($frames > 0 && length($codes) == $frames * 16 * 4, "voice from the clip: $frames frames");
        ok(length($spk) == 1024 * 4, 'the speaker embedding: 1024 floats');
        require File::Temp;
        my $dir = File::Temp::tempdir(CLEANUP => 1);
        my $made = "$dir/clip-voice";
        ROCm::Inference::speech_voice_write($made, {ref_text => 'a test transcript', frames => $frames, codes => $codes, spk => $spk});
        my $back = ROCm::Inference::speech_voice($made);
        is($back->{frames}, $frames, 'the voice directory reads back');
        is($back->{codes}, $codes, 'its codes are the same bytes');
    }
}

done_testing;
