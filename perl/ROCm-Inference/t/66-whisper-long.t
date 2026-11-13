# Whisper beyond the English clip (PLAN.md W1-W3): long-form audio (windows of 30 s cut at quiet points) and the translate
# task, against the transformers references. Env-gated; without the variables the file skips.
#   ROCM_INFERENCE_TEST_WHISPER   a Whisper GGUF (rocm-convert of a multilingual checkpoint: whisper-tiny)
#   ROCM_INFERENCE_TEST_LONG_WAV  a WAV of the English clip repeated (over 30 s; text: ROCM_INFERENCE_TEST_LONG_COUNT x the sentence)
#   ROCM_INFERENCE_TEST_FR_WAV, ROCM_INFERENCE_TEST_FR_TEXT, ROCM_INFERENCE_TEST_FR_TRANSLATE: a French clip (16 kHz) and the
#                                 transformers transcription (language fr) and translation (into English) of it
use strict;
use warnings;
use Test::More;
use Encode qw(decode);
use ROCm::Inference;

my $gguf = $ENV{ROCM_INFERENCE_TEST_WHISPER};
plan skip_all => 'set ROCM_INFERENCE_TEST_WHISPER to run the Whisper tests'
    unless $gguf && -f $gguf;

sub slurp {
    my ($path) = @_;
    open my $fh, '<:raw', $path or die "$path: $!";
    local $/;
    return <$fh>;
}

my $w = ROCm::Inference::whisper_load($gguf);

SKIP: {
    my ($wav, $want_tr, $want_tl) = @ENV{qw(ROCM_INFERENCE_TEST_FR_WAV ROCM_INFERENCE_TEST_FR_TEXT ROCM_INFERENCE_TEST_FR_TRANSLATE)};
    skip 'set ROCM_INFERENCE_TEST_FR_WAV, _FR_TEXT and _FR_TRANSLATE for the French clip', 2
        unless $wav && -f $wav && $want_tr && -f $want_tr && $want_tl && -f $want_tl;
    my $audio = slurp($wav);
    my ($tr) = $w->transcribe($audio, language => 'fr', task => 'transcribe');
    my ($tl) = $w->transcribe($audio, language => 'fr', task => 'translate');
    my $ref_tr = decode('UTF-8', slurp($want_tr));
    my $ref_tl = decode('UTF-8', slurp($want_tl));
    s/\A\s+|\s+\z//g for $tr, $tl, $ref_tr, $ref_tl;
    is($tr, $ref_tr, 'French: the transcription equals transformers\'');
    is($tl, $ref_tl, 'French: the translation (task translate) equals transformers\'');
}

SKIP: {
    my ($long, $count) = @ENV{qw(ROCM_INFERENCE_TEST_LONG_WAV ROCM_INFERENCE_TEST_LONG_COUNT)};
    skip 'set ROCM_INFERENCE_TEST_LONG_WAV (over 30 s) and ROCM_INFERENCE_TEST_LONG_COUNT', 2
        unless $long && -f $long && $count;
    my $audio = slurp($long);
    my ($text, $n) = $w->transcribe($audio, language => 'en');
    ok($n > 0, "long-form: $n tokens");
    my $sentence = 'Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.';
    my $got = () = $text =~ /\Q$sentence\E/g;
    is($got, $count, "the sentence $count times across the windows");
}

done_testing;
