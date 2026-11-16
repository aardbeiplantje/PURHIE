use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);
use File::Basename qw(basename);

# bin/rocm-inference-server --models-dir: the talker a directory's speech files give. A BF16 talker before one with a
# Q8_0 predictor (the predictor's Q8_0 recipe), which is before an F32 one. The rank is the tensors' bytes, not the name:
# by name the Q8_0 file would come first ("qwen3tts-talker-predq8" sorts before "qwen3tts-talker.").
my ($bf16, $q8, $f32, $codec) = @ENV{qw(ROCM_INFERENCE_TEST_TTS ROCM_INFERENCE_TEST_TTS_Q8 ROCM_INFERENCE_TEST_TTS_F32
                                       ROCM_INFERENCE_TEST_TTS_CODEC)};
plan skip_all => 'set ROCM_INFERENCE_TEST_TTS (a BF16 talker), ROCM_INFERENCE_TEST_TTS_Q8 (a talker with a Q8_0 predictor) '
    . 'and ROCM_INFERENCE_TEST_TTS_CODEC to run'
    unless $bf16 && $q8 && $codec && -f $bf16 && -f $q8 && -f $codec;

# The talker the server loads for a directory holding these files (its "load speech (--tts FILE" log line)
sub talker_loaded {
    my (@talkers) = @_;
    my $dir = tempdir(CLEANUP => 1);
    for my $f (@talkers, $codec) {
        symlink $f, "$dir/" . basename($f) or die "symlink $f: $!";
    }
    my $pid = open my $srv, '-|', 'sh', '-c', 'exec "$0" "$@" 2>&1', $^X, (map { "-I$_" } @INC),
        'bin/rocm-inference-server', '--models-dir', $dir, '--listen', "$dir/sock";
    die "cannot start the server: $!" unless $pid;
    my $found;
    eval {
        local $SIG{ALRM} = sub { die "timeout\n" };
        alarm 120;
        while (my $line = <$srv>) {
            if ($line =~ /\bload\s+speech \(--tts (\S+)/) { $found = basename($1); last }
        }
        alarm 0;
    };
    alarm 0;
    kill 'TERM', $pid;
    close $srv;
    return $found;
}

is(talker_loaded($bf16, $q8), basename($bf16), 'the BF16 talker, before the Q8_0 predictor one');
is(talker_loaded($q8, $bf16), basename($bf16), 'the same whatever the order the files are listed');
SKIP: {
    skip 'set ROCM_INFERENCE_TEST_TTS_F32 (an F32 talker) to test the last choice', 1 unless $f32 && -f $f32;
    is(talker_loaded($f32, $q8), basename($q8), 'the Q8_0 predictor talker, before the F32 one');
}

done_testing;
