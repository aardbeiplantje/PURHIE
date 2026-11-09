use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);
use File::Spec;

use ROCm::Inference ();
use ROCm::Inference::GGUF qw(gguf_read gguf_read_rows gguf_write_header gguf_type_id gguf_type_info);

# The quantizer without a model or a GPU: ROCm::Inference::quantize (the engine's encoders), ROCm::Inference::GGUF
# (a header written and read back) and bin/rocm-convert on a GGUF made here. That the encoders write the
# reference's bytes is tests/quantize_verify.sh's (it needs the llama.cpp oracle).

my ($F32, $F16, $BF16, $Q8_0, $FP4, $FAST) = map { gguf_type_id($_) } qw(f32 f16 bf16 q8_0 q4_0_rocmfp4 q4_0_rocmfp4_fast);

# Row sizes and what can be encoded
is(ROCm::Inference::quant_row_size($FAST, 4096), 4096 / 32 * 17, 'FAST: 17 bytes a block of 32');
is(ROCm::Inference::quant_row_size($FP4, 4096), 4096 / 32 * 18, 'ROCmFP4: 18 bytes a block of 32');
is(ROCm::Inference::quant_row_size($Q8_0, 100), 0, 'Q8_0: 100 values are not whole blocks');
ok(ROCm::Inference::quant_can_encode($_), (gguf_type_info($_))[0] . ' can be encoded') for $F32, $F16, $BF16, $Q8_0, $FP4, $FAST;
ok(!ROCm::Inference::quant_can_encode(gguf_type_id('q6_K')), 'Q6_K cannot (decoded only)');

# Exact cases: F16 / BF16 of representable values, Q8_0 of a block with max 127, FAST of zeros
my @v = map { ($_ - 16) / 4 } 0 .. 31;  # -4 .. 3.75 in quarters: exact in F16 and BF16
my $f32 = pack 'f<*', @v;
is_deeply([unpack 'f<*', ROCm::Inference::quantize($F32, $F16, ROCm::Inference::quantize($F16, $F32, $f32, 1, 32), 1, 32)],
          \@v, 'F32 -> F16 -> F32 is exact for quarters');
is_deeply([unpack 'f<*', ROCm::Inference::quantize($F32, $BF16, ROCm::Inference::quantize($BF16, $F32, $f32, 1, 32), 1, 32)],
          \@v, 'F32 -> BF16 -> F32 is exact for quarters');
my $q8 = ROCm::Inference::quantize($Q8_0, $F32, pack('f<*', map { $_ - 15 } 0 .. 31), 1, 32);
is(length $q8, 34, 'Q8_0: one block');
is_deeply([unpack 'x2c32', $q8], [map { int(($_ - 15) * 127 / 16 + ($_ >= 15 ? 0.5 : -0.5)) } 0 .. 31],
          'Q8_0: values scaled by 127 / max, rounded half away from zero');
is(ROCm::Inference::quantize($FAST, $F32, pack('f<*', (0) x 64), 2, 32), "\0" x 34, 'FAST: zeros are zero bytes');
my $big = pack 'f<*', map { sin($_) } 1 .. 64 * 512;
is(ROCm::Inference::quantize($FAST, $F32, $big, 64, 512, 1), ROCm::Inference::quantize($FAST, $F32, $big, 64, 512, 8),
   'FAST: 8 threads = 1 thread');
eval { ROCm::Inference::quantize($FAST, $F32, "\0" x 100, 1, 32) };
like($@, qr/bytes of input/, 'short input: dies');

# A GGUF written and read back: every metadata type, arrays of numbers and strings, tensors at aligned offsets
my $dir = tempdir(CLEANUP => 1);
my @kv = (['general.architecture', 8, 'qwen3'], ['t.u8', 0, 200], ['t.i8', 1, -5], ['t.u16', 2, 60000],
          ['t.i16', 3, -3000], ['t.u32', 4, 4000000000], ['t.i32', 5, -2000000000], ['t.f32', 6, 0.5],
          ['t.bool', 7, 1], ['t.u64', 10, 2**40], ['t.i64', 11, -2**40], ['t.f64', 12, 0.25],
          ['t.arr', 9, [5, [1, -2, 3]]], ['t.strs', 9, [8, ['a', '', "\x{e9}t\x{e9}"]]], ['t.empty', 9, [4, []]]);
utf8::encode($kv[-2][2][1][2]);
my @w1 = map { sin($_) } 1 .. 4 * 64;
my @tensors = ({name => 'blk.0.attn_q.weight', ne => [64, 4], type => $F32, data => pack('f<*', @w1)},
               {name => 'blk.0.attn_norm.weight', ne => [64], type => $F32, data => pack('f<*', (1) x 64)},
               {name => 'blk.0.ssm_conv1d.weight', ne => [4, 8], type => $F32, data => pack('f<*', (0.5) x 32)},
               {name => 'blk.0.ffn_up.weight', ne => [48, 2], type => $F32, data => pack('f<*', (0.25) x 96)},
               {name => 'output.weight', ne => [64, 2], type => $F32, data => pack('f<*', map { cos($_) } 1 .. 128)});
$_->{bytes} = length $_->{data} for @tensors;
my $in = File::Spec->catfile($dir, 'in.gguf');
{
    my ($h) = gguf_write_header(\@kv, \@tensors, 32);
    open my $f, '>:raw', $in or die $!;
    print {$f} $h;
    print {$f} $_->{data}, "\0" x ((-length $_->{data}) % 32) for @tensors;
    close $f;
}
my $g = gguf_read($in);
is_deeply([map { [$_->[0], $_->[1]] } @{$g->{kv}}], [map { [$_->[0], $_->[1]] } @kv], 'metadata keys and types read back');
is_deeply({map { $_->[0] => $_->[2] } @{$g->{kv}}}, {map { $_->[0] => $_->[2] } @kv}, 'metadata values read back');
is_deeply([map { [$_->{name}, $_->{type}, $_->{bytes}, @{$_->{ne}}] } @{$g->{tensors}}],
          [map { [$_->{name}, $_->{type}, $_->{bytes}, @{$_->{ne}}] } @tensors], 'tensor table read back');
open my $fh, '<:raw', $in or die $!;
is(gguf_read_rows($fh, $g->{tensors}[0], 1, 2), substr($tensors[0]{data}, 256, 512), 'rows 1 .. 2 of a tensor');
is(join('', map { $_->{offset} % 32 } @{$g->{tensors}}), '00000', 'tensor offsets aligned');

# rocm-convert (a GGUF in): the matrix encoded, the norm and the convolution kept, rows of 48 values (not blocks of 32) as F16,
# the LM head Q8_0 (recipe strix, the default)
my $out = File::Spec->catfile($dir, 'out.gguf');
my $script = File::Spec->catfile('bin', 'rocm-convert');
my $rc = system($^X, (map { "-I$_" } @INC), $script, '-q', '-r', 'plain', $in, "$out.plain", 'fast');
is((gguf_type_info(gguf_read("$out.plain")->{tensors}[4]{type}))[0], 'q4_0_rocmfp4_fast', 'recipe plain: the LM head FAST');
$rc = system($^X, (map { "-I$_" } @INC), $script, '-q', $in, $out, 'fast');
is($rc, 0, 'rocm-convert ran');
my $q = gguf_read($out);
is_deeply([map { (gguf_type_info($_->{type}))[0] } @{$q->{tensors}}], [qw(q4_0_rocmfp4_fast f32 f32 f16 q8_0)],
          'types: the matrix FAST, the norm and conv1d kept, 48-value rows F16, the LM head Q8_0');
open my $qh, '<:raw', $out or die $!;
is(gguf_read_rows($qh, $q->{tensors}[0], 0, 4), ROCm::Inference::quantize($FAST, $F32, $tensors[0]{data}, 4, 64),
   'the matrix: the encoder\'s bytes');
is(gguf_read_rows($qh, $q->{tensors}[1], 0, 1), $tensors[1]{data}, 'the norm: copied');
is(gguf_read_rows($qh, $q->{tensors}[4], 0, 2), ROCm::Inference::quantize($Q8_0, $F32, $tensors[4]{data}, 2, 64),
   'the LM head: the Q8_0 encoder\'s bytes');
my %m = map { $_->[0] => $_->[2] } @{$q->{kv}};
is($m{'general.file_type'}, 103, 'general.file_type 103 (Q4_0_ROCMFP4_FAST)');
is($m{'general.quantization_version'}, 2, 'general.quantization_version 2');
is($q->{kv}[-1][0], 'general.file_type', 'file_type last (as llama-quantize writes it)');
is($m{'t.strs'}[1][2], $kv[13][2][1][2], 'string arrays copied');
$rc = system($^X, (map { "-I$_" } @INC), $script, '-q', $in, $out, 'fast');
isnt($rc, 0, 'an existing output is not overwritten without --force');
$rc = system($^X, (map { "-I$_" } @INC), $script, '-q', '-t', 'attn_q=keep', '-t', 'ffn_up=bf16', '-f', $in, $out, 'q8_0');
is_deeply([map { (gguf_type_info($_->{type}))[0] } @{gguf_read($out)->{tensors}}], [qw(f32 f32 f32 bf16 q8_0)],
          '--type rules: keep, and BF16 (blocks of 1) for the 48-value rows') if $rc == 0;

done_testing;
