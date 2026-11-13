# Images (Stable Diffusion, sd-turbo, PLAN.md I5): the PNG of an image at a small size, and the same seed the same bytes.
# Env-gated: ROCM_INFERENCE_TEST_SD names the image model's GGUF (rocm-convert of the diffusers pipeline).
use strict;
use warnings;
use Test::More;
use ROCm::Inference;

my $gguf = $ENV{ROCM_INFERENCE_TEST_SD};
plan skip_all => 'set ROCM_INFERENCE_TEST_SD to run the image test' unless $gguf && -f $gguf;

my $img = ROCm::Inference::sd_load($gguf);
my $png = $img->generate('a photo of a red fox in the snow', size => '128x128', seed => 7);
is(substr($png, 0, 8), "\x89PNG\r\n\x1a\n", 'the PNG signature');
is(substr($png, 12, 4), 'IHDR', 'the header chunk');
is(unpack('N', substr($png, 16, 4)), 128, 'the width');
is(unpack('N', substr($png, 20, 4)), 128, 'the height');
my $again = $img->generate('a photo of a red fox in the snow', size => '128x128', seed => 7);
is($again, $png, 'the same seed, the same bytes');
my $other = $img->generate('a photo of a red fox in the snow', size => '128x128', seed => 8);
isnt($other, $png, 'another seed, another image');
eval { $img->generate('x', size => '100x100') };
like($@, qr/multiples of 8/, 'a size that is not a multiple of 8 is refused');

done_testing;
