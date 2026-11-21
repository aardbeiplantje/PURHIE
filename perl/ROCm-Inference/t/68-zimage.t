# Z-Image-Turbo (text to image, PLAN.md Z-Image): the PNG of an image at 256 x 256, the same seed the same bytes, another seed
# another image. Env-gated: ROCM_INFERENCE_TEST_ZIMAGE names a directory in ComfyUI's layout (diffusion_models/, vae/,
# text_encoders/qwen_3_4b.safetensors); ROCM_INFERENCE_TEST_ZIMAGE_TOKENIZER names the Qwen3 tokenizer.json (ComfyUI ships none).
use strict;
use warnings;
use Test::More;
use ROCm::Inference;

my $dir = $ENV{ROCM_INFERENCE_TEST_ZIMAGE};
my $tok = $ENV{ROCM_INFERENCE_TEST_ZIMAGE_TOKENIZER};
plan skip_all => 'set ROCM_INFERENCE_TEST_ZIMAGE and ROCM_INFERENCE_TEST_ZIMAGE_TOKENIZER to run the Z-Image test'
    unless $dir && -d $dir && $tok && -f $tok;

my $img = ROCm::Inference::ZImage->new($dir, tokenizer => $tok);
my $prompt = 'a red square on a blue background';
my $png = $img->generate($prompt, size => '256x256', seed => 0);
is(substr($png, 0, 8), "\x89PNG\r\n\x1a\n", 'the PNG signature');
is(substr($png, 12, 4), 'IHDR', 'the header chunk');
is(unpack('N', substr($png, 16, 4)), 256, 'the width');
is(unpack('N', substr($png, 20, 4)), 256, 'the height');
my $again = $img->generate($prompt, size => '256x256', seed => 0);
is($again, $png, 'the same seed, the same bytes');
my $other = $img->generate($prompt, size => '256x256', seed => 1);
isnt($other, $png, 'another seed, another image');
eval { $img->generate($prompt, size => '128x128') };
like($@, qr/256x256 only/, 'another size is refused for now');

done_testing;
