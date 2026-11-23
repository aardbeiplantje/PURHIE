import os
import sys
os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'true'
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '0'
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
#os.environ['LLVM_PATH'] = os.environ['ROCM_PATH']
os.environ['HSA_OVERRIDE_GFX_VERSION']='11.5.1'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'
os.environ['MIOPEN_USER_DB_PATH'] = '/tmp/miopen_db'
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['HSA_XNACK'] = '0'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
import torch
torch.backends.cudnn.enabled = False
from diffusers import ZImagePipeline

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

# 2. Generate Image
# The starting latent is drawn on the CPU with torch.manual_seed(42), as ComfyUI draws it, so the result can be compared
# with ComfyUI's image for the same seed (drawing it on the GPU gives different noise).
noise = torch.randn((1, 16, 128, 128), generator=torch.Generator("cpu").manual_seed(42))  # float32, as ComfyUI's
# the initial noise as raw float32 (16 x 128 x 128): the engine reads it with ZI_NOISE=<file>, so both start from the same latent
noise.numpy().astype("float32").tofile("initial_noise_seed42.f32")
latents = noise.to("cuda", torch.bfloat16)
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    latents=latents,
).images[0]

image.save("example_cpu_noise.png")
