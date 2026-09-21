# ================================================================
# FLUX.1-DEV — NON-QUANTIZED BASELINE
# ================================================================
#
# Hardware:
#   RTX A5000 24 GB
#
# Model:
#   black-forest-labs/FLUX.1-dev
#
# Precision:
#   BF16
#
# Quantization:
#   NONE
#
# GPU:
#   Physical GPU 1
#
# Offloading:
#   CPU offload enabled
#
# Project:
#   /DATA/swagata/akshat_ICLR/
#
# ================================================================


# ================================================================
# 1. GPU SELECTION
# ================================================================
#
# GPU 0 is being used by another process.
# Physical GPU 1 is the GPU allocated to us.
#
# CUDA_VISIBLE_DEVICES makes physical GPU 1 appear as cuda:0
# inside this Python process.
# ================================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ================================================================
# 2. IMPORTS
# ================================================================

import gc

import torch

from diffusers import FluxPipeline


# ================================================================
# 3. CONFIGURATION
# ================================================================

MODEL_ID = (
    "black-forest-labs/FLUX.1-dev"
)


# ------------------------------------------------
# Project directories
# ------------------------------------------------

PROJECT_DIR = (
    "/DATA/swagata/akshat_ICLR"
)


RESULTS_DIR = (
    os.path.join(
        PROJECT_DIR,
        "results"
    )
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ------------------------------------------------
# Output file
# ------------------------------------------------

OUTPUT_PATH = (
    os.path.join(
        RESULTS_DIR,
        "flux_baseline.png"
    )
)


# ------------------------------------------------
# Prompt
# ------------------------------------------------

PROMPT = (
    "a gilded hourglass trapping a bird, "
    "symbolizing fleeting time and mortality, "
    "surrealist cinematic artwork, "
    "dark atmospheric lighting, "
    "highly detailed, "
    "dramatic composition"
)


# ------------------------------------------------
# Generation parameters
# ------------------------------------------------

SEED = 42

HEIGHT = 512

WIDTH = 512

NUM_INFERENCE_STEPS = 28

GUIDANCE_SCALE = 3.5


# ================================================================
# 4. SYSTEM INFORMATION
# ================================================================

print()
print("=" * 70)
print("FLUX.1-DEV")
print("=" * 70)

print()

print(
    "Project:",
    PROJECT_DIR
)

print(
    "Model:",
    MODEL_ID
)

print(
    "Precision: BF16"
)

print(
    "Quantization: NONE"
)

print(
    "CPU offload: ENABLED"
)

print()


# ================================================================
# 5. CUDA CHECK
# ================================================================

print("=" * 70)
print("GPU CHECK")
print("=" * 70)

print()

print(
    "PyTorch:",
    torch.__version__
)

print(
    "PyTorch CUDA:",
    torch.version.cuda
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "\nCUDA is not available.\n"
        "Check your PyTorch installation."
    )


# ================================================================
# 6. GPU INFORMATION
# ================================================================

GPU_NAME = (
    torch.cuda.get_device_name(0)
)


GPU_PROPERTIES = (
    torch.cuda.get_device_properties(0)
)


GPU_MEMORY_GB = (
    GPU_PROPERTIES.total_memory
    /
    (1024 ** 3)
)


print()

print(
    "GPU:",
    GPU_NAME
)

print(
    "VRAM:",
    f"{GPU_MEMORY_GB:.2f} GB"
)

print()


# ================================================================
# 7. LOAD FLUX
# ================================================================

print("=" * 70)
print("LOADING FLUX.1-DEV")
print("=" * 70)

print()

print(
    "This is the full-precision/BF16 model."
)

print(
    "No 4-bit quantization is being used."
)

print()

print(
    "Loading model..."
)


pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
)


print()

print(
    "Model loaded successfully."
)


# ================================================================
# 8. ENABLE CPU OFFLOADING
# ================================================================
#
# The A5000 has 24 GB VRAM.
#
# FLUX.1-dev is large enough that putting the entire pipeline
# directly onto the GPU is undesirable.
#
# CPU offloading moves components between CPU and GPU as needed.
# ================================================================

print()

print("=" * 70)
print("ENABLING CPU OFFLOAD")
print("=" * 70)

print()


pipe.enable_model_cpu_offload()


print(
    "CPU offloading enabled."
)


# ================================================================
# 9. GENERATION INFORMATION
# ================================================================

print()

print("=" * 70)
print("GENERATION SETTINGS")
print("=" * 70)

print()

print(
    "Prompt:"
)

print(
    PROMPT
)

print()

print(
    "Resolution:",
    f"{WIDTH} x {HEIGHT}"
)

print(
    "Inference steps:",
    NUM_INFERENCE_STEPS
)

print(
    "Guidance scale:",
    GUIDANCE_SCALE
)

print(
    "Seed:",
    SEED
)

print()


# ================================================================
# 10. RANDOM GENERATOR
# ================================================================

generator = (
    torch.Generator(
        device="cpu"
    )
    .manual_seed(
        SEED
    )
)


# ================================================================
# 11. GENERATE IMAGE
# ================================================================

print("=" * 70)
print("GENERATING IMAGE")
print("=" * 70)

print()


with torch.inference_mode():

    result = pipe(

        prompt=PROMPT,

        height=HEIGHT,

        width=WIDTH,

        num_inference_steps=(
            NUM_INFERENCE_STEPS
        ),

        guidance_scale=(
            GUIDANCE_SCALE
        ),

        generator=generator
    )


# ================================================================
# 12. EXTRACT IMAGE
# ================================================================

image = (
    result.images[0]
)


# ================================================================
# 13. SAVE IMAGE
# ================================================================

print()

print("=" * 70)
print("SAVING IMAGE")
print("=" * 70)

print()

print(
    "Output:",
    OUTPUT_PATH
)


image.save(
    OUTPUT_PATH
)


print()

print(
    "Image saved successfully."
)


# ================================================================
# 14. GPU MEMORY INFORMATION
# ================================================================

print()

print("=" * 70)
print("GPU MEMORY")
print("=" * 70)

print()


allocated_gb = (
    torch.cuda.memory_allocated(0)
    /
    (1024 ** 3)
)


reserved_gb = (
    torch.cuda.memory_reserved(0)
    /
    (1024 ** 3)
)


max_allocated_gb = (
    torch.cuda.max_memory_allocated(0)
    /
    (1024 ** 3)
)


print(
    "Currently allocated:",
    f"{allocated_gb:.2f} GB"
)

print(
    "Currently reserved:",
    f"{reserved_gb:.2f} GB"
)

print(
    "Maximum allocated:",
    f"{max_allocated_gb:.2f} GB"
)


# ================================================================
# 15. FINAL STATUS
# ================================================================

print()

print("=" * 70)
print("COMPLETE")
print("=" * 70)

print()

print(
    "Model:",
    MODEL_ID
)

print(
    "GPU:",
    GPU_NAME
)

print(
    "Precision: BF16"
)

print(
    "Quantization: NONE"
)

print(
    "Seed:",
    SEED
)

print(
    "Steps:",
    NUM_INFERENCE_STEPS
)

print(
    "Resolution:",
    f"{WIDTH} x {HEIGHT}"
)

print()

print(
    "Image:",
    OUTPUT_PATH
)

print()


# ================================================================
# 16. CLEANUP
# ================================================================

gc.collect()

torch.cuda.empty_cache()


print(
    "GPU cache cleared."
)

print()