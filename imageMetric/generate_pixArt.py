# ============================================================
# PixArt-Alpha batch generation
#
# Uses EXISTING sampled dataset
# Precision: FP16
# CPU offload: enabled
# Seed: 42 for EVERY image
# ============================================================

import os

# MUST be set before importing torch.
# setdefault (not =) so wait_for_gpu.py can override this externally.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline

import pipeline_common as pc


# ============================================================
# CONFIG
# ============================================================

# Existing dataset, shared across all three models.
SAMPLED_PATH = Path("./sd3_output/sd3_sampled_prompts.csv")

OUTPUT_DIR = Path("./pixart_output")
IMAGE_DIR = OUTPUT_DIR / "pixart_images"
METADATA_PATH = OUTPUT_DIR / "pixart_generation_log.jsonl"

MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"

HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 4.5

# SAME SEED FOR EVERY IMAGE (matches FLUX/SD3 for a controlled comparison)
SEED = 42

# Rebuild the pipeline from scratch every N images — mitigates VRAM growth
# over long runs (see FLUX/PixArt note in pipeline_common.py).
RELOAD_EVERY = 100

REQUIRED_COLUMNS = {"id", "dataset_type", "name_of_work", "content", "source"}


# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GPU CHECK
# ============================================================

pc.print_gpu_info()


# ============================================================
# LOAD EXISTING DATASET
# ============================================================

print("=" * 70)
print("LOADING EXISTING SAMPLED DATASET")
print("=" * 70)

sampled = pc.load_sampled_csv(SAMPLED_PATH, REQUIRED_COLUMNS)

print(f"Seed for ALL images: {SEED}")
print()


# ============================================================
# PIPELINE
# ============================================================

def load_pipeline():
    print("=" * 70)
    print("LOADING PIXART")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print("Precision: FP16")
    print("Resolution:", f"{WIDTH}x{HEIGHT}")
    print("Seed:", SEED)
    print()

    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

    # CPU offloading is important for a 24 GB GPU. This was previously
    # MISSING here (unlike FLUX/SD3), which left the full model
    # permanently resident on GPU — pixart.log shows every single image
    # OOMing once the shared card had other tenants, because there was
    # no way to free any of PixArt's own footprint between images.
    pipe.enable_model_cpu_offload()

    # Attention slicing if available.
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    print("PixArt pipeline loaded.")
    print()

    return pipe


def generate_one(pipe, prompt, generator):
    result = pipe(
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    return result.images[0]


def build_record_extra(row, elapsed):
    return {
        "dataset_type": row["dataset_type"],
        "name_of_work": row["name_of_work"],
        "source": row["source"],
        "model": MODEL_ID,
        "precision": "fp16",
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "generation_time_seconds": elapsed,
    }


# ============================================================
# FIND ALREADY GENERATED IMAGES
# ============================================================

remaining, already_done = pc.compute_remaining(sampled, IMAGE_DIR, METADATA_PATH)


# ============================================================
# GENERATE
# ============================================================

stats = pc.run_generation_loop(
    remaining,
    load_pipeline=load_pipeline,
    generate_one=generate_one,
    build_record_extra=build_record_extra,
    image_dir=IMAGE_DIR,
    metadata_path=METADATA_PATH,
    seed=SEED,
    # Was device="cuda" — not guaranteed to reproduce identical noise
    # across different GPU architectures (A5000 vs 3090).
    generator_device="cpu",
    reload_every=RELOAD_EVERY,
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("PIXART BATCH RUN COMPLETE")
print("=" * 70)
print(f"Generated:    {stats.generated}")
print(f"OOM skipped:  {stats.oom_skipped}")
print(f"Other failed: {stats.other_failed}")
print(f"Reloads:      {stats.reloads}")
print(f"Images: {IMAGE_DIR.resolve()}")
print(f"Log: {METADATA_PATH.resolve()}")
print(f"Every image used seed: {SEED}")
print()
