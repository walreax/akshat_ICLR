# ============================================================
# FLUX.1-schnell batch generation
#
# Uses EXISTING SD3 sampled dataset
# Precision: bfloat16 (fp16 is known to NaN/degrade on FLUX's transformer)
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
from diffusers import FluxPipeline

import pipeline_common as pc


# ============================================================
# CONFIG
# ============================================================

# Existing SD3-sampled dataset, shared across all three models.
SAMPLED_PATH = Path("./sd3_output/sd3_sampled_prompts.csv")

OUTPUT_DIR = Path("./flux_output")
IMAGE_DIR = OUTPUT_DIR / "flux_images"
METADATA_PATH = OUTPUT_DIR / "flux_generation_log.jsonl"

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

HEIGHT = 1024
WIDTH = 1024

# schnell is a distilled, guidance-free model: 4 steps, guidance_scale=0.
# It also caps at 256 text tokens (passing >256 raises ValueError).
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
MAX_SEQUENCE_LENGTH = 256

# SAME SEED FOR EVERY IMAGE (matches PixArt/SD3 for a controlled comparison)
SEED = 42

# Rebuild the pipeline from scratch every N images — mitigates the VRAM
# growth observed in FLUX.log (OOM starting ~image 298/600 with a
# steady ~11.6-12 GiB "in use" baseline that del+gc+empty_cache alone
# didn't reclaim).
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
# LOAD EXISTING SD3 DATASET
# ============================================================

print("=" * 70)
print("LOADING EXISTING SD3 DATASET")
print("=" * 70)

sampled = pc.load_sampled_csv(SAMPLED_PATH, REQUIRED_COLUMNS)

print(f"Seed for ALL images: {SEED}")
print()


# ============================================================
# PIPELINE
# ============================================================

def load_pipeline():
    print("=" * 70)
    print("LOADING FLUX.1-SCHNELL")
    print("=" * 70)
    print("Model:", MODEL_ID)
    print("Precision: bfloat16")
    print("Resolution:", f"{WIDTH}x{HEIGHT}")
    print("Seed:", SEED)
    print()

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    # enable_model_cpu_offload() keeps the entire ~12B-param transformer
    # resident on GPU while active. At 1024x1024 that peak was observed
    # to hit ~17.4 GiB on a shared 24GB card -- fine when the card is
    # otherwise empty, but leaves no room at all once another tenant's
    # job is present (confirmed: every image OOM'd once a co-resident
    # 6.25GB job showed up). enable_sequential_cpu_offload() streams
    # individual sub-layers instead, trading some throughput for a much
    # lower peak -- the same choice train_sae.py already makes.
    pipe.enable_sequential_cpu_offload()

    # Reduce VAE decode peak memory at 1024x1024 -- cheap, no quality cost.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("FLUX pipeline loaded.")
    print()

    return pipe


def generate_one(pipe, prompt, generator):
    result = pipe(
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        generator=generator,
    )
    return result.images[0]


def build_record_extra(row, elapsed):
    return {
        "dataset_type": row["dataset_type"],
        "name_of_work": row["name_of_work"],
        "source": row["source"],
        "model": MODEL_ID,
        "precision": "bf16",
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "generation_time_seconds": elapsed,
    }


# ============================================================
# FIND ALREADY GENERATED IMAGES
# ============================================================

remaining, already_done = pc.compute_remaining(sampled, IMAGE_DIR, METADATA_PATH)


# ============================================================
# GENERATE IMAGES
# ============================================================

stats = pc.run_generation_loop(
    remaining,
    load_pipeline=load_pipeline,
    generate_one=generate_one,
    build_record_extra=build_record_extra,
    image_dir=IMAGE_DIR,
    metadata_path=METADATA_PATH,
    seed=SEED,
    reload_every=RELOAD_EVERY,
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("FLUX BATCH RUN COMPLETE")
print("=" * 70)
print(f"Generated:    {stats.generated}")
print(f"OOM skipped:  {stats.oom_skipped}")
print(f"Other failed: {stats.other_failed}")
print(f"Reloads:      {stats.reloads}")
print(f"Images: {IMAGE_DIR.resolve()}")
print(f"Log: {METADATA_PATH.resolve()}")
print(f"Every image used seed: {SEED}")
print()
