# ============================================================
# SD3 Medium batch generation
#
# Precision: FP16
# T5: enabled (was disabled to save VRAM, but that truncated prompts to
#              77 tokens via CLIP-only encoding — vs FLUX's 256 and
#              PixArt's ~120 tokens via T5 — an unfair comparison for a
#              paper measuring cross-model image quality)
# CPU offload: enabled
# Seed: 42 for EVERY image (unified with FLUX/PixArt)
# ============================================================

import os

# MUST be set before importing torch.
# setdefault (not =) so wait_for_gpu.py can override this externally.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline

import pipeline_common as pc


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("./dataset")
OUTPUT_DIR = Path("./sd3_output")

POEM_CSV = DATA_DIR / "poem.csv"
STORY_CSV = DATA_DIR / "story.csv"

N_POEM_SAMPLES = 200
N_STORY_SAMPLES = 400
SAMPLE_SEED = 43

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# T5 re-enabled: see header note above. enable_model_cpu_offload() keeps
# this within a 24GB budget by moving it to CPU when not in use.
INCLUDE_T5 = True

HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.0

# SAME SEED FOR EVERY IMAGE (was BASE_GEN_SEED + row index — unified with
# FLUX/PixArt so the only thing varying across models is the model itself).
SEED = 42

# Rebuild the pipeline from scratch every N images — mitigates VRAM growth
# over long runs (see FLUX/PixArt note in pipeline_common.py).
RELOAD_EVERY = 100

IMAGE_DIR = OUTPUT_DIR / "sd3_images"
METADATA_PATH = OUTPUT_DIR / "sd3_generation_log.jsonl"

SAMPLED_COLUMNS = {"id", "dataset_type", "name_of_work", "content", "source"}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GPU CHECK
# ============================================================

pc.print_gpu_info()


# ============================================================
# LOAD OR CREATE THE SAMPLED PROMPT SET
# ============================================================
# This is the shared dataset FLUX/PixArt also read from. If it already
# exists (the normal case — it's generated once), reuse it as-is instead
# of resampling, so re-running this script doesn't require poem.csv /
# story.csv to still be present and doesn't touch the shared file.

sampled_path = OUTPUT_DIR / "sd3_sampled_prompts.csv"

if sampled_path.exists():

    print("=" * 60)
    print("REUSING EXISTING SAMPLED PROMPTS")
    print("=" * 60)
    print(f"Found: {sampled_path.resolve()}")
    print()

    sampled = pc.load_sampled_csv(sampled_path, SAMPLED_COLUMNS)

else:

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    if not POEM_CSV.exists():
        raise FileNotFoundError(f"Could not find: {POEM_CSV.resolve()}")
    if not STORY_CSV.exists():
        raise FileNotFoundError(f"Could not find: {STORY_CSV.resolve()}")

    poem_df = pd.read_csv(POEM_CSV)
    story_df = pd.read_csv(STORY_CSV)

    print(f"poem.csv:  {len(poem_df)} rows")
    print(f"story.csv: {len(story_df)} rows")

    required_raw_columns = {"name_of_work", "content", "source"}

    for name, df, required_n in [
        ("poem.csv", poem_df, N_POEM_SAMPLES),
        ("story.csv", story_df, N_STORY_SAMPLES),
    ]:
        missing = required_raw_columns - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{name} is missing columns: {sorted(missing)}. "
                f"Found columns: {list(df.columns)}"
            )

        if len(df) < required_n:
            raise RuntimeError(
                f"{name} has only {len(df)} rows, "
                f"but {required_n} are required."
            )

    poem_sample = poem_df.sample(n=N_POEM_SAMPLES, random_state=SAMPLE_SEED).copy()
    story_sample = story_df.sample(n=N_STORY_SAMPLES, random_state=SAMPLE_SEED).copy()

    poem_sample["dataset_type"] = "poem"
    story_sample["dataset_type"] = "story"

    sampled = pd.concat([poem_sample, story_sample], ignore_index=True)

    sampled["id"] = [
        f"{row.dataset_type}_{i:04d}"
        for i, row in enumerate(sampled.itertuples())
    ]

    sampled = sampled[["id", "dataset_type", "name_of_work", "content", "source"]]

    sampled.to_csv(sampled_path, index=False)

    print()
    print(
        f"Sampled {len(poem_sample)} poems + "
        f"{len(story_sample)} stories = {len(sampled)} total"
    )
    print(f"Saved sample list -> {sampled_path.resolve()}")
    print()

print(f"Seed for ALL images: {SEED}")
print()


# ============================================================
# PIPELINE
# ============================================================

def load_pipeline():
    print("=" * 60)
    print("LOADING SD3 MEDIUM")
    print("=" * 60)
    print("Precision: FP16")
    print("T5 enabled:", INCLUDE_T5)
    print("CPU offload: enabled")
    print()

    kwargs = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
    }

    # Disable the third text encoder (T5) only if explicitly configured.
    if not INCLUDE_T5:
        kwargs["text_encoder_3"] = None
        kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, **kwargs)

    # This moves parts of the model between CPU/GPU automatically.
    pipe.enable_model_cpu_offload()

    print("SD3 pipeline loaded.")
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
        "include_t5": INCLUDE_T5,
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
# DONE
# ============================================================

print("=" * 60)
print("SD3 BATCH RUN COMPLETE")
print("=" * 60)
print(f"Generated:    {stats.generated}")
print(f"OOM skipped:  {stats.oom_skipped}")
print(f"Other failed: {stats.other_failed}")
print(f"Reloads:      {stats.reloads}")
print(f"Images:  {IMAGE_DIR.resolve()}")
print(f"Log:     {METADATA_PATH.resolve()}")
print()
