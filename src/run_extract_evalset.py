"""
run_extract_evalset.py
========================
Same as run_extract.py, but points at the SAME 600-prompt set already
used for image generation + the real CLIP/BLIP2/VQAScore/Inception Score
evaluation in imageMetric/evaluation_output/ (imageMetric/sd3_output/
sd3_sampled_prompts.csv -- identical content to pixart's copy, verified).
Running extraction against this exact set (instead of our separate
calib_set_600/test_set_600 dev/test split) means centroid deviation and
attention-concentration can be directly compared against the real
external-metric scores we already have for these 600 images.

Usage:
    python run_extract_evalset.py sd3
    python run_extract_evalset.py pixart
"""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "imageMetric", "sd3_output", "sd3_sampled_prompts.csv")
DATASET_PATH = os.path.normpath(DATASET_PATH)

MODELS = {
    "sd3": {
        "script": os.path.join(SCRIPT_DIR, "sd_activation_extractor.py"),
        "extra": ["--mode", "sae"],
        "sae_dir": os.path.join(SCRIPT_DIR, "sae_checkpoints"),
    },
    "pixart": {
        "script": os.path.join(SCRIPT_DIR, "pixart_activation_extractor.py"),
        "extra": [],
        "sae_dir": os.path.join(SCRIPT_DIR, "sae_checkpoints_provided"),
    },
}

SAE_BLOCKS = "4,8,12,16,20"
TIMESTEPS = "0.5,0.75,1.0"
SEEDS = "42"
HEATMAP_SAMPLE_RATE = "0.02"
LATENT_TOP_K = "32"
LATENT_MAP_SIZE = "32"

RETRY_DELAY_SECONDS = 20
MAX_ATTEMPTS = 200


def build_command(model: str, out_dir: str) -> list:
    cfg = MODELS[model]
    return [
        sys.executable, cfg["script"], *cfg["extra"],
        "--load-sae",
        "--dataset", DATASET_PATH,
        "--sae-blocks", SAE_BLOCKS,
        "--sae-dir", cfg["sae_dir"],
        "--seeds", SEEDS,
        "--timesteps", TIMESTEPS,
        "--shuffle-seed", "-1",
        "--heatmap-sample-rate", HEATMAP_SAMPLE_RATE,
        "--save-latents",
        "--latent-top-k", LATENT_TOP_K,
        "--latent-map-size", LATENT_MAP_SIZE,
        "--out-dir", out_dir,
        "--resume",
    ]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        print(f"usage: python run_extract_evalset.py {{{'|'.join(MODELS)}}}")
        sys.exit(2)
    model = sys.argv[1]
    out_dir = os.path.join(SCRIPT_DIR, f"cfs_{model}_evalset600")
    print(f"[run_extract_evalset] {model}: dataset={DATASET_PATH}\n[run_extract_evalset] out_dir={out_dir}")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[run_extract_evalset] {model} attempt {attempt}/{MAX_ATTEMPTS} starting...")
        start = time.time()
        result = subprocess.run(build_command(model, out_dir))
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"[run_extract_evalset] {model} finished after {attempt} attempt(s), {elapsed:.0f}s on this attempt.")
            return
        print(f"[run_extract_evalset] {model} attempt {attempt} exited {result.returncode} after {elapsed:.0f}s -- retrying in {RETRY_DELAY_SECONDS}s\n")
        time.sleep(RETRY_DELAY_SECONDS)
    print(f"\n[run_extract_evalset] {model} gave up after {MAX_ATTEMPTS} attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
