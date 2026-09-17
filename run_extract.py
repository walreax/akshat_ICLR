"""
run_extract.py
===============
Runs the Concept Fidelity Score data extraction for one model, to
completion, automatically: the calibration set first (for the contrastive
concept->latent mapping), then the 600-prompt test set. Each is a single
hooked generation per prompt with --save-latents, so one pass yields
everything CFS needs (mean activation of every latent + top-k spatial
maps, per block/timestep). Crashes are retried with --resume, so nothing
is recomputed and nothing is lost.

Usage:
    python run_extract.py sd3
    python run_extract.py pixart

That's the only argument. Everything else is a constant below.
"""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

DATASETS = [  # (name, path) -- calibration first, test second
    ("calib", os.path.join(DATA_DIR, "calib_set_600.csv")),
    ("test", os.path.join(DATA_DIR, "test_set_600.csv")),
]

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

# Pre-registered in writeups/metric_spec.html: all five SAE blocks are
# extracted (cheap), CFS uses {8,12,16,20}; timesteps mid-to-late, t=0.0
# dropped on the latent-collapse evidence.
SAE_BLOCKS = "4,8,12,16,20"
TIMESTEPS = "0.5,0.75,1.0"
SEEDS = "42"
HEATMAP_SAMPLE_RATE = "0.02"   # a dozen visual spot-checks per run; disk is tight
LATENT_TOP_K = "32"
LATENT_MAP_SIZE = "32"

RETRY_DELAY_SECONDS = 20
MAX_ATTEMPTS = 200


def build_command(model: str, dataset_path: str, out_dir: str) -> list:
    cfg = MODELS[model]
    return [
        sys.executable, cfg["script"], *cfg["extra"],
        "--load-sae",
        "--dataset", dataset_path,
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


def run_to_completion(model: str, name: str, dataset_path: str) -> None:
    out_dir = os.path.join(SCRIPT_DIR, f"cfs_{model}_{name}")
    print(f"\n[run_extract] {model} / {name}: dataset={dataset_path}\n[run_extract] out_dir={out_dir}")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[run_extract] {model}/{name} attempt {attempt}/{MAX_ATTEMPTS} starting...")
        start = time.time()
        try:
            result = subprocess.run(build_command(model, dataset_path, out_dir))
        except KeyboardInterrupt:
            print("\n[run_extract] stopped by user (Ctrl+C). Progress is saved; rerun to pick back up.")
            sys.exit(130)
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"[run_extract] {model}/{name} finished after {attempt} attempt(s), {elapsed:.0f}s on this attempt.")
            return
        print(f"[run_extract] {model}/{name} attempt {attempt} exited with code {result.returncode} "
              f"after {elapsed:.0f}s -- retrying in {RETRY_DELAY_SECONDS}s\n")
        time.sleep(RETRY_DELAY_SECONDS)
    print(f"\n[run_extract] {model}/{name} gave up after {MAX_ATTEMPTS} attempts -- something structural is wrong.")
    sys.exit(1)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        print(f"usage: python run_extract.py {{{'|'.join(MODELS)}}}")
        sys.exit(2)
    model = sys.argv[1]
    for name, path in DATASETS:
        run_to_completion(model, name, path)
    print(f"\n[run_extract] {model}: calibration + test extraction complete.")


if __name__ == "__main__":
    main()
