"""
run_pixart.py
==============
Runs pixart_activation_extractor.py to completion, automatically, testing
the already-trained PixArt SAEs (sae_checkpoints_provided/, d_in=1152)
against the same 600-prompt benchmark set used for the SD3/FLUX/PixArt
CLIP/BLIP-2/VQAScore/Inception scoring.

No config to pass on the command line -- everything is set as a constant
below. If the underlying process dies for any reason (OOM, a crashed
shared GPU, an environment hiccup), this just relaunches it with --resume
and keeps going; evaluate_concepts' own resume logic (checks
concept_metrics.csv for which (concept, seed) pairs are already done)
means nothing gets recomputed and nothing gets lost.

Usage:
    python run_pixart.py

That's it. Edit the constants below if you want different settings; there
is deliberately no CLI to memorize.
"""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIXART_SCRIPT = os.path.join(SCRIPT_DIR, "pixart_activation_extractor.py")

DATASET = "/DATA/swagata/imageMetric/sd3_output/sd3_sampled_prompts.csv"  # identical to pixart_sampled_prompts.csv (same md5)
SAE_DIR = os.path.join(SCRIPT_DIR, "sae_checkpoints_provided")
OUT_DIR = os.path.join(SCRIPT_DIR, "pixart_benchmark_full")

SAE_BLOCKS = "4,8,12,16,20"
SEEDS = "42"                  # matches the seed the benchmark images were generated with
HEATMAP_SAMPLE_RATE = "1.0"   # save every heatmap image

RETRY_DELAY_SECONDS = 20
MAX_ATTEMPTS = 200


def build_command() -> list:
    return [
        sys.executable, PIXART_SCRIPT,
        "--load-sae",
        "--dataset", DATASET,
        "--sae-blocks", SAE_BLOCKS,
        "--sae-dir", SAE_DIR,
        "--seeds", SEEDS,
        "--heatmap-sample-rate", HEATMAP_SAMPLE_RATE,
        "--out-dir", OUT_DIR,
        "--resume",
    ]


def main():
    print(f"[run_pixart] dataset={DATASET}")
    print(f"[run_pixart] sae_dir={SAE_DIR}")
    print(f"[run_pixart] out_dir={OUT_DIR}")
    print(f"[run_pixart] will retry automatically on crash, up to {MAX_ATTEMPTS} attempts\n")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[run_pixart] attempt {attempt}/{MAX_ATTEMPTS} starting...")
        start = time.time()
        try:
            result = subprocess.run(build_command())
        except KeyboardInterrupt:
            print("\n[run_pixart] stopped by user (Ctrl+C). Progress so far is saved; "
                  "just run `python run_pixart.py` again to pick back up.")
            sys.exit(130)

        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"\n[run_pixart] finished successfully after {attempt} attempt(s), "
                  f"{elapsed:.0f}s on this attempt.")
            print(f"[run_pixart] results in {OUT_DIR}/concept_metrics.csv")
            return

        print(f"[run_pixart] attempt {attempt} exited with code {result.returncode} "
              f"after {elapsed:.0f}s -- retrying in {RETRY_DELAY_SECONDS}s "
              f"(already-completed rows are safe, --resume will skip them)\n")
        time.sleep(RETRY_DELAY_SECONDS)

    print(f"\n[run_pixart] gave up after {MAX_ATTEMPTS} attempts. This is no longer a "
          f"transient crash -- something structural is wrong (bad dataset path, "
          f"missing SAE checkpoints, no GPU available at all). Check the last "
          f"attempt's output above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
