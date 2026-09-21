"""
run_sd3.py
==========
Trains fresh SAEs on train_set_4200.csv and runs sd_activation_extractor.py
to completion, automatically. No config to pass on the command line --
everything is set as a constant below. If the underlying process dies for
any reason (OOM, a crashed shared GPU, an environment hiccup), this just
relaunches it and keeps going.

Usage:
    python run_sd3.py

That's it. Edit the constants below if you want different settings; there
is deliberately no CLI to memorize.

Two different things get "resumed" differently on a crash-and-relaunch:
  - The EVALUATION phase (after SAEs are trained) tracks completed rows in
    concept_metrics.csv and --resume skips them, so it never redoes work.
  - The COLLECTION phase (gathering activations to train the SAEs on) has
    no such checkpoint -- each relaunch collects from scratch. This is
    still bounded and fast (it stops once the token budget per block is
    full, not when the whole dataset is exhausted), so a relaunch here
    costs a few minutes, not hours.
"""

import os
import subprocess
import sys
import time

# ============================================================
# Everything that used to be a CLI flag. Edit here, not on the
# command line.
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SD_SCRIPT = os.path.join(SCRIPT_DIR, "sd_activation_extractor.py")

DATASET = os.path.join(SCRIPT_DIR, "train_set_4200.csv")
# A different directory from your existing sae_checkpoints/ -- this trains
# fresh SAEs from scratch on train_set_4200.csv and saves them here, rather
# than overwriting the checkpoints you already have.
SAE_DIR = os.path.join(SCRIPT_DIR, "sae_checkpoints_trained")
OUT_DIR = os.path.join(SCRIPT_DIR, "sd3_train_full")

SAE_BLOCKS = "4,8,12,16,20"
SEEDS = "42"
HEATMAP_SAMPLE_RATE = "1.0"   # save every heatmap image

# How long to wait before relaunching after a crash, and how many times to
# try before giving up and telling you it's genuinely stuck (as opposed to
# just hitting a transient OOM/contention blip).
RETRY_DELAY_SECONDS = 20
MAX_ATTEMPTS = 200


def build_command() -> list:
    return [
        sys.executable, SD_SCRIPT,
        "--mode", "sae",
        # no --load-sae: this trains fresh SAEs from scratch on DATASET
        "--dataset", DATASET,
        "--sae-blocks", SAE_BLOCKS,
        "--sae-dir", SAE_DIR,
        "--seeds", SEEDS,
        "--heatmap-sample-rate", HEATMAP_SAMPLE_RATE,
        "--out-dir", OUT_DIR,
        "--resume",
    ]


def main():
    print(f"[run_sd3] dataset={DATASET}")
    print(f"[run_sd3] sae_dir={SAE_DIR}")
    print(f"[run_sd3] out_dir={OUT_DIR}")
    print(f"[run_sd3] will retry automatically on crash, up to {MAX_ATTEMPTS} attempts\n")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[run_sd3] attempt {attempt}/{MAX_ATTEMPTS} starting...")
        start = time.time()
        try:
            result = subprocess.run(build_command())
        except KeyboardInterrupt:
            print("\n[run_sd3] stopped by user (Ctrl+C). Progress so far is saved; "
                  "just run `python run_sd3.py` again to pick back up.")
            sys.exit(130)

        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"\n[run_sd3] finished successfully after {attempt} attempt(s), "
                  f"{elapsed:.0f}s on this attempt.")
            print(f"[run_sd3] results in {OUT_DIR}/concept_metrics.csv")
            return

        print(f"[run_sd3] attempt {attempt} exited with code {result.returncode} "
              f"after {elapsed:.0f}s -- retrying in {RETRY_DELAY_SECONDS}s "
              f"(already-completed rows are safe, --resume will skip them)\n")
        time.sleep(RETRY_DELAY_SECONDS)

    print(f"\n[run_sd3] gave up after {MAX_ATTEMPTS} attempts. This is no longer a "
          f"transient crash -- something structural is wrong (bad dataset path, "
          f"missing SAE checkpoints, no GPU available at all). Check the last "
          f"attempt's output above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
