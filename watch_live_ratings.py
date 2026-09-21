"""
watch_live_ratings.py
======================
Polls the human-eval app's live results over SSH and reprints a status
snapshot every few seconds: total vs. valid (post-fix) row counts,
per-annotator breakdown, and the most recent submissions.

Runs locally (Windows) -- shells out to ssh each tick rather than
requiring `watch` (Linux-only) or a persistent SSH session, so it works
the same way every other script in this project talks to .29.

Usage:
    python watch_live_ratings.py [interval_seconds]
"""

import subprocess
import sys
import time

HOST = "sriparna@172.16.1.29"
REMOTE_CMD = (
    "source /data/sriparna/miniconda3/etc/profile.d/conda.sh && "
    "conda activate iclr && "
    "cd /data/sriparna/swagata/iclr/human && "
    "python live_status.py"
)

INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def clear_screen():
    print("\033[H\033[J", end="")


def main():
    print(f"Watching live human-eval ratings on {HOST} every {INTERVAL}s. Ctrl+C to stop.\n", flush=True)
    try:
        while True:
            result = subprocess.run(
                ["ssh", HOST, REMOTE_CMD],
                capture_output=True, text=True, timeout=30,
            )
            clear_screen()
            print(result.stdout.strip() or "(no output)", flush=True)
            if result.returncode != 0 and result.stderr.strip():
                stderr = result.stderr.strip()
                # the conda-not-found bashrc warning is harmless noise, skip it
                if "conda: command not found" not in stderr:
                    print("\n[stderr]", stderr, flush=True)
            print(f"\n(refreshing every {INTERVAL}s -- Ctrl+C to stop)", flush=True)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
