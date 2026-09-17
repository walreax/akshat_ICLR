#!/usr/bin/env python3
# ============================================================
# Wait for a GPU with enough free VRAM, then launch a command on it.
#
# Never falls back to CPU: if no candidate GPU clears the free-memory
# threshold, this keeps polling (default every 5 minutes) indefinitely.
# Does not import torch, so waiting itself never touches a GPU.
#
# Usage:
#   python wait_for_gpu.py --min-free-gb 18 --gpus 0,1 \
#       --check-interval-sec 300 -- python generate_FLUX.py
#
# The chosen physical GPU index is exported as CUDA_VISIBLE_DEVICES for
# the launched command, which then sees it as cuda:0.
# ============================================================

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


def query_gpu_free_mb():
    """Returns {physical_index: free_mib} for every GPU nvidia-smi reports."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    free_by_index = {}
    for line in result.stdout.strip().splitlines():
        index_str, free_str = (part.strip() for part in line.split(","))
        free_by_index[int(index_str)] = int(free_str)

    return free_by_index


def pick_gpu(candidate_indices, min_free_mb, select):
    try:
        free_by_index = query_gpu_free_mb()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[wait_for_gpu] nvidia-smi query failed ({e!r}), will retry.")
        return None

    eligible = {
        idx: free_mb
        for idx, free_mb in free_by_index.items()
        if idx in candidate_indices and free_mb >= min_free_mb
    }

    if not eligible:
        status = ", ".join(
            f"GPU{idx}={free_by_index.get(idx, '?')}MiB free"
            for idx in candidate_indices
        )
        print(f"[wait_for_gpu] no GPU with >= {min_free_mb:.0f} MiB free yet ({status})")
        return None

    if select == "most-free":
        return max(eligible, key=eligible.get)

    return next(iter(eligible))  # first-fit, in candidate order


def main():
    if "--" in sys.argv:
        split_at = sys.argv.index("--")
        own_args = sys.argv[1:split_at]
        command = sys.argv[split_at + 1:]
    else:
        own_args = sys.argv[1:]
        command = []

    parser = argparse.ArgumentParser(
        description="Wait for a free GPU, then launch a command on it."
    )
    parser.add_argument("--min-free-gb", type=float, required=True)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated physical GPU indices to consider (default: all visible)",
    )
    parser.add_argument("--check-interval-sec", type=float, default=300.0)
    parser.add_argument(
        "--select",
        choices=["most-free", "first-fit"],
        default="most-free",
    )
    args = parser.parse_args(own_args)

    if not command:
        parser.error(
            "no command given — pass it after '--', "
            "e.g. ... -- python generate_FLUX.py"
        )

    min_free_mb = args.min_free_gb * 1024

    if args.gpus:
        candidate_indices = [int(x) for x in args.gpus.split(",")]
    else:
        try:
            candidate_indices = sorted(query_gpu_free_mb().keys())
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[wait_for_gpu] could not query nvidia-smi to list GPUs: {e!r}")
            sys.exit(1)

    print(f"[wait_for_gpu] candidates: {candidate_indices}, need >= {args.min_free_gb} GiB free")
    print(f"[wait_for_gpu] checking every {args.check_interval_sec:.0f}s, GPU-only (no CPU fallback)")

    chosen = None
    while chosen is None:
        chosen = pick_gpu(candidate_indices, min_free_mb, args.select)

        if chosen is None:
            print(
                f"[wait_for_gpu] {datetime.now().isoformat(timespec='seconds')} "
                f"— sleeping {args.check_interval_sec:.0f}s"
            )
            time.sleep(args.check_interval_sec)

    free_by_index = query_gpu_free_mb()
    print(
        f"[wait_for_gpu] launching on GPU {chosen} "
        f"(free={free_by_index.get(chosen, '?')} MiB) -> {' '.join(command)}"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(chosen)

    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
