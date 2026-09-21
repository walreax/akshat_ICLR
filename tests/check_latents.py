"""Inspect the first latent .npz files written by --save-latents (sanity check for the new code path)."""
import glob
import os
import sys

import numpy as np

for out_dir in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(out_dir, "latents", "*.npz")))
    csv_path = os.path.join(out_dir, "concept_metrics.csv")
    n_rows = sum(1 for _ in open(csv_path)) - 1 if os.path.exists(csv_path) else 0
    print(f"== {out_dir}: {len(files)} latent file(s), {n_rows} metric rows")
    if not files:
        continue
    z = np.load(files[0])
    keys = sorted(z.files)
    cells = sorted({k.rsplit("|", 1)[0] for k in keys})
    print(f"   first file: {os.path.basename(files[0])}  ({os.path.getsize(files[0]) / 1024:.0f} KB)")
    print(f"   cells: {len(cells)} -> {cells[:3]} ...")
    c = cells[0]
    mean, top, maps = z[c + "|mean"], z[c + "|top_idx"], z.get(c + "|maps")
    print(f"   mean: shape={mean.shape} dtype={mean.dtype} nonzero={int((mean > 0).sum())} max={float(mean.max()):.3f}")
    print(f"   top_idx: {top[:8].tolist()}  l0={float(z[c + '|l0']):.1f} active latents/token")
    if maps is not None:
        print(f"   maps: shape={maps.shape} dtype={maps.dtype} max={float(maps.max()):.3f}")
    else:
        print("   maps: MISSING (non-square token grid?)")
