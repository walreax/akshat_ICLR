"""
smoke_test_ablation.py
=======================
Minimal end-to-end check that SAEFeaturePatcher actually does something
causal: pick one real row out of pixart_benchmark_full/concept_metrics.csv
(a concept, seed, block, step, latent_idx the metric already scored),
regenerate it four ways with the exact same seed --

    baseline        -- no patch, sanity reference
    ablate          -- zero the real latent's contribution
    amplify         -- 3x the real latent's contribution
    control-ablate  -- zero a DIFFERENT, unrelated latent by the same rule

-- and report how much each changed vs. baseline, overall and inside a
circular mask centered on the row's own (centroid_x, centroid_y). This is
a smoke test, not the real study: one row, no statistics, just "does the
mechanism work and is it roughly localized" before building
run_ablation.py on top of it.

Usage:
    python smoke_test_ablation.py
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

from sd_activation_extractor import SparseAutoencoder, SAEFeaturePatcher
from pixart_activation_extractor import PixArtInternalsExtractor

CSV_PATH = "/DATA/swagata/akshat_ICLR/pixart_benchmark_full/concept_metrics.csv"
SAE_DIR = "/DATA/swagata/akshat_ICLR/sae_checkpoints_provided"
OUT_DIR = "/DATA/swagata/akshat_ICLR/ablation_smoke_test"
DEVICE = "cuda"
AMPLIFY_FACTOR = 3.0


def pick_row() -> pd.Series:
    df = pd.read_csv(CSV_PATH)
    df = df[df["concentration"] > 0.6]  # a reasonably well-localized concept, not noise
    row = df.sort_values("concentration", ascending=False).iloc[0]
    print(f"[pick_row] concept_id={row['concept_id']!r}  block={row['block']}  "
          f"seed={row['seed']}  step_index={row['step_index']}  "
          f"latent_idx={row['latent_idx']}  concentration={row['concentration']:.3f}")
    return row


def mean_abs_diff(a: Image.Image, b: Image.Image, mask=None) -> float:
    ta = torch.from_numpy(np.array(a.convert("RGB"))).float()
    tb = torch.from_numpy(np.array(b.convert("RGB"))).float()
    diff = (ta - tb).abs().mean(dim=-1)  # (H, W)
    if mask is not None:
        diff = diff * mask
        return (diff.sum() / mask.sum().clamp(min=1)).item()
    return diff.mean().item()


def circular_mask(size, cx_frac: float, cy_frac: float, radius_frac: float = 0.18) -> torch.Tensor:
    w, h = size
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    cx, cy = cx_frac * w, cy_frac * h
    r = radius_frac * min(w, h)
    return ((xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2).float()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    row = pick_row()
    block = row["block"]
    seed = int(row["seed"])
    step_index = int(row["step_index"])
    latent_idx = int(row["latent_idx"])
    prompt = row["prompt"]

    ckpt_path = os.path.join(SAE_DIR, f"sae_{block.replace('.', '_')}.pt")
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    d_in = state_dict["encoder.weight"].shape[1]
    d_hidden = state_dict["encoder.weight"].shape[0]
    sae = SparseAutoencoder(d_in, d_hidden).to(DEVICE)
    sae.load_state_dict(state_dict)
    sae.eval()
    print(f"[main] loaded SAE for {block}: d_in={d_in} d_hidden={d_hidden}")

    control_idx = random.Random(0).randrange(d_hidden)
    while control_idx == latent_idx:
        control_idx = random.Random(control_idx + 1).randrange(d_hidden)
    print(f"[main] control latent_idx={control_idx}")

    extractor = PixArtInternalsExtractor()

    def gen(mode: str, patch_latent=None):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        patcher = None
        if patch_latent is not None:
            patcher = SAEFeaturePatcher(
                sae, patch_latent, block, target_step=step_index,
                mode="ablate" if mode != "amplify" else "amplify",
                factor=AMPLIFY_FACTOR, do_cfg=True,
            )
        result = extractor.extract(prompt, capture_blocks=False, generator=generator, patcher=patcher)
        if patcher is not None:
            assert patcher.applied, f"patcher for mode={mode!r} never fired -- target_step/block mismatch"
        return result["image"]

    print("[main] generating baseline ...")
    baseline = gen("baseline")
    print("[main] generating ablated (real latent) ...")
    ablated = gen("ablate", patch_latent=latent_idx)
    print("[main] generating amplified (real latent) ...")
    amplified = gen("amplify", patch_latent=latent_idx)
    print("[main] generating control-ablated (unrelated latent) ...")
    control = gen("control", patch_latent=control_idx)

    baseline.save(os.path.join(OUT_DIR, "baseline.png"))
    ablated.save(os.path.join(OUT_DIR, "ablated.png"))
    amplified.save(os.path.join(OUT_DIR, "amplified.png"))
    control.save(os.path.join(OUT_DIR, "control.png"))

    w, h = baseline.size
    cx_frac = row["centroid_x"] / 64.0  # heatmap_out_size default is 64
    cy_frac = row["centroid_y"] / 64.0
    mask = circular_mask((w, h), cx_frac, cy_frac)

    print("\n=== mean abs pixel diff vs. baseline (0-255 scale) ===")
    for name, img in [("ablated", ablated), ("amplified", amplified), ("control", control)]:
        local = mean_abs_diff(baseline, img, mask=mask)
        glob = mean_abs_diff(baseline, img)
        print(f"  {name:10s}  local={local:6.2f}  global={glob:6.2f}  specificity={local / (glob + 1e-3):5.2f}")

    print(f"\nSaved 4 images -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
