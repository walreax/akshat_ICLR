"""
top_concepts.py
================
For each model (SD3, PixArt) and each (block, timestep_fraction), ranks
every SAE latent that was ever selected as the argmax concept, by how many
times it was selected across the whole benchmark set -- the SAE's "top
concepts" at that block/timestep. Read-only analysis over existing
--load-sae (test-set) concept_metrics.csv output; no new generation or
training.

Latent indices are NOT comparable across blocks (block 4's latent #42 and
block 20's latent #42 come from different SAEs), so ranking is always
within a single (model, block, timestep_fraction) group.

Usage:
    python top_concepts.py

No CLI to memorize -- edit the constants below if the input paths change.
Handles partial data gracefully: a model whose run hasn't produced any
evaluation rows yet just reports zero groups, rather than erroring, so
this can be re-run anytime as SD3_CSV / PIXART_CSV grow.
"""

import os
import pandas as pd

SD3_CSV = "/DATA/swagata/akshat_ICLR/sd3_benchmark_sae_test/concept_metrics.csv"
PIXART_CSV = "/DATA/swagata/akshat_ICLR/pixart_benchmark_full/concept_metrics.csv"

TOP_N = 32

OUT_FULL = "top_concepts_full.csv"
OUT_TOP32 = "top_concepts_top32.csv"


def load(path: str, model: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [skip] {model}: no file at {path}")
        return pd.DataFrame(columns=["model", "block", "timestep_fraction", "latent_idx", "concentration"])
    df = pd.read_csv(path, usecols=["block", "timestep_fraction", "latent_idx", "concentration"])
    if df.empty:
        print(f"  [skip] {model}: {path} has 0 evaluation rows so far")
        return pd.DataFrame(columns=["model", "block", "timestep_fraction", "latent_idx", "concentration"])
    df["model"] = model
    print(f"  {model}: {len(df)} rows from {path}")
    return df


def main():
    print("Loading source data...")
    frames = [f for f in [load(SD3_CSV, "SD3"), load(PIXART_CSV, "PixArt")] if not f.empty]
    if not frames:
        print("\nNo evaluation rows available yet in either file. Nothing to rank.")
        return
    combined = pd.concat(frames, ignore_index=True)

    group_cols = ["model", "block", "timestep_fraction", "latent_idx"]
    agg = (
        combined.groupby(group_cols)
        .agg(frequency=("concentration", "size"), mean_concentration=("concentration", "mean"))
        .reset_index()
    )

    # Rank within each (model, block, timestep_fraction) cell -- frequency
    # descending, mean_concentration as the tiebreaker.
    rank_cols = ["model", "block", "timestep_fraction"]
    agg = agg.sort_values(rank_cols + ["frequency", "mean_concentration"],
                           ascending=[True, True, True, False, False])
    agg["rank"] = agg.groupby(rank_cols).cumcount() + 1
    agg["is_top32"] = agg["rank"] <= TOP_N

    agg.to_csv(OUT_FULL, index=False)
    agg[agg["is_top32"]].to_csv(OUT_TOP32, index=False)
    print(f"\nWrote {len(agg)} total (model, block, timestep, latent) rows -> {OUT_FULL}")
    print(f"Wrote {agg['is_top32'].sum()} top-{TOP_N} rows -> {OUT_TOP32}")

    print(f"\n=== per (model, block, timestep) summary ===")
    for key, group in agg.groupby(rank_cols):
        model, block, frac = key
        n_latents = len(group)
        top5 = group.nsmallest(5, "rank")[["latent_idx", "frequency", "mean_concentration"]]
        print(f"\n{model} / {block} / t={frac}: {n_latents} distinct latent(s) seen")
        for _, row in top5.iterrows():
            print(f"    latent {int(row['latent_idx']):5d}  freq={int(row['frequency']):4d}  "
                  f"mean_concentration={row['mean_concentration']:.4f}")


if __name__ == "__main__":
    main()
