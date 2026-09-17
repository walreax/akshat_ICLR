import pandas as pd

df = pd.read_csv("cfs_sd3_calib/concept_metrics.csv")
print("total rows:", len(df))
pair_counts = df.groupby(["concept_id", "seed"]).size()
print("distinct (concept,seed) pairs:", len(pair_counts))
print("rows-per-pair distribution:\n", pair_counts.value_counts().sort_index())
dup = pair_counts[pair_counts > 15]
print("\npairs with MORE than 15 rows (should be impossible if truly unique):", len(dup))
if len(dup):
    print(dup.head(10))
    example = dup.index[0]
    rows = df[(df.concept_id == example[0]) & (df.seed == example[1])]
    print(f"\nexample duplicate pair {example}: {len(rows)} rows")
    print(rows[["block", "step_index", "timestep_fraction", "latent_idx"]].to_string())

import os
idx = pd.read_csv("cfs_sd3_calib/latents/latents_index.csv", header=None, names=["concept_id", "seed", "file"])
print("\nlatents_index entries:", len(idx))
print(idx.to_string())
