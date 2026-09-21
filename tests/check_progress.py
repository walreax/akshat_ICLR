import os
import pandas as pd

for out_dir in ["cfs_sd3_calib", "cfs_sd3_test", "cfs_pixart_calib", "cfs_pixart_test"]:
    csv_path = os.path.join(out_dir, "concept_metrics.csv")
    npz_dir = os.path.join(out_dir, "latents")
    if not os.path.exists(csv_path):
        print(f"{out_dir}: not started")
        continue
    df = pd.read_csv(csv_path)
    n_pairs = df.groupby(["concept_id", "seed"]).ngroups if len(df) else 0
    n_npz = len([f for f in os.listdir(npz_dir) if f.endswith(".npz")]) if os.path.isdir(npz_dir) else 0
    print(f"{out_dir}: {len(df)} rows, {n_pairs} pairs done, {n_npz} latent files  (target 600 pairs)")
