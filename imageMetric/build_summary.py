# ============================================================
# Assembles the final 4x3 metrics table from evaluate_models.py's and
# evaluate_vqascore.py's outputs. Run this last, after both have
# completed -- it needs no GPU and works in any env with pandas.
# ============================================================

import json

import pandas as pd

import eval_common as ec


if not ec.MODELS_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"{ec.MODELS_RESULTS_PATH} not found -- run evaluate_models.py first."
    )

if not ec.VQASCORE_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"{ec.VQASCORE_RESULTS_PATH} not found -- run evaluate_vqascore.py first."
    )

if not ec.INCEPTION_SCORES_PATH.exists():
    raise FileNotFoundError(
        f"{ec.INCEPTION_SCORES_PATH} not found -- run evaluate_models.py first."
    )

models_df = pd.read_csv(ec.MODELS_RESULTS_PATH)
vqascore_df = pd.read_csv(ec.VQASCORE_RESULTS_PATH)

# Merge on id -- each script's own file has its own "text" column too
# (both built from the same eval_df), so drop the duplicate before merging.
results_df = models_df.merge(
    vqascore_df.drop(columns=["text"]),
    on="id",
    how="inner",
)

if len(results_df) != len(models_df) or len(results_df) != len(vqascore_df):
    raise RuntimeError(
        f"evaluate_models.py produced {len(models_df)} rows and "
        f"evaluate_vqascore.py produced {len(vqascore_df)} rows, but "
        f"merging on id gave {len(results_df)} -- their ids don't fully "
        f"match. Make sure both were run against the same generated images."
    )

with ec.INCEPTION_SCORES_PATH.open("r", encoding="utf-8") as f:
    inception_scores = json.load(f)

required_columns = [
    f"{metric}_{model}"
    for metric in ("CLIP", "BLIP2", "VQAScore")
    for model in ec.MODEL_DIRS
]

missing_columns = [c for c in required_columns if c not in results_df.columns]

if missing_columns:
    raise RuntimeError(
        f"Merged results are missing columns: {missing_columns}. "
        f"Make sure both evaluate_models.py AND evaluate_vqascore.py "
        f"have completed all three models."
    )

# Write the merged table out once, here, where there's no concurrent
# writer to race with.
results_df.to_csv(ec.PER_ENTRY_PATH, index=False)


summary = pd.DataFrame(
    index=["Inception Score", "VQAScore", "CLIP Score", "BLIP-2 Score"],
    columns=list(ec.MODEL_DIRS.keys()),
    dtype=float,
)

for model_name in ec.MODEL_DIRS:
    summary.loc["Inception Score", model_name] = inception_scores[model_name]["mean"]
    summary.loc["VQAScore", model_name] = results_df[f"VQAScore_{model_name}"].mean()
    summary.loc["CLIP Score", model_name] = results_df[f"CLIP_{model_name}"].mean()
    summary.loc["BLIP-2 Score", model_name] = results_df[f"BLIP2_{model_name}"].mean()


summary.to_csv(ec.SUMMARY_PATH)
summary.to_json(ec.SUMMARY_JSON_PATH, orient="index", indent=2)


print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

print("\n" + "=" * 70)
print("FILES")
print("=" * 70)
print(f"Per-entry scores : {ec.PER_ENTRY_PATH}")
print(f"Summary CSV      : {ec.SUMMARY_PATH}")
print(f"Summary JSON     : {ec.SUMMARY_JSON_PATH}")
print("\nDone.")
