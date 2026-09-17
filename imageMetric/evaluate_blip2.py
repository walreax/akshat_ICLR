import os

# MUST be set before importing torch.
# setdefault (not =) so wait_for_gpu.py can override this externally.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================================
# BLIP-2 ITM Score
# ============================================================
#
# Runs in the "vqa_eval" env (transformers==4.49.0/torch==2.5.1), NOT
# "iclr" -- an earlier version of this script used transformers' own
# natively-maintained Blip2ForImageTextRetrieval to avoid t2v_metrics'
# vendored LAVIS/Q-Former code, which worked for short prompts, but
# real (long) poem/story prompts crashed transformers 5.16.1's newest
# attention-masking internals (create_bidirectional_mask /
# _ignore_bidirectional_mask_sdpa) with a CUDA index-out-of-bounds
# error, independent of the truncation fix that was also needed.
# t2v_metrics.ITMScore(model="blip2-itm") -- the same LAVIS path
# VQAScore already uses successfully -- handles the same long-prompt
# case cleanly in this env, since transformers==4.49.0 is what that
# vendored code was actually written against. See evaluate_vqascore.py's
# header for the vqa_eval env setup (same env, same t2v_metrics install).
#
# Resumes/extends MODELS_RESULTS_PATH (which already has CLIP's columns
# from evaluate_models.py's run in the iclr env) -- do not run this
# concurrently with anything else that also writes to that file.

import gc

import torch

import eval_common as ec

DEVICE = "cuda"
BATCH_SIZE = 8
BLIP_MODEL = "blip2-itm"


eval_df = ec.build_eval_df()
results_df = ec.load_or_init_results(eval_df, ec.MODELS_RESULTS_PATH)


print("\nLoading t2v_metrics...")

import t2v_metrics


print("\n" + "=" * 70)
print("BLIP-2 ITM SCORE")
print("=" * 70)

blip_scorer = t2v_metrics.ITMScore(
    model=BLIP_MODEL,
    device=DEVICE
)

for model_name in ec.MODEL_DIRS:

    scores = ec.evaluate_t2v_metric(
        eval_df,
        "BLIP-2",
        blip_scorer,
        model_name,
        batch_size=BATCH_SIZE
    )

    results_df[f"BLIP2_{model_name}"] = scores

    results_df.to_csv(ec.MODELS_RESULTS_PATH, index=False)

del blip_scorer
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("evaluate_blip2.py COMPLETE")
print("=" * 70)
print(f"Per-entry scores : {ec.MODELS_RESULTS_PATH}")
