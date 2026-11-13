import os

# MUST be set before importing torch.
# setdefault (not =) so wait_for_gpu.py can override this externally.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================================
# VQAScore (t2v_metrics, clip-flant5-xl)
# ============================================================
#
# Split out from evaluate_models.py because this specific model is a
# bespoke LLaVA-style checkpoint (zhiqiulin/clip-flant5-xl) whose
# loading code was written against transformers==4.49.0. Against the
# newer transformers used everywhere else in this project (needed for
# SD3/FLUX generation), it doesn't just fail to import -- it loads
# then crashes with a CUDA index-out-of-bounds error deep in the
# tokenizer/embedding path, which is a correctness risk, not just an
# API-shape mismatch. Run this script in a SEPARATE conda env ("vqa_eval")
# pinned to exactly torch==2.5.1 / transformers==4.49.0 (t2v_metrics'
# own declared requirements), e.g.:
#
#   conda create -n vqa_eval python=3.10
#   conda activate vqa_eval
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#   pip install transformers==4.49.0
#   pip install --no-deps t2v_metrics
#   pip install opencv-python-headless einops sentencepiece open_clip_torch \
#       omegaconf iopath fvcore fairscale torchvision==0.20.1 pandas
#
# t2v_metrics also hard-requires ffmpeg on PATH at import time (used by
# unrelated video-scoring submodules it still imports even though we
# only touch the image path) -- if `conda install ffmpeg` is too slow
# on your solver, point LD_LIBRARY_PATH at another env's already-built
# ffmpeg instead of re-solving one from scratch.
#
# Runs in the SAME node/filesystem as evaluate_models.py and writes
# into the same per_entry_scores.csv, adding only the VQAScore_*
# columns -- run it before, after, or concurrently (different GPU)
# with evaluate_models.py, order doesn't matter.

import gc

import torch

import eval_common as ec

DEVICE = "cuda"
BATCH_SIZE = 8  # clip-flant5-xl is a multi-billion-parameter VLM.
VQA_MODEL = "clip-flant5-xl"


eval_df = ec.build_eval_df()
results_df = ec.load_or_init_results(eval_df, ec.VQASCORE_RESULTS_PATH)


print("\nLoading t2v_metrics...")

import t2v_metrics


print("\n" + "=" * 70)
print("VQASCORE")
print("=" * 70)

vqa_scorer = t2v_metrics.VQAScore(
    model=VQA_MODEL,
    device=DEVICE
)

for model_name in ec.MODEL_DIRS:

    scores = ec.evaluate_t2v_metric(
        eval_df,
        "VQAScore",
        vqa_scorer,
        model_name,
        batch_size=BATCH_SIZE
    )

    results_df[f"VQAScore_{model_name}"] = scores

    results_df.to_csv(ec.VQASCORE_RESULTS_PATH, index=False)

del vqa_scorer
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("evaluate_vqascore.py COMPLETE")
print("=" * 70)
print(f"Per-entry scores : {ec.VQASCORE_RESULTS_PATH}")
