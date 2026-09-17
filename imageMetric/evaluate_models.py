import os

# MUST be set before importing torch.
# setdefault (not =) so wait_for_gpu.py can override this externally.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json

import torch
from PIL import Image

import eval_common as ec

# ============================================================
# CONFIG
# ============================================================
#
# Computes CLIP Score, BLIP-2 ITM Score, and Inception Score. VQAScore
# lives in evaluate_vqascore.py instead -- it needs an exact pinned
# transformers==4.49.0/torch==2.5.1 environment that's incompatible
# with what this script needs, see that script's header for why.

DEVICE = "cuda"

# CLIP ViT-L/14 is tiny; larger batches are fine. Inception needs
# uint8 image tensors decoded per-batch, kept modest.
BATCH_SIZES = {
    "CLIP": 32,
    "BLIP2": 8,
    "Inception": 16,
}

CLIP_MODEL = "openai:ViT-L-14"
# Native transformers checkpoint, not t2v_metrics' "blip2-itm" -- see
# NativeBLIP2ITMScorer below for why.
BLIP_MODEL = "Salesforce/blip2-itm-vit-g"


eval_df = ec.build_eval_df()
results_df = ec.load_or_init_results(eval_df, ec.MODELS_RESULTS_PATH)


print("\nLoading t2v_metrics...")

import t2v_metrics


# ============================================================
# 1. CLIP SCORE
# ============================================================

print("\n" + "=" * 70)
print("METRIC 1/3: CLIP SCORE")
print("=" * 70)

clip_scorer = t2v_metrics.CLIPScore(
    model=CLIP_MODEL,
    device=DEVICE
)

for model_name in ec.MODEL_DIRS:

    scores = ec.evaluate_t2v_metric(
        eval_df,
        "CLIP",
        clip_scorer,
        model_name,
        batch_size=BATCH_SIZES["CLIP"]
    )

    results_df[f"CLIP_{model_name}"] = scores

    # Checkpoint after every metric/model pair so a late OOM or
    # preemption doesn't lose already-computed columns.
    results_df.to_csv(ec.MODELS_RESULTS_PATH, index=False)

del clip_scorer
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 2. BLIP-2 ITM SCORE
# ============================================================
#
# Uses transformers' own natively-maintained Blip2ForImageTextRetrieval
# instead of t2v_metrics.ITMScore. t2v_metrics vendors a 3+ year old
# copy of LAVIS's BERT/Q-Former implementation (written against
# transformers ~4.15-4.49) that has accumulated multiple internal-API
# incompatibilities with the transformers version installed here --
# this sidesteps that entirely by using the checkpoint through
# transformers' current, maintained BLIP-2 support.

class NativeBLIP2ITMScorer:
    """
    Drop-in replacement for t2v_metrics.ITMScore(model="blip2-itm"):
    same (images, texts) -> per-pair match-probability calling
    convention, so evaluate_t2v_metric() needs no changes.
    """

    def __init__(self, model, device="cuda"):
        from transformers import Blip2Processor, Blip2ForImageTextRetrieval

        self.device = device
        self.processor = Blip2Processor.from_pretrained(model)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            model, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

    def __call__(self, images, texts):
        pil_images = [Image.open(p).convert("RGB") for p in images]

        inputs = self.processor(
            images=pil_images,
            text=texts,
            return_tensors="pt",
            padding=True,
            # Some prompts (long poems/stories) run past BLIP-2's
            # Q-Former text encoder's 512-token position-embedding
            # limit -- without this it crashes instead of truncating.
            truncation=True,
            max_length=512,
        ).to(self.device, torch.float16)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_image_text_matching_head=True
            )

        # logits_per_image: [B, 2] (no-match, match), paired per index --
        # not a cross-product matrix like CLIP's.
        probs = torch.nn.functional.softmax(
            outputs.logits_per_image,
            dim=-1
        )

        return probs[:, 1]


print("\n" + "=" * 70)
print("METRIC 2/3: BLIP-2 ITM SCORE")
print("=" * 70)

blip_scorer = NativeBLIP2ITMScorer(
    model=BLIP_MODEL,
    device=DEVICE
)

for model_name in ec.MODEL_DIRS:

    scores = ec.evaluate_t2v_metric(
        eval_df,
        "BLIP-2",
        blip_scorer,
        model_name,
        batch_size=BATCH_SIZES["BLIP2"]
    )

    results_df[f"BLIP2_{model_name}"] = scores

    results_df.to_csv(ec.MODELS_RESULTS_PATH, index=False)

del blip_scorer
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 3. INCEPTION SCORE
# ============================================================
#
# Not a per-entry metric (no text pairing), so it lives in its own
# small JSON file rather than per_entry_scores.csv -- build_summary.py
# reads both files back together.

print("\n" + "=" * 70)
print("METRIC 3/3: INCEPTION SCORE")
print("=" * 70)

from torchmetrics.image.inception import InceptionScore
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def calculate_inception_score(model_name, batch_size):

    print(f"\nCalculating Inception Score: {model_name} (batch_size={batch_size})")

    metric = InceptionScore(
        feature="logits_unbiased",
        splits=10,
        normalize=False
    ).to(DEVICE)

    image_paths = [
        row[f"{model_name}_image"]
        for _, row in eval_df.iterrows()
    ]

    skipped = 0

    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc=f"Inception {model_name}"
    ):

        batch_paths = image_paths[start:start + batch_size]

        tensors = []

        for path in batch_paths:

            image = Image.open(path).convert("RGB")

            # InceptionScore expects uint8 [0,255]
            tensor = pil_to_tensor(image)

            tensors.append(tensor)

        batch = torch.stack(tensors).to(DEVICE)

        try:
            with torch.no_grad():
                metric.update(batch)

        except torch.cuda.OutOfMemoryError as e:

            print(
                f"\nCUDA OOM updating Inception Score for "
                f"[{start}:{start + len(batch_paths)}] — skipping this batch."
            )
            print(repr(e))

            skipped += len(batch_paths)

            del batch
            gc.collect()
            torch.cuda.empty_cache()

    if skipped:
        print(
            f"Inception Score for {model_name} computed from "
            f"{len(image_paths) - skipped}/{len(image_paths)} images "
            f"({skipped} skipped due to OOM)."
        )

    mean, std = metric.compute()

    mean = float(mean.detach().cpu())
    std = float(std.detach().cpu())

    del metric

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"{model_name}: "
        f"IS = {mean:.6f} ± {std:.6f}"
    )

    return mean, std


inception_scores = {}

for model_name in ec.MODEL_DIRS:

    mean, std = calculate_inception_score(
        model_name,
        batch_size=BATCH_SIZES["Inception"]
    )

    inception_scores[model_name] = {
        "mean": mean,
        "std": std
    }

with ec.INCEPTION_SCORES_PATH.open("w", encoding="utf-8") as f:
    json.dump(inception_scores, f, indent=2)

print(f"\nSaved: {ec.INCEPTION_SCORES_PATH}")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("evaluate_models.py COMPLETE")
print("=" * 70)
print(f"Per-entry scores  : {ec.MODELS_RESULTS_PATH}")
print(f"Inception scores  : {ec.INCEPTION_SCORES_PATH}")
print()
print(
    "Run evaluate_vqascore.py (in the vqa_eval env) for VQAScore, then "
    "build_summary.py to assemble the final 4x3 table."
)
