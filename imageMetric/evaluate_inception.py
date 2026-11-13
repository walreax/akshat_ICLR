import os

# MUST be set before importing torch.
# setdefault (not =) so wait_for_gpu.py can override this externally.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================================
# Inception Score
# ============================================================
#
# Split out from evaluate_models.py so it can run concurrently with
# evaluate_blip2.py on a separate GPU -- writes its own JSON file, no
# shared state with anything else.

import gc
import json

import torch
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

import eval_common as ec

DEVICE = "cuda"
BATCH_SIZE = 16


eval_df = ec.build_eval_df()


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


print("\n" + "=" * 70)
print("INCEPTION SCORE")
print("=" * 70)

inception_scores = {}

for model_name in ec.MODEL_DIRS:

    mean, std = calculate_inception_score(
        model_name,
        batch_size=BATCH_SIZE
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
print("evaluate_inception.py COMPLETE")
print("=" * 70)
