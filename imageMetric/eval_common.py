# ============================================================
# Shared dataset/scoring logic for the evaluation scripts
# (evaluate_models.py, evaluate_vqascore.py, build_summary.py).
#
# Split into three scripts because VQAScore (t2v_metrics'
# clip-flant5-xl) needs an exact pinned transformers==4.49.0/
# torch==2.5.1 environment ("vqa_eval") that's incompatible with the
# transformers version generate_sd3.py's env ("iclr") needs -- see
# evaluate_vqascore.py's header for the full story.
#
# evaluate_models.py and evaluate_vqascore.py each write to their OWN
# results file (MODELS_RESULTS_PATH / VQASCORE_RESULTS_PATH), never a
# shared one -- when they were both writing into one per_entry_scores.csv,
# running concurrently on separate GPUs, each one's periodic checkpoint
# write clobbered the other's already-written columns (each process
# holds its own in-memory results_df read once at startup, so neither
# sees the other's writes). build_summary.py is the only place the two
# get merged, after both have finished, so there's no concurrent-write
# race at all.

from pathlib import Path

import pandas as pd


SAMPLED_PATH = Path("./sd3_output/sd3_sampled_prompts.csv")

MODEL_DIRS = {
    "SD3": Path("./sd3_output/sd3_images"),
    "PixArt": Path("./pixart_output/pixart_images"),
    "FLUX": Path("./flux_output/flux_images"),
}

EVAL_DIR = Path("./evaluation_output")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODELS_RESULTS_PATH = EVAL_DIR / "per_entry_scores_models.csv"
VQASCORE_RESULTS_PATH = EVAL_DIR / "per_entry_scores_vqascore.csv"
PER_ENTRY_PATH = EVAL_DIR / "per_entry_scores.csv"  # merged, written by build_summary.py only
INCEPTION_SCORES_PATH = EVAL_DIR / "inception_scores.json"
SUMMARY_PATH = EVAL_DIR / "metrics_4x3.csv"
SUMMARY_JSON_PATH = EVAL_DIR / "metrics_4x3.json"


def find_image(image_dir, sample_id):
    """
    Find the image corresponding to a dataset ID.
    Supports PNG/JPG/JPEG/WEBP.
    """
    for ext in ["png", "jpg", "jpeg", "webp"]:
        path = image_dir / f"{sample_id}.{ext}"
        if path.exists():
            return path

    return None


def build_eval_df():
    """
    Loads the shared sampled-prompts dataset and keeps only the entries
    for which ALL THREE models generated an image -- the common
    evaluation set every metric script scores.
    """

    print("=" * 70)
    print("Loading sampled dataset")
    print("=" * 70)

    df = pd.read_csv(SAMPLED_PATH)

    for col in ["id", "content"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {SAMPLED_PATH}")

    print(f"Dataset entries: {len(df)}")

    print("\nChecking generated images...")

    for model_name, image_dir in MODEL_DIRS.items():
        print(f"{model_name}: {image_dir}")

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    entries = []

    for _, row in df.iterrows():

        sample_id = str(row["id"])
        text = str(row["content"])

        images = {}

        for model_name, image_dir in MODEL_DIRS.items():
            image_path = find_image(image_dir, sample_id)

            if image_path is not None:
                images[model_name] = str(image_path)

        if len(images) == len(MODEL_DIRS):

            entries.append({
                "id": sample_id,
                "text": text,
                **{
                    f"{model}_image": path
                    for model, path in images.items()
                }
            })

    eval_df = pd.DataFrame(entries)

    print()
    print(f"Common evaluation entries: {len(eval_df)}")

    missing = len(df) - len(eval_df)

    if missing:
        print(f"Entries excluded because one or more models are missing: {missing}")

    if len(eval_df) == 0:
        raise RuntimeError("No common images found across all three models.")

    return eval_df


def load_or_init_results(eval_df, path):
    """
    Loads this script's own results CSV (see module docstring for why
    each script has its own -- MODELS_RESULTS_PATH or
    VQASCORE_RESULTS_PATH, never the shared PER_ENTRY_PATH) if one
    exists from a previous run of THIS SAME script (so a resumed run
    doesn't lose already-computed columns), otherwise starts a fresh
    id/text-only table. Falls back to fresh if the existing file's ids
    don't match eval_df (stale/incompatible run).
    """

    ids = eval_df["id"].tolist()

    if path.exists():
        existing = pd.read_csv(path)

        if existing["id"].tolist() == ids:
            print(f"Reusing existing results file: {path}")
            return existing

        print(
            f"Existing {path} has different/mismatched ids "
            f"-- starting fresh instead of merging into it."
        )

    return pd.DataFrame({"id": ids, "text": eval_df["text"].tolist()})


def evaluate_t2v_metric(eval_df, metric_name, scorer, model_name, batch_size):
    """
    Evaluate one metric for one image-generation model.

    On CUDA OOM, halves the batch size and retries; if it still OOMs at
    batch_size=1, that entry is scored as NaN and skipped rather than
    aborting the whole run.

    Returns:
        numpy array containing one score per dataset entry.
    """

    import gc

    import numpy as np
    import torch
    from tqdm import tqdm

    print()
    print("-" * 70)
    print(f"{metric_name} | {model_name} (batch_size={batch_size})")
    print("-" * 70)

    images = [
        row[f"{model_name}_image"]
        for _, row in eval_df.iterrows()
    ]

    texts = [
        row["text"]
        for _, row in eval_df.iterrows()
    ]

    scores = []
    start = 0

    with tqdm(total=len(images), desc=f"{metric_name} {model_name}") as pbar:

        while start < len(images):

            end = min(start + batch_size, len(images))

            batch_images = images[start:end]
            batch_texts = texts[start:end]

            try:
                with torch.no_grad():

                    batch_scores = scorer(
                        images=batch_images,
                        texts=batch_texts
                    )

                batch_scores = batch_scores.detach().float().cpu()

                # t2v_metrics scorers (CLIPScore, VQAScore) return a full
                # [len(images), len(texts)] cross-product matrix, even
                # when called with equal-length matched lists -- see
                # t2v_metrics/score.py's own forward() docstring ("if
                # there are m images and n texts, return a m x n
                # tensor"). We want batch_images[i] paired with
                # batch_texts[i], i.e. the diagonal, not every
                # combination. Custom scorers (e.g. NativeBLIP2ITMScorer)
                # already return a plain 1D per-pair tensor, so only take
                # the diagonal when the shape is actually square B x B.
                if (
                    batch_scores.ndim >= 2
                    and batch_scores.shape[0] == len(batch_images)
                    and batch_scores.shape[1] == len(batch_texts)
                ):
                    batch_scores = torch.diagonal(batch_scores)

                batch_scores = np.asarray(batch_scores.numpy()).reshape(-1)

                if len(batch_scores) != len(batch_images):
                    raise RuntimeError(
                        f"{metric_name} returned {len(batch_scores)} scores "
                        f"for {len(batch_images)} images -- expected one "
                        f"score per image. Check the scorer's output shape."
                    )

                scores.extend(batch_scores.tolist())

                pbar.update(end - start)
                start = end

            except torch.cuda.OutOfMemoryError as e:

                gc.collect()
                torch.cuda.empty_cache()

                if batch_size > 1:
                    print(
                        f"\nCUDA OOM at batch_size={batch_size} "
                        f"[{start}:{end}] — retrying at "
                        f"batch_size={batch_size // 2}"
                    )
                    batch_size = max(1, batch_size // 2)
                    continue

                print(
                    f"\nCUDA OOM even at batch_size=1 for entry "
                    f"{start} — skipping (NaN)."
                )
                print(repr(e))

                scores.append(float("nan"))
                pbar.update(1)
                start += 1

    import numpy as np
    return np.asarray(scores)
