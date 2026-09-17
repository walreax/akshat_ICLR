# ============================================================
# Shared utilities for the imageMetric batch generation scripts
# (generate_FLUX.py, generate_pixArt.py, generate_sd3.py).
#
# IMPORTANT: each calling script must set CUDA_VISIBLE_DEVICES
# (via os.environ.setdefault, so wait_for_gpu.py can override it)
# BEFORE importing this module, since this module imports torch.
# ============================================================

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch


def print_gpu_info(header: str = "GPU INFORMATION") -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 70)
    print(header)
    print("=" * 70)

    print("Visible CUDA devices:", torch.cuda.device_count())
    print("Using GPU:", torch.cuda.get_device_name(0))

    free_memory, total_memory = torch.cuda.mem_get_info()
    print(f"Free VRAM:  {free_memory / 1024**3:.2f} GiB")
    print(f"Total VRAM: {total_memory / 1024**3:.2f} GiB")
    print()


def load_sampled_csv(
    path: Path,
    required_columns: set,
    print_summary: bool = True,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find:\n{path.resolve()}")

    df = pd.read_csv(path)

    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing columns: {sorted(missing)}")

    if print_summary:
        print(f"Loaded {len(df)} samples.")
        if "dataset_type" in df.columns:
            print(f"Poems: {(df['dataset_type'] == 'poem').sum()}")
            print(f"Stories: {(df['dataset_type'] == 'story').sum()}")
        print()

    return df


def compute_remaining(
    sampled: pd.DataFrame,
    image_dir: Path,
    metadata_path: Path,
    id_column: str = "id",
    image_ext: str = "png",
):
    already_done = set()

    def _is_valid_image(path: Path) -> bool:
        # A 0-byte file means a write got interrupted mid-save (e.g. the
        # process was killed) -- exists() alone would wrongly treat that
        # as complete and skip it forever on every future resume.
        try:
            return path.stat().st_size > 0
        except OSError:
            return False

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    image_path = metadata_path.parent / record["image"]
                    if _is_valid_image(image_path):
                        already_done.add(record[id_column])
                except Exception:
                    pass

    # Also check the image directory directly, in case an image exists
    # but isn't in the log.
    for image_path in image_dir.glob(f"*.{image_ext}"):
        if _is_valid_image(image_path):
            already_done.add(image_path.stem)

    remaining = sampled[~sampled[id_column].isin(already_done)]

    print("=" * 70)
    print("BATCH STATUS")
    print("=" * 70)
    print(f"Total samples:    {len(sampled)}")
    print(f"Already complete: {len(already_done)}")
    print(f"Remaining:        {len(remaining)}")
    print()

    return remaining, already_done


@dataclass
class GenerationStats:
    generated: int = 0
    oom_skipped: int = 0
    other_failed: int = 0
    reloads: int = 0


def run_generation_loop(
    remaining: pd.DataFrame,
    *,
    load_pipeline: Callable[[], Any],
    generate_one: Callable[[Any, str, "torch.Generator"], Any],
    build_record_extra: Callable[[pd.Series, float], dict],
    image_dir: Path,
    metadata_path: Path,
    seed: int,
    generator_device: str = "cpu",
    reload_every: int = 100,
    oom_retries: int = 1,
    oom_retry_sleep_sec: float = 5.0,
    prompt_column: str = "content",
    prompt_max_chars: int = 2000,
    id_column: str = "id",
    image_ext: str = "png",
) -> GenerationStats:
    """
    Owns the per-image generation loop: prompt/generator setup, OOM
    retry-then-reload-then-skip, image save, jsonl logging (append+flush),
    per-iteration cleanup, and a periodic full pipeline reload every
    `reload_every` images to fight the sustained VRAM growth observed
    with enable_model_cpu_offload() over long runs.
    """

    stats = GenerationStats()

    print("Loading pipeline...")
    pipe = load_pipeline()
    print("Pipeline loaded.")
    print()

    since_reload = 0
    total = len(remaining)

    with open(metadata_path, "a", encoding="utf-8") as log_f:

        for count, (_, row) in enumerate(remaining.iterrows(), start=1):

            image_id = row[id_column]

            print("=" * 70)
            print(f"[{count}/{total}] {image_id}")
            print(f"Seed: {seed}")
            print()

            prompt = str(row[prompt_column])[:prompt_max_chars]

            attempt = 0
            image = None
            elapsed = 0.0

            while True:
                generator = torch.Generator(device=generator_device).manual_seed(seed)

                try:
                    start_time = time.time()
                    image = generate_one(pipe, prompt, generator)
                    elapsed = time.time() - start_time
                    break

                except torch.cuda.OutOfMemoryError as e:
                    print("CUDA OUT OF MEMORY")
                    print(repr(e))

                    del generator
                    gc.collect()
                    torch.cuda.empty_cache()

                    if attempt < oom_retries:
                        attempt += 1
                        print(
                            f"Rebuilding pipeline and retrying "
                            f"(attempt {attempt}/{oom_retries})..."
                        )
                        time.sleep(oom_retry_sleep_sec)

                        del pipe
                        gc.collect()
                        torch.cuda.empty_cache()

                        pipe = load_pipeline()
                        stats.reloads += 1
                        since_reload = 0
                        continue

                    print("Giving up on this image after retry — skipping.")
                    stats.oom_skipped += 1
                    break

                except Exception as e:
                    print("GENERATION FAILED")
                    print(repr(e))

                    del generator
                    stats.other_failed += 1
                    break

            if image is None:
                print()
                continue

            image_filename = f"{image_id}.{image_ext}"
            image_path = image_dir / image_filename
            image.save(image_path)

            record = {
                "id": image_id,
                **build_record_extra(row, elapsed),
                "seed": seed,
                "image": f"{image_dir.name}/{image_filename}",
            }

            log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_f.flush()

            stats.generated += 1

            del image
            del generator
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Generated in {elapsed:.1f}s")
            print(f"Saved -> {image_path}")
            print()

            since_reload += 1
            if reload_every and since_reload >= reload_every:
                print(
                    f"Reached {reload_every} images since last reload — "
                    f"rebuilding pipeline to reclaim memory."
                )
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                pipe = load_pipeline()
                stats.reloads += 1
                since_reload = 0
                print("Pipeline reloaded.")
                print()

    return stats
