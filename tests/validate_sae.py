from __future__ import annotations

import gc
import json
import os
import random
import tarfile
import tempfile
import hashlib
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-dev"
HF_TOKEN = os.environ.get("HF_TOKEN")

SEED = 123
NUM_STORIES = 10

HEIGHT = 512
WIDTH = 512

NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
MAX_SEQUENCE_LENGTH = 512
DTYPE = torch.bfloat16

TARGET_STEPS = {
    "t1.0": 0,
    "t0.5": NUM_INFERENCE_STEPS // 2,
    "t0.0": NUM_INFERENCE_STEPS - 1,
}

STEP_TO_BUCKET = {
    step: bucket
    for bucket, step in TARGET_STEPS.items()
}

BLOCK_NAME = "double_mid"
BLOCK_KIND = "double"
BLOCK_INDEX = 10

WEIGHTS_DIR = Path(
    "flux_sae_training/weights"
)

OUTPUT_DIR = Path(
    "flux_sae_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# SAE
# ============================================================

class TopKSAE(nn.Module):
    def __init__(
        self,
        d_in,
        n_features,
        k,
    ):
        super().__init__()

        self.d_in = d_in
        self.n_features = n_features
        self.k = k

        self.b = nn.Parameter(
            torch.zeros(d_in)
        )

        w_dec = torch.randn(
            d_in,
            n_features,
        )

        w_dec = w_dec / (
            w_dec.norm(
                dim=0,
                keepdim=True,
            )
            + 1e-8
        )

        self.W_dec = nn.Parameter(
            w_dec
        )

        self.W_enc = nn.Parameter(
            w_dec.t().clone()
        )

    def encode(self, x):
        pre = F.relu(
            (x - self.b)
            @ self.W_enc.t()
        )

        values, indices = torch.topk(
            pre,
            self.k,
            dim=-1,
        )

        z = torch.zeros_like(
            pre
        )

        z.scatter_(
            -1,
            indices,
            values,
        )

        return z

    def decode(self, z):
        return (
            z
            @ self.W_dec.t()
            + self.b
        )


def load_sae(
    bucket,
    device,
):
    path = (
        WEIGHTS_DIR
        / (
            f"sae_{BLOCK_NAME}_"
            f"{bucket}.pt"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing SAE weight: {path}"
        )

    payload = torch.load(
        path,
        map_location=device,
    )

    sae = TopKSAE(
        payload["d_in"],
        payload["n_features"],
        payload["k"],
    ).to(device)

    sae.load_state_dict(
        payload["state_dict"]
    )

    sae.eval()

    return sae, payload


# ============================================================
# VIST
# ============================================================

VIST_URL = (
    "https://visionandlanguage.net/"
    "VIST/json_files/story-in-sequence/"
    "SIS-with-labels.tar.gz"
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36"
)


def clean(x):
    return "" if x is None else str(x).strip()


def norm(x):
    return "".join(
        c.lower() if c.isalnum() else "_"
        for c in str(x)
    ).strip("_")


def find_value(
    row,
    candidates,
):
    normalized = {
        norm(k): k
        for k in row
    }

    for candidate in candidates:
        key = norm(candidate)

        if key in normalized:
            return row[
                normalized[key]
            ]

    for nk, original in normalized.items():
        for candidate in candidates:
            key = norm(candidate)

            if (
                key in nk
                or nk in key
            ):
                return row[original]

    return None


def download_vist(path):
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": (
            "https://visionandlanguage.net/"
            "VIST/dataset.html"
        ),
        "Accept": "*/*",
    }

    try:
        with requests.get(
            VIST_URL,
            headers=headers,
            stream=True,
            timeout=120,
        ) as response:

            response.raise_for_status()

            with path.open("wb") as f:
                for chunk in response.iter_content(
                    1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

            return

    except Exception as exc:
        print(
            "requests failed:",
            exc,
            "- trying curl",
        )

    import shutil
    import subprocess

    curl = shutil.which("curl")

    if not curl:
        raise RuntimeError(
            "curl is required."
        )

    subprocess.run(
        [
            curl,
            "-L",
            "--fail",
            "--retry",
            "4",
            "-A",
            USER_AGENT,
            "-e",
            (
                "https://visionandlanguage.net/"
                "VIST/dataset.html"
            ),
            "-o",
            str(path),
            VIST_URL,
        ],
        check=True,
    )


def extract_records(obj):
    if isinstance(obj, dict):
        annotations = obj.get(
            "annotations"
        )

        if isinstance(
            annotations,
            list,
        ):
            records = []

            for group in annotations:
                if isinstance(
                    group,
                    list,
                ):
                    items = [
                        x
                        for x in group
                        if isinstance(
                            x,
                            dict,
                        )
                    ]

                    if items:
                        records.append(
                            items
                        )

                elif isinstance(
                    group,
                    dict,
                ):
                    records.append(group)

            return records

    if isinstance(obj, list):
        records = []

        for group in obj:
            if isinstance(group, list):
                items = [
                    x
                    for x in group
                    if isinstance(x, dict)
                ]

                if items:
                    records.append(items)

            elif isinstance(group, dict):
                records.append(group)

        return records

    return []


def load_vist_stories(
    n,
):
    with tempfile.TemporaryDirectory(
        prefix="vist_validate_"
    ) as temp_name:

        temp = Path(temp_name)
        archive = (
            temp
            / "SIS-with-labels.tar.gz"
        )

        print("Downloading VIST...")
        download_vist(archive)

        extracted = temp / "vist"
        extracted.mkdir()

        print("Extracting VIST...")

        with tarfile.open(
            archive,
            "r:gz",
        ) as tar:
            tar.extractall(
                extracted,
                filter="data",
            )

        stories = []
        seen = set()

        for path in extracted.rglob(
            "*.json"
        ):
            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    obj = json.load(f)
            except Exception:
                continue

            for record in extract_records(
                obj
            ):
                if not isinstance(
                    record,
                    list,
                ):
                    continue

                items = list(record)

                def order_key(item):
                    value = find_value(
                        item,
                        [
                            "image_order",
                            "sentence_order",
                            "order",
                        ],
                    )

                    try:
                        return int(value)
                    except Exception:
                        return 999

                items.sort(
                    key=order_key
                )

                sentences = []

                for item in items:
                    text = clean(
                        find_value(
                            item,
                            [
                                "storytext",
                                "story_text",
                                "sentence",
                                "caption",
                                "text",
                            ],
                        )
                    )

                    if text:
                        sentences.append(text)

                if not sentences:
                    continue

                content = " ".join(
                    sentences[:5]
                ).strip()

                if not content:
                    continue

                if content in seen:
                    continue

                seen.add(content)

                story_id = clean(
                    find_value(
                        items[0],
                        [
                            "story_id",
                            "sequence_id",
                            "album_id",
                            "storylet_id",
                        ],
                    )
                )

                stories.append(
                    {
                        "id":
                            story_id
                            or f"vist_{len(stories)}",
                        "content":
                            content,
                    }
                )

                if len(stories) >= n:
                    return stories

        return stories


# ============================================================
# STEP TRACKER
# ============================================================

class StepTracker:
    def __init__(self):
        self.step = -1


class TransformerStepWrapper:
    def __init__(
        self,
        transformer,
        tracker,
    ):
        self.transformer = transformer
        self.tracker = tracker
        self.original = transformer.forward

    def install(self):
        original = self.original
        tracker = self.tracker

        def wrapped(
            *args,
            **kwargs,
        ):
            tracker.step += 1
            return original(
                *args,
                **kwargs,
            )

        self.transformer.forward = (
            wrapped
        )

    def remove(self):
        self.transformer.forward = (
            self.original
        )


# ============================================================
# DELTA CAPTURE
# ============================================================

class DeltaCapture:
    def __init__(
        self,
        tracker,
    ):
        self.tracker = tracker
        self.input = None
        self.data = {
            0: None,
            NUM_INFERENCE_STEPS // 2: None,
            NUM_INFERENCE_STEPS - 1: None,
        }

    def pre_hook(
        self,
        module,
        args,
        kwargs,
    ):
        step = self.tracker.step

        if step not in TARGET_STEPS.values():
            self.input = None
            return

        hidden_states = kwargs.get(
            "hidden_states",
            args[0] if args else None,
        )

        if hidden_states is None:
            raise RuntimeError(
                "hidden_states not found."
            )

        # Double-stream FLUX block:
        # hidden_states is the image stream.
        self.input = (
            hidden_states
            .detach()[0]
            .float()
            .cpu()
            .clone()
        )

    def post_hook(
        self,
        module,
        args,
        kwargs,
        output,
    ):
        step = self.tracker.step

        if (
            step not in TARGET_STEPS.values()
            or self.input is None
        ):
            return output

        if isinstance(
            output,
            tuple,
        ):
            output_image = (
                output[1]
                .detach()[0]
                .float()
                .cpu()
                .clone()
            )
        else:
            output_image = (
                output
                .detach()[0]
                .float()
                .cpu()
                .clone()
            )

        if (
            output_image.shape
            != self.input.shape
        ):
            raise RuntimeError(
                "Shape mismatch: "
                f"input={tuple(self.input.shape)} "
                f"output={tuple(output_image.shape)}"
            )

        delta = (
            output_image
            - self.input
        )

        self.data[
            step
        ] = delta

        self.input = None

        return output


# ============================================================
# METRICS
# ============================================================

def reconstruction_stats(
    x,
    x_hat,
    z,
):
    x = x.float()
    x_hat = x_hat.float()

    mse = (
        (x - x_hat) ** 2
    ).mean().item()

    variance = (
        (x - x.mean()) ** 2
    ).mean().item()

    if variance > 1e-12:
        r2 = (
            1.0
            - mse / variance
        )
    else:
        r2 = 0.0

    # Fraction of latent entries that are non-zero.
    active_fraction = (
        z.ne(0)
        .float()
        .mean()
        .item()
    )

    # Average number of active features per token.
    active_per_token = (
        z.ne(0)
        .sum(dim=-1)
        .float()
        .mean()
        .item()
    )

    mean_abs = (
        x.abs()
        .mean()
        .item()
    )

    recon_mean_abs = (
        x_hat.abs()
        .mean()
        .item()
    )

    return {
        "mse":
            mse,
        "r2":
            r2,
        "explained_variance":
            r2,
        "input_variance":
            variance,
        "input_mean_abs":
            mean_abs,
        "reconstruction_mean_abs":
            recon_mean_abs,
        "active_fraction":
            active_fraction,
        "active_features_per_token":
            active_per_token,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    seed_everything(
        SEED
    )

    device = torch.device(
        "cuda"
    )

    print("=" * 72)
    print("SAE VALIDATION")
    print("=" * 72)
    print(
        f"Stories: {NUM_STORIES}"
    )
    print(
        f"Block: {BLOCK_NAME}"
    )
    print(
        f"Block index: {BLOCK_INDEX}"
    )
    print(
        f"Steps: {sorted(TARGET_STEPS.values())}"
    )

    # --------------------------------------------------------
    # Load SAE weights
    # --------------------------------------------------------

    saes = {}

    for bucket in TARGET_STEPS:
        sae, payload = load_sae(
            bucket,
            device,
        )

        saes[
            bucket
        ] = sae

        print(
            f"{bucket}: "
            f"d={payload['d_in']} "
            f"features={payload['n_features']} "
            f"k={payload['k']}"
        )

    # --------------------------------------------------------
    # VIST
    # --------------------------------------------------------

    stories = load_vist_stories(
        NUM_STORIES
    )

    print(
        f"Loaded {len(stories)} stories."
    )

    # --------------------------------------------------------
    # FLUX
    # --------------------------------------------------------

    print(
        "Loading FLUX..."
    )

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        token=HF_TOKEN,
    )

    pipe.enable_sequential_cpu_offload(
        gpu_id=0
    )

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    transformer = pipe.transformer

    if (
        BLOCK_INDEX
        >= len(
            transformer.transformer_blocks
        )
    ):
        raise RuntimeError(
            f"double_mid index {BLOCK_INDEX} "
            f"is invalid. "
            f"Double blocks: "
            f"{len(transformer.transformer_blocks)}"
        )

    block = (
        transformer
        .transformer_blocks[
            BLOCK_INDEX
        ]
    )

    tracker = StepTracker()
    wrapper = TransformerStepWrapper(
        transformer,
        tracker,
    )

    wrapper.install()

    all_results = []

    try:
        for i, story in enumerate(
            stories
        ):
            print()
            print(
                f"[{i + 1}/{len(stories)}] "
                f"{story['id']}"
            )

            capture = DeltaCapture(
                tracker
            )

            handles = [
                block.register_forward_pre_hook(
                    capture.pre_hook,
                    with_kwargs=True,
                ),
                block.register_forward_hook(
                    capture.post_hook,
                    with_kwargs=True,
                ),
            ]

            tracker.step = -1

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(
                SEED + i
            )

            with torch.no_grad():
                result = pipe(
                    prompt=story["content"],
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    generator=generator,
                )

            for handle in handles:
                handle.remove()

            story_result = {
                "id":
                    story["id"],
                "content":
                    story["content"],
                "metrics": {},
            }

            # Save the normal FLUX image.
            image_path = (
                OUTPUT_DIR
                / f"{i:03d}.png"
            )

            result.images[0].save(
                image_path
            )

            story_result[
                "image"
            ] = str(
                image_path
            )

            for step, bucket in (
                STEP_TO_BUCKET.items()
            ):
                delta = capture.data[
                    step
                ]

                if delta is None:
                    print(
                        f"  WARNING: no delta "
                        f"captured for {bucket}"
                    )
                    continue

                sae = saes[
                    bucket
                ]

                x = delta.reshape(
                    -1,
                    delta.shape[-1],
                )

                with torch.no_grad():
                    z = sae.encode(
                        x.to(device)
                    )

                    x_hat = sae.decode(
                        z
                    )

                stats = reconstruction_stats(
                    x,
                    x_hat.cpu(),
                    z.cpu(),
                )

                story_result[
                    "metrics"
                ][
                    bucket
                ] = stats

                print(
                    f"  {bucket}: "
                    f"MSE={stats['mse']:.6f} "
                    f"R2={stats['r2']:.4f} "
                    f"active="
                    f"{stats['active_features_per_token']:.1f}"
                )

            all_results.append(
                story_result
            )

            gc.collect()
            torch.cuda.empty_cache()

    finally:
        wrapper.remove()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_json = (
        OUTPUT_DIR
        / "validation.json"
    )

    output_json.write_text(
        json.dumps(
            all_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Aggregate statistics.
    aggregate = {}

    for bucket in TARGET_STEPS:
        vals = [
            item["metrics"][bucket]
            for item in all_results
            if bucket in item["metrics"]
        ]

        if not vals:
            continue

        aggregate[bucket] = {
            "mean_mse":
                float(
                    np.mean(
                        [
                            x["mse"]
                            for x in vals
                        ]
                    )
                ),
            "mean_r2":
                float(
                    np.mean(
                        [
                            x["r2"]
                            for x in vals
                        ]
                    )
                ),
            "mean_explained_variance":
                float(
                    np.mean(
                        [
                            x["explained_variance"]
                            for x in vals
                        ]
                    )
                ),
            "mean_active_features_per_token":
                float(
                    np.mean(
                        [
                            x[
                                "active_features_per_token"
                            ]
                            for x in vals
                        ]
                    )
                ),
        }

    aggregate_path = (
        OUTPUT_DIR
        / "aggregate.json"
    )

    aggregate_path.write_text(
        json.dumps(
            aggregate,
            indent=2,
        ),
        encoding="utf-8",
    )

    del pipe

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)
    print(
        f"Per-story results: "
        f"{output_json}"
    )
    print(
        f"Aggregate results: "
        f"{aggregate_path}"
    )
    print()
    print(
        json.dumps(
            aggregate,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()