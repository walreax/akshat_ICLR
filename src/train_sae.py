"""
train_sae.py
============

FLUX SAE training using the project's existing, version-robust FLUX
residual extraction logic.

The important fix is that FLUX blocks in recent diffusers versions pass
hidden_states through kwargs. The hooks below therefore NEVER assume
args[0] exists.

Pipeline
--------
prompt
  -> FLUX.1-schnell
  -> selected block
  -> Δ = image_stream_output - image_stream_input
  -> TopK SAE

We train one SAE per:
    block × timestep bucket

Default buckets:
    t1.0 = first denoising step
    t0.5 = middle step
    t0.0 = final step

Default blocks:
    double_early / double_mid / double_late
    single_early / single_mid / single_late

This is a practical FLUX adaptation of the paper's SAE methodology.
The paper itself uses SD1.4; this script uses FLUX because that is the
generator in your project.

IMPORTANT
---------
This is an ONLINE / STREAMING SAE trainer. It does not try to keep all
5,000 VIST activations in RAM. Each FLUX generation feeds the observed
delta vectors directly into the corresponding SAE trainer.

Start with:

After verifying the run:
    TRAIN_PROMPTS = 5_000

Run:
    python train_sae.py

Background:
    nohup python -u train_sae.py > train_sae.log 2>&1 &

Check:
    tail -f train_sae.log
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import FluxPipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# Keep token in environment, not source code.
HF_TOKEN = os.environ.get("HF_TOKEN")

SEED = 42

# VIST training corpus: 5,000 story texts.
TRAIN_STORIES = 2_520
TRAIN_POEMS = 1_680
TRAIN_TOTAL = TRAIN_STORIES + TRAIN_POEMS
SEED = 42

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

# Smoke test first. This WAS silently set to TRAIN_TOTAL (4,200) --
# not actually a smoke test. At FLUX-dev's 28 steps that was
# observed to take ~5+ min/prompt with sequential_cpu_offload, i.e.
# ~2 WEEKS of continuous GPU time for a single block. Verify a real
# run end-to-end at this small size first, then raise it deliberately.
TRAIN_PROMPTS = 20

# Set to TRAIN_TOTAL (4_200) for the full-corpus run, once verified.
# Note: switching MODEL_ID to FLUX.1-schnell (4 steps vs dev's 28)
# should cut per-prompt generation time roughly ~7x versus the dev
# run this estimate above is based on -- re-time it on this hardware
# before committing to a full run, rather than assuming the ~7x
# holds exactly (offload overhead doesn't necessarily scale linearly
# with step count).
#
# The exact filtered 5,000 VIST list used by the paper is not bundled
# here, so this streams the public VIST captions.
# VIST-only training corpus: 5,000 five-sentence story texts.


# ------------------------------------------------------------
# FLUX generation
# ------------------------------------------------------------

HEIGHT = 512
WIDTH = 512

# schnell is distilled/guidance-free: 4 steps, guidance_scale=0 is
# its intended operating point (not dev's 28 / 3.5). It also caps at
# 256 text tokens (dev supports up to 512) -- a larger value raises
# a ValueError.
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
MAX_SEQUENCE_LENGTH = 256

DTYPE = torch.bfloat16

# Paper-inspired temporal buckets.
TIMESTEP_BUCKETS = {
    "t1.0": 0,
    "t0.5": NUM_INFERENCE_STEPS // 2,
    "t0.0": NUM_INFERENCE_STEPS - 1,
}

TARGET_STEPS = set(
    TIMESTEP_BUCKETS.values()
)

STEP_TO_BUCKET = {
    step: bucket
    for bucket, step in TIMESTEP_BUCKETS.items()
}

# ------------------------------------------------------------
# FLUX blocks
# ------------------------------------------------------------

# Structural analog of the six blocks used in your current FLUX SAE
# prototype. Indices are resolved dynamically.
# Select a specific integer FLUX block directly.
# This is the safe default for the 24 GB A5000 smoke/full test.
TRAIN_BLOCK_CONFIG = {
    "double_mid": ("double", 10),
}

# Set True to derive early/mid/late integer indices automatically.
USE_ALL_BLOCKS = False

ALL_BLOCK_FRACTIONS = {
    "double_early": ("double", 0.00),
    "double_mid": ("double", 0.50),
    "double_late": ("double", 1.00),
    "single_early": ("single", 0.00),
    "single_mid": ("single", 0.50),
    "single_late": ("single", 1.00),
}


# ------------------------------------------------------------
# SAE
# ------------------------------------------------------------

SAE_EXPANSION_FACTOR = 4
SAE_TOPK = 32

# One optimizer update per chunk.
SAE_BATCH_SIZE = 512

SAE_LR = 1e-3

# We use online training, so each vector is seen once.
# You can increase this for repeated training over the stream.
SAE_PASSES = 1

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

ROOT = Path("flux_sae_training")
WEIGHTS_DIR = ROOT / "weights"
ROOT.mkdir(
    parents=True,
    exist_ok=True,
)
WEIGHTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROMPTS_FILE = ROOT / "prompts.jsonl"
DIAGNOSTICS_FILE = ROOT / "diagnostics.json"
PROGRESS_FILE = ROOT / "progress.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

VIST_URL = (
    "https://visionandlanguage.net/"
    "VIST/json_files/story-in-sequence/"
    "SIS-with-labels.tar.gz"
)

VIST_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36"
)

POEMSUM_BASE = (
    "https://raw.githubusercontent.com/"
    "Ridwan230/PoemSum/main/Dataset/"
)
POEMSUM_FILES = [
    "poemsum_train.csv",
    "poemsum_valid.csv",
    "poemsum_test.csv",
]


def _clean(x):
    return "" if x is None else str(x).strip()


def _norm(x):
    return "".join(
        c.lower() if c.isalnum() else "_"
        for c in str(x)
    ).strip("_")


def _find_value(row, candidates):
    normalized = {
        _norm(k): k
        for k in row
    }

    for candidate in candidates:
        key = _norm(candidate)
        if key in normalized:
            return row[normalized[key]]

    for nk, original in normalized.items():
        for candidate in candidates:
            key = _norm(candidate)
            if key in nk or nk in key:
                return row[original]

    return None


def _download_vist(path):
    print("Downloading VIST SIS...")

    import requests

    headers = {
        "User-Agent": VIST_USER_AGENT,
        "Referer": "https://visionandlanguage.net/VIST/dataset.html",
        "Accept": "*/*",
    }

    try:
        with requests.get(
            VIST_URL,
            headers=headers,
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()

            with path.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
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
    if curl is None:
        raise RuntimeError(
            "curl is required for VIST download."
        )

    subprocess.run(
        [
            curl,
            "-L",
            "--fail",
            "--retry", "4",
            "-A", VIST_USER_AGENT,
            "-e", "https://visionandlanguage.net/VIST/dataset.html",
            "-o", str(path),
            VIST_URL,
        ],
        check=True,
    )


def _extract_records(obj):
    if isinstance(obj, dict):
        annotations = obj.get("annotations")

        if isinstance(annotations, list):
            out = []

            for group in annotations:
                if isinstance(group, list):
                    items = [
                        x for x in group
                        if isinstance(x, dict)
                    ]
                    if items:
                        out.append(items)
                elif isinstance(group, dict):
                    out.append(group)

            return out

        for key in (
            "data",
            "stories",
            "story",
            "records",
        ):
            value = obj.get(key)
            if isinstance(value, list):
                return [
                    x for x in value
                    if isinstance(x, dict)
                ]

    if isinstance(obj, list):
        out = []

        for group in obj:
            if isinstance(group, list):
                items = [
                    x for x in group
                    if isinstance(x, dict)
                ]
                if items:
                    out.append(items)
            elif isinstance(group, dict):
                out.append(group)

        return out

    return []


def load_vist_texts(n_stories):
    with tempfile.TemporaryDirectory(
        prefix="vist_sae_train_"
    ) as temp_name:
        temp = Path(temp_name)
        archive = temp / "SIS-with-labels.tar.gz"

        _download_vist(archive)

        extracted = temp / "vist"
        extracted.mkdir()

        print("Extracting VIST SIS...")

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

        for path in extracted.rglob("*.json"):
            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    obj = json.load(f)
            except Exception:
                continue

            for record in _extract_records(obj):
                if not isinstance(record, list):
                    continue

                items = list(record)

                def order_key(item):
                    value = _find_value(
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

                items.sort(key=order_key)

                sentences = []

                for item in items:
                    sentence = _clean(
                        _find_value(
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
                    if sentence:
                        sentences.append(sentence)

                if not sentences:
                    continue

                content = " ".join(
                    sentences[:5]
                ).strip()

                if not content or content in seen:
                    continue

                seen.add(content)

                story_id = _clean(
                    _find_value(
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
                        "id": story_id or f"vist_{len(stories):06d}",
                        "prompt": content,
                        "source": "vist",
                    }
                )

                if len(stories) >= n_stories:
                    return stories

        return stories


def _find_poem_col(df, names):
    exact = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in exact:
            return exact[name.lower()]

    for col in df.columns:
        value = str(col).strip().lower()
        for name in names:
            if name.lower() in value:
                return col

    return None


def load_poemsum_texts(n_poems):
    import requests
    import pandas as pd
    from io import StringIO

    poems = []
    seen = set()

    for filename in POEMSUM_FILES:
        url = POEMSUM_BASE + filename
        print(
            f"Downloading PoemSum: {filename}"
        )

        response = requests.get(
            url,
            timeout=120,
        )
        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text)
        )

        text_col = _find_poem_col(
            df,
            [
                "poem",
                "poem_text",
                "text",
                "content",
                "poetry",
                "document",
            ],
        )

        if text_col is None:
            raise RuntimeError(
                f"Could not find poem text column in {filename}: "
                f"{list(df.columns)}"
            )

        title_col = _find_poem_col(
            df,
            [
                "title",
                "poem_title",
                "name_of_work",
                "name",
            ],
        )

        for row_index, row in df.iterrows():
            if pd.isna(row[text_col]):
                continue

            content = str(
                row[text_col]
            ).strip()

            if not content or content in seen:
                continue

            seen.add(content)

            name = (
                str(row[title_col]).strip()
                if title_col is not None
                and pd.notna(row[title_col])
                else f"Poem {len(poems) + 1}"
            )

            poems.append(
                {
                    "id": (
                        f"poem_sum:{filename}:"
                        f"{row_index}"
                    ),
                    "prompt": content,
                    "source": "poem_sum",
                    "name_of_work": name,
                }
            )

            if len(poems) >= n_poems:
                return poems

    return poems


def load_mixed_texts(n_stories, n_poems):
    stories = load_vist_texts(n_stories)
    poems = load_poemsum_texts(n_poems)

    if len(stories) < n_stories:
        raise RuntimeError(
            f"Only {len(stories)} VIST stories loaded; "
            f"need {n_stories}."
        )

    if len(poems) < n_poems:
        raise RuntimeError(
            f"Only {len(poems)} PoemSum poems loaded; "
            f"need {n_poems}."
        )

    mixed = stories + poems
    random.Random(SEED).shuffle(mixed)
    return mixed

def save_prompts(prompts):
    with PROMPTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in prompts:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                ) + "\n"
            )


# ============================================================
# SAE
# ============================================================

class TopKSAE(nn.Module):
    """
    z = TopK(ReLU(W_enc (x-b)))
    x_hat = W_dec z + b
    """

    def __init__(
        self,
        d_in: int,
        n_features: int,
        k: int,
    ):
        super().__init__()

        self.d_in = d_in
        self.n_features = n_features
        self.k = k

        self.b = nn.Parameter(
            torch.zeros(
                d_in
            )
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

    def encode(
        self,
        x: torch.Tensor,
    ):
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

    def decode(
        self,
        z: torch.Tensor,
    ):
        return (
            z @ self.W_dec.t()
            + self.b
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class SAETrainer:
    def __init__(
        self,
        d_in: int,
        device: torch.device,
        label: str,
    ):
        self.label = label
        self.device = device

        n_features = (
            SAE_EXPANSION_FACTOR
            * d_in
        )

        self.sae = TopKSAE(
            d_in,
            n_features,
            SAE_TOPK,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=SAE_LR,
        )

        self.num_vectors = 0
        self.num_updates = 0
        self.loss_sum = 0.0

        self.mean = torch.zeros(
            d_in,
            device=device,
        )
        self.mean_count = 0

        self._bias_initialized = False

    def _initialize_bias(
        self,
        x: torch.Tensor,
    ):
        if self._bias_initialized:
            return

        with torch.no_grad():
            self.mean.copy_(
                x.mean(
                    dim=0
                )
            )
            self.sae.b.copy_(
                self.mean
            )

        self._bias_initialized = True

    def train_chunk(
        self,
        x: torch.Tensor,
    ):
        if x.numel() == 0:
            return

        x = x.to(
            self.device,
            dtype=torch.float32,
        )

        self._initialize_bias(
            x
        )

        self.sae.train()

        # The FLUX generation is run under torch.no_grad(), but SAE
        # optimization must use autograd.
        with torch.enable_grad():
            x_hat, z = self.sae(
                x
            )

            loss = F.mse_loss(
                x_hat,
                x,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            self.optimizer.step()

        # Normalize decoder columns after every update.
        with torch.no_grad():
            self.sae.W_dec.div_(
                self.sae.W_dec.norm(
                    dim=0,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

        self.num_vectors += (
            x.shape[0]
        )

        self.num_updates += 1

        self.loss_sum += float(
            loss.detach().cpu()
        )

    @property
    def mean_loss(self):
        if self.num_updates == 0:
            return None

        return (
            self.loss_sum
            / self.num_updates
        )

    def save(
        self,
        path: Path,
        metadata: dict,
    ):
        torch.save(
            {
                "state_dict":
                    self.sae.state_dict(),
                "d_in":
                    self.sae.d_in,
                "n_features":
                    self.sae.n_features,
                "k":
                    self.sae.k,
                "block":
                    metadata["block"],
                "bucket":
                    metadata["bucket"],
                "num_vectors":
                    self.num_vectors,
                "num_updates":
                    self.num_updates,
                "mean_loss":
                    self.mean_loss,
                "model":
                    MODEL_ID,
                "seed":
                    SEED,
            },
            path,
        )


# ============================================================
# FLUX BLOCK CONFIGURATION
# ============================================================

def build_block_config(transformer):
    """
    Return:
        name -> (kind, integer_index)

    The old version accidentally returned a float for the smoke-test
    block index, which caused:
        TypeError: 'float' object cannot be interpreted as an integer
    when indexing ModuleList.
    """
    if not USE_ALL_BLOCKS:
        return dict(TRAIN_BLOCK_CONFIG)

    n_double = len(
        transformer.transformer_blocks
    )
    n_single = len(
        transformer.single_transformer_blocks
    )

    result = {}

    for name, (kind, fraction) in ALL_BLOCK_FRACTIONS.items():
        n = (
            n_double
            if kind == "double"
            else n_single
        )

        index = int(
            round(
                fraction * (n - 1)
            )
        )

        index = max(
            0,
            min(
                index,
                n - 1,
            ),
        )

        result[name] = (
            kind,
            index,
        )

    return result


# ============================================================
# STEP TRACKING
# ============================================================

class StepTracker:
    def __init__(self):
        self.step = -1


class TransformerStepWrapper:
    """
    Counts one transformer.forward call per FLUX denoising step.
    """

    def __init__(
        self,
        transformer,
        tracker: StepTracker,
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

        self.transformer.forward = wrapped

    def remove(self):
        self.transformer.forward = (
            self.original
        )


# ============================================================
# ROBUST FLUX DELTA RECORDER
# ============================================================

class BlockDeltaRecorder:
    """
    Version-robust FLUX block hook.

    IMPORTANT:
    Current diffusers may pass:
        hidden_states
        encoder_hidden_states
    as keyword arguments.

    Therefore:
        kwargs.get("hidden_states", args[0] if args else None)

    is used instead of blindly accessing args[0].
    """

    def __init__(
        self,
        block_kind: str,
        block_name: str,
        target_steps,
        tracker: StepTracker,
        text_length: int,
        callback,
    ):
        self.block_kind = block_kind
        self.block_name = block_name
        self.target_steps = target_steps
        self.tracker = tracker
        self.text_length = text_length
        self.callback = callback

        self.cached_input = None
        self.cached_mode = None

    def pre_hook(
        self,
        module,
        args,
        kwargs,
    ):
        step = self.tracker.step

        if step not in self.target_steps:
            self.cached_input = None
            self.cached_mode = None
            return

        hidden_states = kwargs.get(
            "hidden_states",
            args[0] if args else None,
        )

        if hidden_states is None:
            self.cached_input = None
            self.cached_mode = None

            raise RuntimeError(
                f"[{self.block_name}] "
                "Could not find hidden_states. "
                f"args={len(args)}, "
                f"kwargs={list(kwargs.keys())}"
            )

        encoder_hidden_states = (
            kwargs.get(
                "encoder_hidden_states",
                None,
            )
        )

        # Current diffusers:
        # hidden_states is image-only.
        if (
            self.block_kind == "double"
            or encoder_hidden_states is not None
        ):
            if hidden_states.ndim != 3:
                raise RuntimeError(
                    f"[{self.block_name}] "
                    f"Unexpected hidden_states shape: "
                    f"{tuple(hidden_states.shape)}"
                )

            self.cached_input = (
                hidden_states
                .detach()[0]
                .float()
                .clone()
            )

            self.cached_mode = (
                "image_only"
            )

        else:
            # Older single-stream implementation:
            # [text ; image] concatenated.
            if (
                hidden_states.ndim != 3
            ):
                raise RuntimeError(
                    f"[{self.block_name}] "
                    f"Unexpected single-stream shape: "
                    f"{tuple(hidden_states.shape)}"
                )

            self.cached_input = (
                hidden_states
                .detach()[0, self.text_length:, :]
                .float()
                .clone()
            )

            self.cached_mode = (
                "text_image_concat"
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
            step not in self.target_steps
            or self.cached_input is None
        ):
            return output

        # Current FLUX blocks return:
        # (encoder_hidden_states, hidden_states)
        if isinstance(
            output,
            tuple,
        ):
            if len(output) < 2:
                raise RuntimeError(
                    f"[{self.block_name}] "
                    "Tuple output has fewer than 2 elements."
                )

            image_out = (
                output[1]
                .detach()[0]
                .float()
                .clone()
            )

        else:
            # Older single-stream version returns a concatenated tensor.
            if self.cached_mode == (
                "text_image_concat"
            ):
                image_out = (
                    output
                    .detach()[
                        0,
                        self.text_length:,
                        :,
                    ]
                    .float()
                    .clone()
                )
            else:
                image_out = (
                    output
                    .detach()[0]
                    .float()
                    .clone()
                )

        if (
            image_out.shape
            != self.cached_input.shape
        ):
            raise RuntimeError(
                f"[{self.block_name}] "
                "FLUX block delta shape mismatch:\n"
                f"input={tuple(self.cached_input.shape)}\n"
                f"output={tuple(image_out.shape)}\n"
                f"step={step}\n"
                f"mode={self.cached_mode}"
            )

        delta = (
            image_out
            - self.cached_input
        )

        bucket = STEP_TO_BUCKET[step]

        self.callback(
            self.block_name,
            bucket,
            delta,
        )

        self.cached_input = None
        self.cached_mode = None

        return output


# ============================================================
# TRAINING STATE
# ============================================================

class OnlineSAEPool:
    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device

        self.trainers: Dict[
            Tuple[str, str],
            SAETrainer,
        ] = {}

    def consume(
        self,
        block_name: str,
        bucket: str,
        delta: torch.Tensor,
    ):
        """
        delta:
            [num_image_tokens, d]
        """

        if delta.ndim != 2:
            raise RuntimeError(
                f"Expected [tokens, d], got "
                f"{tuple(delta.shape)}"
            )

        key = (
            block_name,
            bucket,
        )

        if key not in self.trainers:
            trainer = SAETrainer(
                d_in=delta.shape[-1],
                device=self.device,
                label=(
                    f"{block_name}/{bucket}"
                ),
            )

            self.trainers[key] = (
                trainer
            )

            print(
                f"Created SAE "
                f"{block_name}/{bucket}: "
                f"d={delta.shape[-1]}, "
                f"features="
                f"{SAE_EXPANSION_FACTOR * delta.shape[-1]}"
            )

        trainer = self.trainers[
            key
        ]

        # Avoid keeping graph/reference memory.
        x = delta.detach().cpu()

        if (
            x.shape[0]
            > SAE_BATCH_SIZE
        ):
            for chunk in x.split(
                SAE_BATCH_SIZE
            ):
                trainer.train_chunk(
                    chunk
                )
        else:
            trainer.train_chunk(
                x
            )

    def save_all(
        self,
        completed_prompts: int,
    ):
        diagnostics = {}

        for (
            block,
            bucket,
        ), trainer in self.trainers.items():

            path = (
                WEIGHTS_DIR
                / (
                    f"sae_{block}_"
                    f"{bucket}.pt"
                )
            )

            trainer.save(
                path,
                {
                    "block":
                        block,
                    "bucket":
                        bucket,
                },
            )

            diagnostics[
                f"{block}/{bucket}"
            ] = {
                "d_in":
                    trainer.sae.d_in,
                "n_features":
                    trainer.sae.n_features,
                "k":
                    trainer.sae.k,
                "num_vectors":
                    trainer.num_vectors,
                "num_updates":
                    trainer.num_updates,
                "mean_loss":
                    trainer.mean_loss,
                "completed_prompts":
                    completed_prompts,
            }

        with DIAGNOSTICS_FILE.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                diagnostics,
                f,
                indent=2,
            )

        print(
            f"Saved {len(self.trainers)} SAE weights."
        )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    seed_everything(
        SEED
    )

    device = torch.device(
        "cuda"
    )

    print()
    print("=" * 72)
    print("FLUX SAE TRAINING")
    print("=" * 72)
    print(
        f"Model: {MODEL_ID}"
    )
    print(
        f"Mixed prompts: {TRAIN_TOTAL:,} (60% VIST / 40% PoemSum)"
    )
    print(
        f"Steps: {NUM_INFERENCE_STEPS}"
    )
    print(
        f"Target steps: {sorted(TARGET_STEPS)}"
    )
    print(
        f"SAE TopK: {SAE_TOPK}"
    )
    print(
        f"SAE expansion: {SAE_EXPANSION_FACTOR}x"
    )
    print(
        f"Resolution: {HEIGHT}x{WIDTH}"
    )

    if HF_TOKEN:
        print(
            "HF_TOKEN detected."
        )
    else:
        print(
            "No HF_TOKEN found; using local Hugging Face access."
        )

    # --------------------------------------------------------
    # Mixed VIST + PoemSum prompts
    # --------------------------------------------------------

    print("\nLoading mixed training corpus...")
    print(f"VIST: {TRAIN_STORIES:,}")
    print(f"PoemSum: {TRAIN_POEMS:,}")
    print(f"Total: {TRAIN_TOTAL:,}")

    corpus = load_mixed_texts(
        TRAIN_STORIES,
        TRAIN_POEMS,
    )

    save_prompts(corpus)

    prompts = [
        item["prompt"]
        for item in corpus
    ]

    print(f"Loaded {len(prompts):,} mixed text prompts.")
    print(
        "Source mix: "
        f"VIST={sum(x.get('source') == 'vist' for x in corpus):,}, "
        f"PoemSum={sum(x.get('source') == 'poem_sum' for x in corpus):,}"
    )

    # --------------------------------------------------------
    # FLUX
    # --------------------------------------------------------

    print(
        "\nLoading FLUX..."
    )

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        token=HF_TOKEN,
    )

    # Safer for shared 24 GB GPUs.
    pipe.enable_sequential_cpu_offload(
        gpu_id=0
    )

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    transformer = pipe.transformer

    print(
        "FLUX loaded."
    )

    # Freeze FLUX. We only optimize SAE weights.
    transformer.requires_grad_(
        False
    )

    if hasattr(
        pipe,
        "text_encoder",
    ):
        pipe.text_encoder.requires_grad_(
            False
        )

    if hasattr(
        pipe,
        "text_encoder_2",
    ):
        pipe.text_encoder_2.requires_grad_(
            False
        )

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    print(
        f"FLUX double blocks: {len(transformer.transformer_blocks)}"
    )
    print(
        f"FLUX single blocks: {len(transformer.single_transformer_blocks)}"
    )

    block_config = build_block_config(
        transformer
    )

    print(
        "\nBlock configuration:"
    )

    for name, (
        kind,
        index,
    ) in block_config.items():
        print(
            f"  {name:16s} "
            f"-> {kind}[{index}]"
        )

    # --------------------------------------------------------
    # SAE pool
    # --------------------------------------------------------

    pool = OnlineSAEPool(
        device
    )

    def receive_delta(
        block_name,
        bucket,
        delta,
    ):
        pool.consume(
            block_name,
            bucket,
            delta,
        )

    tracker = StepTracker()

    # --------------------------------------------------------
    # Install robust hooks
    # --------------------------------------------------------

    recorder_handles = []

    for block_name, (
        kind,
        index,
    ) in block_config.items():

        if kind == "double":
            block = (
                transformer
                .transformer_blocks[
                    index
                ]
            )
        else:
            block = (
                transformer
                .single_transformer_blocks[
                    index
                ]
            )

        recorder = BlockDeltaRecorder(
            block_kind=kind,
            block_name=block_name,
            target_steps=TARGET_STEPS,
            tracker=tracker,
            text_length=MAX_SEQUENCE_LENGTH,
            callback=receive_delta,
        )

        recorder_handles.extend(
            [
                block.register_forward_pre_hook(
                    recorder.pre_hook,
                    with_kwargs=True,
                ),
                block.register_forward_hook(
                    recorder.post_hook,
                    with_kwargs=True,
                ),
            ]
        )

    step_wrapper = (
        TransformerStepWrapper(
            transformer,
            tracker,
        )
    )

    step_wrapper.install()

    # --------------------------------------------------------
    # Generate every prompt
    # --------------------------------------------------------

    try:
        for idx, prompt in enumerate(
            prompts
        ):
            tracker.step = -1

            generator = (
                torch.Generator(
                    device="cpu"
                ).manual_seed(
                    SEED + idx
                )
            )

            started = time.time()

            with torch.no_grad():
                pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    generator=generator,
                )

            elapsed = (
                time.time()
                - started
            )

            if (
                idx == 0
                or (idx + 1) % 10 == 0
            ):
                print(
                    f"[{idx + 1}/{len(prompts)}] "
                    f"generation={elapsed:.1f}s"
                )

            # Save checkpoint every 10 prompts.
            if (
                (idx + 1) % 10 == 0
                or idx == len(prompts) - 1
            ):
                pool.save_all(
                    idx + 1
                )

                with PROGRESS_FILE.open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "completed_prompts":
                                idx + 1,
                            "total_prompts":
                                len(prompts),
                        },
                        f,
                        indent=2,
                    )

            gc.collect()
            torch.cuda.empty_cache()

    finally:
        for handle in recorder_handles:
            handle.remove()

        step_wrapper.remove()

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    pool.save_all(
        len(prompts)
    )

    del pipe

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(
        "Weights:",
        WEIGHTS_DIR,
    )
    print(
        "Diagnostics:",
        DIAGNOSTICS_FILE,
    )


if __name__ == "__main__":
    main()