"""
PixArt-Alpha SAE training on one NVIDIA RTX A5000 / RTX 5000-class GPU.

Model:
    PixArt-alpha/PixArt-XL-2-512x512
    ~0.6B parameters

Dataset:
    train_set_4200.csv
    2,800 TinyStory + 1,400 PoemSum

SAEs:
    4 PixArt transformer blocks
    x 3 diffusion stages
    = 12 SAEs

Selected blocks:
    0, 9, 18, 27

Selected diffusion steps:
    0, 14, 27

PixArt is frozen.
Only SAE parameters are optimized.

The SAE observes:
    delta = block_output - block_input

The original PixArt output is NEVER modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import PixArtAlphaPipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"

DATASET_FILE = "train_set_4200.csv"

ROOT = Path("pixart_sae_training")
WEIGHTS_DIR = ROOT / "weights"
PROMPTS_FILE = ROOT / "prompts.jsonl"
DIAGNOSTICS_FILE = ROOT / "diagnostics.json"
PROGRESS_FILE = ROOT / "progress.json"

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

WEIGHTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

HEIGHT = 512
WIDTH = 512

NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 4.5

# PixArt inference on the RTX.
PIXART_DTYPE = torch.float16

# Exactly 4 blocks.
BLOCK_INDICES = [
    0,
    9,
    18,
    27,
]

# Exactly 3 diffusion stages.
TARGET_STEPS = [
    0,
    14,
    27,
]

SAE_EXPANSION_FACTOR = 4
SAE_TOPK = 32

# Smaller is safer on 24 GB cards.
SAE_BATCH_SIZE = 256

SAE_LR = 1e-3

CHECKPOINT_EVERY = 10

# In classifier-free guidance the positive prompt is normally
# the second item in the [negative, positive] batch.
CONDITIONAL_INDEX = 1


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


# ============================================================
# DATASET
# ============================================================

def load_dataset(
    path: Path,
):
    df = pd.read_csv(
        path
    )

    if "content" not in df.columns:
        raise RuntimeError(
            "Dataset must contain 'content'. "
            f"Columns: {list(df.columns)}"
        )

    df["content"] = (
        df["content"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["content"] != ""
    ].reset_index(
        drop=True
    )

    if len(df) != 4200:
        raise RuntimeError(
            f"Expected 4200 training samples, "
            f"found {len(df)}."
        )

    return df


def save_prompts(
    df,
):
    with PROMPTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for _, row in df.iterrows():

            item = {
                "id":
                    row.get(
                        "id",
                        "",
                    ),
                "name_of_work":
                    row.get(
                        "name_of_work",
                        "",
                    ),
                "source":
                    row.get(
                        "source",
                        "",
                    ),
                "prompt":
                    row["content"],
            }

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# SAE
# ============================================================

class TopKSAE(
    nn.Module,
):
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
                d_in,
                dtype=torch.float32,
            )
        )

        w_dec = torch.randn(
            d_in,
            n_features,
            dtype=torch.float32,
        )

        w_dec = (
            w_dec
            / (
                w_dec.norm(
                    dim=0,
                    keepdim=True,
                )
                + 1e-8
            )
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


class SAEState:
    def __init__(
        self,
        d_in: int,
        label: str,
        device: torch.device,
    ):

        self.label = label

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

    def update(
        self,
        delta: torch.Tensor,
    ):

        if delta.numel() == 0:
            return

        self.sae.train()

        for chunk in delta.split(
            SAE_BATCH_SIZE
        ):

            x = chunk.to(
                self.sae.b.device,
                dtype=torch.float32,
                non_blocking=True,
            )

            with torch.enable_grad():

                z = self.sae.encode(
                    x
                )

                x_hat = self.sae.decode(
                    z
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
                loss.detach().item()
            )

            del (
                x,
                z,
                x_hat,
                loss,
            )

        torch.cuda.empty_cache()

    @property
    def mean_loss(
        self,
    ):

        if self.num_updates == 0:
            return None

        return (
            self.loss_sum
            / self.num_updates
        )

    def save(
        self,
        path: Path,
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
                "label":
                    self.label,
                "num_vectors":
                    self.num_vectors,
                "num_updates":
                    self.num_updates,
                "mean_loss":
                    self.mean_loss,
                "model":
                    MODEL_ID,
            },
            path,
        )


# ============================================================
# STEP TRACKING
# ============================================================

class StepTracker:
    def __init__(
        self,
    ):
        self.step = -1


class TransformerStepWrapper:
    def __init__(
        self,
        transformer,
        tracker: StepTracker,
    ):

        self.transformer = transformer
        self.tracker = tracker
        self.original = transformer.forward

    def install(
        self,
    ):

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

    def remove(
        self,
    ):

        self.transformer.forward = (
            self.original
        )


# ============================================================
# BLOCK CAPTURE
# ============================================================

class BlockCapture:
    def __init__(
        self,
        tracker: StepTracker,
        target_steps,
        conditional_index: int,
    ):

        self.tracker = tracker

        self.target_steps = set(
            target_steps
        )

        self.conditional_index = (
            conditional_index
        )

        self.inputs: Dict[
            Tuple[int, int],
            torch.Tensor,
        ] = {}

        self.deltas: Dict[
            Tuple[int, int],
            torch.Tensor,
        ] = {}

    def make_pre_hook(
        self,
        block_index: int,
    ):

        def pre_hook(
            module,
            args,
            kwargs,
        ):

            step = self.tracker.step

            if step not in self.target_steps:
                return

            hidden = kwargs.get(
                "hidden_states",
                args[0] if args else None,
            )

            if hidden is None:
                raise RuntimeError(
                    f"Block {block_index}: "
                    "hidden_states not found."
                )

            if hidden.ndim != 3:
                raise RuntimeError(
                    f"Block {block_index}: "
                    f"unexpected hidden shape "
                    f"{tuple(hidden.shape)}"
                )

            batch_index = min(
                self.conditional_index,
                hidden.shape[0] - 1,
            )

            self.inputs[
                (
                    block_index,
                    step,
                )
            ] = (
                hidden[
                    batch_index
                ]
                .detach()
                .float()
                .cpu()
                .contiguous()
            )

        return pre_hook

    def make_post_hook(
        self,
        block_index: int,
    ):

        def post_hook(
            module,
            args,
            kwargs,
            output,
        ):

            step = self.tracker.step

            if step not in self.target_steps:
                return output

            key = (
                block_index,
                step,
            )

            inp = self.inputs.pop(
                key,
                None,
            )

            if inp is None:
                return output

            if isinstance(
                output,
                tuple,
            ):

                out_tensor = output[0]

            else:

                out_tensor = output

            if not isinstance(
                out_tensor,
                torch.Tensor,
            ):

                raise RuntimeError(
                    f"Block {block_index}: "
                    "unsupported output type."
                )

            batch_index = min(
                self.conditional_index,
                out_tensor.shape[0] - 1,
            )

            out = (
                out_tensor[
                    batch_index
                ]
                .detach()
                .float()
                .cpu()
                .contiguous()
            )

            if out.shape != inp.shape:

                raise RuntimeError(
                    f"Block {block_index}: "
                    f"shape mismatch "
                    f"input={tuple(inp.shape)} "
                    f"output={tuple(out.shape)}"
                )

            self.deltas[
                key
            ] = (
                out - inp
            )

            return output

        return post_hook

    def clear(
        self,
    ):

        self.inputs.clear()
        self.deltas.clear()


# ============================================================
# MAIN
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found."
        )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=DATASET_FILE,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    seed_everything(
        SEED
    )

    device = torch.device(
        "cuda"
    )

    print("=" * 72)
    print(
        "PIXART SAE TRAINING - RTX"
    )
    print("=" * 72)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3),
            2,
        ),
        "GB",
    )

    print(
        "Model:",
        MODEL_ID,
    )

    print(
        "Dataset:",
        args.dataset,
    )

    print(
        "Blocks:",
        BLOCK_INDICES,
    )

    print(
        "Steps:",
        TARGET_STEPS,
    )

    print(
        "Total SAEs:",
        len(BLOCK_INDICES)
        * len(TARGET_STEPS),
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset(
        Path(args.dataset)
    )

    start = args.start

    if args.limit is None:
        end = len(df)
    else:
        end = min(
            start + args.limit,
            len(df),
        )

    df = df.iloc[
        start:end
    ].reset_index(
        drop=True
    )

    print(
        f"Using samples "
        f"{start}:{end} "
        f"({len(df)})"
    )

    if "source" in df.columns:

        print(
            "Source mix:",
            df["source"]
            .value_counts()
            .to_dict(),
        )

    save_prompts(
        df
    )

    # --------------------------------------------------------
    # Load PixArt
    # --------------------------------------------------------

    print()
    print(
        "Loading PixArt..."
    )

    pipe = (
        PixArtAlphaPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=PIXART_DTYPE,
        )
    )

    transformer = pipe.transformer

    transformer = transformer.to(
        device
    )

    pipe.transformer = transformer

    transformer.requires_grad_(
        False
    )

    # Text encoder on CPU to save VRAM.
    if hasattr(
        pipe,
        "text_encoder",
    ):
        pipe.text_encoder.to(
            "cpu"
        )
        pipe.text_encoder.requires_grad_(
            False
        )

    # VAE is only needed after generation for image decoding.
    # CPU placement keeps GPU memory available for PixArt + SAE.
    if hasattr(
        pipe,
        "vae",
    ):
        pipe.vae.to(
            "cpu"
        )
        pipe.vae.requires_grad_(
            False
        )

    num_blocks = len(
        transformer.transformer_blocks
    )

    print(
        "PixArt transformer blocks:",
        num_blocks,
    )

    for block_index in BLOCK_INDICES:

        if (
            block_index < 0
            or block_index >= num_blocks
        ):

            raise RuntimeError(
                f"Invalid PixArt block index: "
                f"{block_index}. "
                f"Valid range is 0-{num_blocks-1}."
            )

    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    tracker = StepTracker()

    capture = BlockCapture(
        tracker=tracker,
        target_steps=TARGET_STEPS,
        conditional_index=CONDITIONAL_INDEX,
    )

    handles = []

    for block_index in BLOCK_INDICES:

        block = (
            transformer
            .transformer_blocks[
                block_index
            ]
        )

        handles.append(
            block.register_forward_pre_hook(
                capture.make_pre_hook(
                    block_index
                ),
                with_kwargs=True,
            )
        )

        handles.append(
            block.register_forward_hook(
                capture.make_post_hook(
                    block_index
                ),
                with_kwargs=True,
            )
        )

    step_wrapper = (
        TransformerStepWrapper(
            transformer,
            tracker,
        )
    )

    step_wrapper.install()

    # --------------------------------------------------------
    # SAE state
    # --------------------------------------------------------

    sae_states: Dict[
        Tuple[int, int],
        SAEState,
    ] = {}

    diagnostics = {}

    # --------------------------------------------------------
    # Main training loop
    # --------------------------------------------------------

    try:

        for local_idx, row in df.iterrows():

            global_idx = (
                start + local_idx
            )

            prompt = row[
                "content"
            ]

            tracker.step = -1

            capture.clear()

            generator = (
                torch.Generator(
                    device="cpu"
                ).manual_seed(
                    SEED + global_idx
                )
            )

            started = time.time()

            # PixArt is frozen. No graph is needed.
            with torch.inference_mode():

                pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=(
                        NUM_INFERENCE_STEPS
                    ),
                    guidance_scale=(
                        GUIDANCE_SCALE
                    ),
                    generator=generator,
                )

            elapsed = (
                time.time()
                - started
            )

            # ------------------------------------------------
            # Train all 12 SAE streams from this generation.
            # ------------------------------------------------

            for block_index in BLOCK_INDICES:

                for step in TARGET_STEPS:

                    key = (
                        block_index,
                        step,
                    )

                    delta = (
                        capture.deltas.get(
                            key
                        )
                    )

                    if delta is None:

                        raise RuntimeError(
                            f"Missing delta for "
                            f"block={block_index}, "
                            f"step={step}"
                        )

                    if key not in sae_states:

                        label = (
                            f"block_"
                            f"{block_index:02d}_"
                            f"step_"
                            f"{step:02d}"
                        )

                        sae_states[key] = (
                            SAEState(
                                d_in=delta.shape[-1],
                                label=label,
                                device=device,
                            )
                        )

                        print(
                            "Created SAE:",
                            label,
                            "d_in:",
                            delta.shape[-1],
                            "features:",
                            (
                                delta.shape[-1]
                                * SAE_EXPANSION_FACTOR
                            ),
                        )

                    sae_states[
                        key
                    ].update(
                        delta
                    )

            completed = (
                global_idx + 1
            )

            if (
                completed == 1
                or completed % 10 == 0
                or local_idx == len(df) - 1
            ):

                print(
                    f"[{completed}/{start + len(df)}] "
                    f"generation={elapsed:.1f}s"
                )

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                completed % CHECKPOINT_EVERY == 0
                or local_idx == len(df) - 1
            ):

                for key, state in (
                    sae_states.items()
                ):

                    block_index, step = key

                    path = (
                        WEIGHTS_DIR
                        / (
                            f"sae_block_"
                            f"{block_index:02d}_"
                            f"step_"
                            f"{step:02d}.pt"
                        )
                    )

                    state.save(
                        path
                    )

                    diagnostics[
                        state.label
                    ] = {
                        "block":
                            block_index,
                        "step":
                            step,
                        "d_in":
                            state.sae.d_in,
                        "n_features":
                            state.sae.n_features,
                        "k":
                            state.sae.k,
                        "num_vectors":
                            state.num_vectors,
                        "num_updates":
                            state.num_updates,
                        "mean_loss":
                            state.mean_loss,
                    }

                DIAGNOSTICS_FILE.write_text(
                    json.dumps(
                        diagnostics,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                PROGRESS_FILE.write_text(
                    json.dumps(
                        {
                            "completed":
                                completed,
                            "total":
                                start + len(df),
                            "blocks":
                                BLOCK_INDICES,
                            "steps":
                                TARGET_STEPS,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    "Checkpoint saved."
                )

            capture.clear()

            gc.collect()

            torch.cuda.empty_cache()

    finally:

        for handle in handles:
            handle.remove()

        step_wrapper.remove()

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    for key, state in (
        sae_states.items()
    ):

        block_index, step = key

        path = (
            WEIGHTS_DIR
            / (
                f"sae_block_"
                f"{block_index:02d}_"
                f"step_"
                f"{step:02d}.pt"
            )
        )

        state.save(
            path
        )

    DIAGNOSTICS_FILE.write_text(
        json.dumps(
            diagnostics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        "SAEs:",
        len(sae_states),
    )

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
