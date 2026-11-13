from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-dev"
HF_TOKEN = os.environ.get("HF_TOKEN")

HEIGHT = 512
WIDTH = 512

NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
MAX_SEQUENCE_LENGTH = 512
DTYPE = torch.bfloat16

BLOCK_NAME = "double_mid"
BLOCK_KIND = "double"
BLOCK_INDEX = 10

TARGET_STEPS = {
    "t1.0": 0,
    "t0.5": NUM_INFERENCE_STEPS // 2,
    "t0.0": NUM_INFERENCE_STEPS - 1,
}

STEP_TO_BUCKET = {
    step: bucket
    for bucket, step in TARGET_STEPS.items()
}

WEIGHTS_DIR = Path(
    "flux_sae_training/weights"
)

OUTPUT_DIR = Path(
    "sae_interpretation"
)

TOP_K_CIDS = 20


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

    return sae


# ============================================================
# STEP TRACKING
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

        self.transformer.forward = wrapped

    def remove(self):
        self.transformer.forward = (
            self.original
        )


# ============================================================
# CAPTURE
# ============================================================

class Capture:
    def __init__(
        self,
        tracker,
    ):
        self.tracker = tracker
        self.input = None
        self.deltas = {
            step: None
            for step in TARGET_STEPS.values()
        }

    def pre(
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
                "Could not find hidden_states."
            )

        self.input = (
            hidden_states
            .detach()[0]
            .float()
            .cpu()
            .clone()
        )

    def post(
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
            out = (
                output[1]
                .detach()[0]
                .float()
                .cpu()
                .clone()
            )
        else:
            out = (
                output
                .detach()[0]
                .float()
                .cpu()
                .clone()
            )

        if out.shape != self.input.shape:
            raise RuntimeError(
                "FLUX input/output shape mismatch: "
                f"{tuple(self.input.shape)} vs "
                f"{tuple(out.shape)}"
            )

        self.deltas[
            step
        ] = (
            out - self.input
        )

        self.input = None

        # IMPORTANT:
        # return original FLUX output unchanged.
        return output


# ============================================================
# SIMPLE SPATIAL MAP
# ============================================================

def grid_shape(
    n,
):
    side = int(
        round(
            np.sqrt(n)
        )
    )

    if side * side == n:
        return side, side

    h = int(
        np.floor(
            np.sqrt(n)
        )
    )

    while h > 1:
        if n % h == 0:
            return h, n // h
        h -= 1

    return 1, n


def activation_map(
    values,
):
    values = np.maximum(
        np.asarray(
            values,
            dtype=np.float32,
        ),
        0,
    )

    h, w = grid_shape(
        values.size
    )

    amap = values.reshape(
        h,
        w,
    )

    mx = amap.max()

    if mx > 0:
        amap = amap / mx

    return amap


def save_heatmap(
    amap,
    path,
):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        amap,
        interpolation="nearest",
    )

    plt.axis("off")
    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_CIDS,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
    )

    output_dir = (
        OUTPUT_DIR
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load SAE
    # --------------------------------------------------------

    saes = {}

    for bucket in TARGET_STEPS:
        print(
            f"Loading SAE {bucket}..."
        )

        saes[
            bucket
        ] = load_sae(
            bucket,
            device,
        )

    # --------------------------------------------------------
    # Load FLUX
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
            f"double_mid block {BLOCK_INDEX} "
            f"does not exist. "
            f"Available double blocks: "
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

    capture = Capture(
        tracker
    )

    handles = [
        block.register_forward_pre_hook(
            capture.pre,
            with_kwargs=True,
        ),
        block.register_forward_hook(
            capture.post,
            with_kwargs=True,
        ),
    ]

    # --------------------------------------------------------
    # Normal FLUX generation
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("GENERATING NORMAL FLUX IMAGE")
    print("=" * 72)
    print(args.prompt)
    print()

    tracker.step = -1

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(
        args.seed
    )

    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            generator=generator,
        )

    # Remove hooks after normal generation.
    for handle in handles:
        handle.remove()

    wrapper.remove()

    image_path = (
        output_dir
        / "flux_image.png"
    )

    result.images[0].save(
        image_path
    )

    # --------------------------------------------------------
    # Analyze SAE representations
    # --------------------------------------------------------

    report = {
        "prompt":
            args.prompt,
        "seed":
            args.seed,
        "image":
            str(image_path),
        "block":
            BLOCK_NAME,
        "block_index":
            BLOCK_INDEX,
        "timesteps":
            {},
    }

    for step, bucket in (
        STEP_TO_BUCKET.items()
    ):
        delta = capture.deltas[
            step
        ]

        if delta is None:
            print(
                f"WARNING: no residual for {bucket}"
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

        # Average activation across image tokens.
        mean_activation = (
            z.mean(
                dim=0
            )
            .cpu()
            .numpy()
        )

        top_k = min(
            args.top_k,
            mean_activation.size,
        )

        values, indices = torch.topk(
            z.mean(
                dim=0
            ),
            top_k,
        )

        values = (
            values.cpu()
            .numpy()
        )

        indices = (
            indices.cpu()
            .numpy()
        )

        bucket_dir = (
            output_dir
            / bucket
        )

        bucket_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cid_list = []

        for cid, value in zip(
            indices,
            values,
        ):
            cid = int(cid)
            value = float(value)

            # Spatial activation for this CID.
            cid_map = activation_map(
                z[:, cid]
                .cpu()
                .numpy()
            )

            map_path = (
                bucket_dir
                / f"cid_{cid}.png"
            )

            save_heatmap(
                cid_map,
                map_path,
            )

            cid_list.append(
                {
                    "cid":
                        cid,
                    "mean_activation":
                        value,
                    "max_activation":
                        float(
                            z[:, cid]
                            .max()
                            .item()
                        ),
                    "map":
                        str(
                            map_path
                        ),
                }
            )

        report[
            "timesteps"
        ][
            bucket
        ] = {
            "step":
                step,
            "top_cids":
                cid_list,
            "num_tokens":
                int(
                    x.shape[0]
                ),
            "active_features_per_token":
                float(
                    z.ne(0)
                    .sum(
                        dim=-1
                    )
                    .float()
                    .mean()
                    .item()
                ),
        }

        # Save raw top activation information.
        np.save(
            bucket_dir
            / "mean_cid_activation.npy",
            mean_activation,
        )

        print()
        print(
            f"{bucket} "
            f"(FLUX step {step})"
        )

        for item in cid_list:
            print(
                f"  CID "
                f"{item['cid']:5d} "
                f"activation "
                f"{item['mean_activation']:.4f}"
            )

    report_path = (
        output_dir
        / "report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del pipe

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Normal FLUX image: "
        f"{image_path}"
    )
    print(
        f"SAE report: "
        f"{report_path}"
    )
    print(
        f"Output directory: "
        f"{output_dir}"
    )
    print()



if __name__ == "__main__":
    main()