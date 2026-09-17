"""
pixart_activation_extractor.py
================================
PixArt-alpha port of sd_activation_extractor.py's --mode sae pipeline, for use
with the PixArt SAE checkpoints (sae_transformer_blocks_{4,8,12,16,20}.pt,
d_in=1152, matching PixArt-alpha's transformer hidden size).

Unlike FLUX (which needed flux_activation_extractor.py, a larger port), PixArt
needs almost nothing new: its transformer is a single flat stack of
BasicTransformerBlock modules (pipe.transformer.transformer_blocks), addressed
by index exactly like SD3, and each block returns one hidden_states tensor
(not a tuple), which the base script's TransformerActivationExtractor hook
already handles via its existing "not a tuple" branch. PixArt also uses
ordinary classifier-free guidance (guidance_scale=4.5), matching SD3's
[unconditional, conditional] batch convention, so select_conditional_batch
is reused unchanged too.

Everything model-agnostic (dataset loading, the SAE class, evaluate_concepts,
collect_block_activations, train_sae_on_blocks, resumable CSV writing,
centroid-alignment summary) is imported directly from sd_activation_extractor.py.
The only new code here is PixArtInternalsExtractor, matching
SD3InternalsExtractor's method signature so it's a drop-in substitute.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import PixArtAlphaPipeline

from sd_activation_extractor import (
    TransformerActivationExtractor,
    SAEFeaturePatcher,
    SparseAutoencoder,
    train_sae_on_blocks,
    load_concept_dataset,
    prepare_dataset,
    evaluate_concepts,
    summarize_centroid_alignment,
)


class PixArtInternalsExtractor:
    """Drop-in substitute for SD3InternalsExtractor: same method signature,
    so evaluate_concepts / collect_block_activations / train_sae_on_blocks /
    probe_block_dim all work against it completely unchanged."""

    def __init__(self, model_id: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
                 device: str = "cuda", dtype: torch.dtype = torch.float16,
                 height: int = 1024, width: int = 1024):
        print(f"Loading {model_id} ...")
        self.pipe = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=dtype)
        # Shared, not dedicated, GPUs -- see the identical fix + rationale
        # already applied to SD3InternalsExtractor in sd_activation_extractor.py.
        self.pipe.enable_model_cpu_offload()
        self.transformer = self.pipe.transformer
        self.height = height
        self.width = width
        self.block_extractor = TransformerActivationExtractor(self.transformer)

    def extract(self, prompt: str, num_inference_steps: int = 20, guidance_scale: float = 4.5,
                capture_blocks: bool = False, capture_attention: bool = False,
                block_names: Optional[List[str]] = None,
                attn_layer_names=None, heatmap_size: int = 32, keep_heads: bool = False,
                block_step_stride: int = 1, generator=None,
                patcher: Optional["SAEFeaturePatcher"] = None) -> Dict:
        self.block_extractor.clear()
        self.block_extractor.step_stride = max(1, block_step_stride)
        if block_names is not None:
            self.block_extractor.block_names = block_names

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            # +1: callback fires AFTER step_index's forward pass already ran
            # using the previous step's number -- see the identical fix and
            # full trace in sd_activation_extractor.py's SD3InternalsExtractor.
            self.block_extractor.set_step(step_index + 1)
            if patcher is not None:
                patcher.set_step(step_index + 1)
            return callback_kwargs

        if capture_blocks:
            self.block_extractor.attach()
        if patcher is not None:
            patcher.set_step(0)
            patcher.attach(self.transformer)
        try:
            out = self.pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=step_callback,
            )
            image = out.images[0]
        finally:
            if capture_blocks:
                self.block_extractor.detach()
            if patcher is not None:
                patcher.detach()

        return {
            "image": image,
            "block_activations": dict(self.block_extractor.activations),
            "attention_maps": {},
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }


def main():
    parser = argparse.ArgumentParser(
        description="PixArt-alpha SAE-based concept centroid/concentration analysis. "
                     "Direct PixArt port of sd_activation_extractor.py's --mode sae pipeline "
                     "(identical output format: concept_metrics.csv/.json, "
                     "centroid_alignment_summary.json)."
    )
    parser.add_argument("--model-id", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sae-blocks", type=str, default="4,8,12,16,20")
    parser.add_argument("--seeds", type=str, default="42,45,12,1000")
    parser.add_argument("--timesteps", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--sae-hidden", type=int, default=4096)
    parser.add_argument("--sae-l1", type=float, default=1e-3)
    parser.add_argument("--sae-epochs", type=int, default=10)
    parser.add_argument("--sae-batch-size", type=int, default=1024)
    parser.add_argument("--sae-lr", type=float, default=1e-3)
    parser.add_argument("--train-timestep-stride", type=int, default=4)
    parser.add_argument("--max-tokens-per-block", type=int, default=200_000)
    parser.add_argument("--sae-dir", type=str, default="./sae_checkpoints_provided")
    parser.add_argument("--load-sae", action="store_true")
    parser.add_argument("--heatmap-out-size", type=int, default=64)
    parser.add_argument("--no-heatmap-images", action="store_true")
    parser.add_argument("--heatmap-sample-rate", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--latent-top-k", type=int, default=64)
    parser.add_argument("--latent-map-size", type=int, default=32)
    args = parser.parse_args()

    raw_dataset = load_concept_dataset(args.dataset)
    shuffle_seed = None if args.shuffle_seed < 0 else args.shuffle_seed
    dataset = prepare_dataset(raw_dataset, limit=args.limit, shuffle_seed=shuffle_seed)
    print(f"[main] using {len(dataset)}/{len(raw_dataset)} dataset entries "
          f"(shuffle_seed={shuffle_seed}, limit={args.limit})")

    block_names = [f"transformer_blocks.{i.strip()}" for i in args.sae_blocks.split(",")]
    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    timestep_fractions = tuple(float(t.strip()) for t in args.timesteps.split(","))
    out_dir = args.out_dir or "./pixart_sae"

    extractor = PixArtInternalsExtractor(
        args.model_id, device=args.device, height=args.height, width=args.width,
    )

    if args.load_sae:
        print(f"Loading SAE checkpoints from {args.sae_dir} ...")
        saes = {}
        for block in block_names:
            ckpt_path = os.path.join(args.sae_dir, f"sae_{block.replace('.', '_')}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [skip] no checkpoint found for {block} at {ckpt_path}")
                continue
            # Read d_in from the checkpoint itself instead of probe_block_dim()'s
            # 1-step generation -- the checkpoint already encodes its own shape,
            # and a live probe isn't needed just to load an existing SAE. This
            # also sidesteps a real diffusers bug: PixArtAlphaPipeline's
            # num_inference_steps==1 branch indexes scheduler.step(...)[1], but
            # DPMSolverMultistepScheduler.step() in the installed diffusers
            # version always returns a 1-tuple, so any 1-step generation
            # (which is all probe_block_dim() ever does) raises
            # "IndexError: tuple index out of range" unconditionally.
            state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            d_in = state_dict["encoder.weight"].shape[1]
            sae = SparseAutoencoder(d_in, args.sae_hidden, l1_coeff=args.sae_l1).to(args.device)
            sae.load_state_dict(state_dict)
            saes[block] = sae
    else:
        saes = train_sae_on_blocks(
            extractor, dataset, block_names, seeds=seeds, num_inference_steps=args.steps,
            guidance_scale=args.guidance, timestep_stride=args.train_timestep_stride,
            max_tokens_per_block=args.max_tokens_per_block, d_hidden=args.sae_hidden,
            l1_coeff=args.sae_l1, sae_epochs=args.sae_epochs, batch_size=args.sae_batch_size,
            lr=args.sae_lr, device=args.device, save_dir=args.sae_dir,
        )

    if not saes:
        raise RuntimeError("No SAEs available (training produced none and/or nothing was loaded).")

    total_generations = len(dataset) * len(seeds)
    print(f"\n[main] evaluation phase: {len(dataset)} concepts x {len(seeds)} seeds "
          f"= {total_generations} generations, x {len(block_names)} block(s) x "
          f"{len(timestep_fractions)} timestep(s) = up to "
          f"{total_generations * len(block_names) * len(timestep_fractions)} metric rows.\n")

    rows = evaluate_concepts(
        extractor, dataset, saes, seeds=seeds, timestep_fractions=timestep_fractions,
        num_inference_steps=args.steps, guidance_scale=args.guidance,
        heatmap_out_size=args.heatmap_out_size, out_dir=out_dir, device=args.device,
        save_heatmap_images=not args.no_heatmap_images,
        heatmap_sample_rate=args.heatmap_sample_rate, resume=args.resume,
        latents_dir=os.path.join(out_dir, "latents") if args.save_latents else None,
        latent_top_k=args.latent_top_k, latent_map_size=args.latent_map_size,
    )
    summary = summarize_centroid_alignment(rows)
    with open(os.path.join(out_dir, "centroid_alignment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved centroid-alignment summary -> {out_dir}/centroid_alignment_summary.json")

    del extractor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
