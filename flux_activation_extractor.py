"""
flux_activation_extractor.py
=============================
FLUX.1-schnell port of sd_activation_extractor.py's `--mode sae` pipeline:
SAE-based concept centroid/concentration analysis across a dataset, seeds,
and a block x timestep grid. Output format (concept_metrics.csv/.json,
centroid_alignment_summary.json) is UNCHANGED from the base script -- only
the model-specific internals differ.

WHAT'S REUSED FROM sd_activation_extractor.py, UNCHANGED (imported, not
copied): dataset loading (`load_concept_dataset`, with a CSV-repair
heuristic bugfixed there for datasets with always-populated metadata
columns), the L1-penalty `SparseAutoencoder` + `train_sae`, resumable
`IncrementalCSVWriter` / `load_existing_results`, `ProgressTracker`,
`heatmap_centroid` / `attention_concentration` / `save_heatmap_overlay`,
`sae_concept_heatmap`, `summarize_centroid_alignment`.

WHAT'S DIFFERENT, AND WHY:

1. Block structure. SD3 has one flat stack of MMDiT blocks addressed by
   index. FLUX has two kinds -- double-stream (`transformer_blocks`, image
   and text attend to each other via separate QKV projections) and
   single-stream (`single_transformer_blocks`, image+text tokens processed
   as one stream, later in the network). Blocks are addressed by name from
   a fixed set: double_early, double_mid, double_late, single_early,
   single_mid, single_late (first/middle/last block of each kind). This
   run uses double_early, double_late, single_early, single_late.

2. Raw activation, not delta. Like the base script's
   `TransformerActivationExtractor`, `FluxBlockActivationExtractor` hooks
   capture a block's raw image-stream OUTPUT (not output-input delta --
   that's a different, paper-accurate design used by this project's
   flux_sae.py for a separate experiment). Uses the same "pick whichever
   output tensor has more tokens" trick as the base script's hook to
   robustly isolate the image stream from a (text, image) tuple output,
   with an explicit text-length slice as a fallback for the case (older
   diffusers versions) where a single-stream block returns one raw
   concatenated [text; image] tensor instead of a tuple. See
   flux_sae.py's BlockDeltaRecorder docstring for the version history this
   handles.

3. One SAE per (block, timestep) cell, not per block. The base script
   trains one SAE per block, pooling activations from every captured
   denoising step, then reuses that same SAE across all evaluated
   timesteps. Here we want a genuine grid (4 blocks x 4 timesteps = 16
   SAEs, matching flux_sae.py's paper-accurate per-(block,timestep)
   design), so training activations are collected and pooled separately
   per (block, timestep) cell instead. FLUX-schnell's distilled schedule
   is only NUM_INFERENCE_STEPS=4 steps total, and the 4 default timestep
   fractions (0.0, 0.33, 0.67, 1.0) land on every single one of them --
   full trajectory coverage, no stride/subsampling needed (unlike SD3's
   28-step schedule, which needs `--train-timestep-stride`).

4. No `--mode interpret` / no attention-map capture. FLUX's attention
   mechanism differs from SD3's separate add_q_proj/add_k_proj processor,
   and `--mode sae` never uses SD3JointAttentionCapture anyway (it always
   passes capture_attention=False) -- so this file is SAE-mode only, no
   FLUX-specific attention-capture class was needed.

5. No classifier-free guidance. FLUX-schnell is guidance-distilled
   (guidance_scale=0.0), so there's no [unconditional, conditional] batch
   dimension to select out of -- generation batch size is always 1.

Everything else -- CSV schema, resumability, progress reporting,
heatmap-sampling, centroid-alignment summary -- is identical to the base
script by construction, since it's the same imported code.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBnbConfig

from sd_activation_extractor import (
    SparseAutoencoder,
    train_sae,
    load_concept_dataset,
    prepare_dataset,
    IncrementalCSVWriter,
    load_existing_results,
    ProgressTracker,
    _should_save_heatmap,
    _slug,
    heatmap_centroid,
    attention_concentration,
    save_heatmap_overlay,
    sae_concept_heatmap,
    summarize_centroid_alignment,
)


# ============================================================
# GPU selection -- same convention as this project's other FLUX
# scripts (generate_FLUX.py, flux.py): pick a GPU via
# CUDA_VISIBLE_DEVICES, set once, before torch/diffusers touch CUDA.
# ============================================================

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ============================================================
# 1. Block resolution (structural analog of SD3's flat index list)
# ============================================================

FLUX_BLOCK_NAMES = (
    "double_early", "double_mid", "double_late",
    "single_early", "single_mid", "single_late",
)


def build_flux_block_config(transformer) -> Dict[str, Tuple[str, int]]:
    """name -> (kind, index). Populated at runtime since block counts vary
    by FLUX checkpoint. 'double' = transformer.transformer_blocks (joint
    image/text attention). 'single' = transformer.single_transformer_blocks
    (concatenated sequence, later in the network)."""
    n_double = len(transformer.transformer_blocks)
    n_single = len(transformer.single_transformer_blocks)
    return {
        "double_early": ("double", 0),
        "double_mid":   ("double", n_double // 2),
        "double_late":  ("double", n_double - 1),
        "single_early": ("single", 0),
        "single_mid":   ("single", n_single // 2),
        "single_late":  ("single", n_single - 1),
    }


def get_flux_block(transformer, kind: str, index: int):
    return (transformer.transformer_blocks[index] if kind == "double"
            else transformer.single_transformer_blocks[index])


# ============================================================
# 2. FLUX block activation extractor (raw output, not delta)
# ============================================================

class FluxBlockActivationExtractor:
    """
    FLUX analog of sd_activation_extractor.py's TransformerActivationExtractor:
    forward hooks on named blocks, capturing the raw image-stream OUTPUT at
    every denoising step. Reuses that class's exact "pick whichever output
    tensor has more tokens" heuristic to isolate the image stream from a
    (encoder_hidden_states, hidden_states) tuple -- image tokens (thousands
    at typical resolutions, here 1024 at 512x512) always vastly outnumber
    text tokens (<=max_sequence_length), so this is robust and
    self-verifying rather than hard-coding tuple order. Falls back to an
    explicit text-length slice for the one case that heuristic can't cover:
    an older-diffusers single-stream block returning one raw, already-
    concatenated [text; image] tensor instead of a tuple.
    """

    def __init__(self, transformer, block_config: Dict[str, Tuple[str, int]], text_length: int):
        self.transformer = transformer
        self.block_config = block_config
        self.text_length = text_length
        self.activations: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        self._step = 0
        self._handles = []

    def _hook(self, name: str, kind: str):
        def fn(module, args, kwargs, output):
            if isinstance(output, tuple):
                candidates = [t for t in output if torch.is_tensor(t)]
                out = max(candidates, key=lambda t: t.shape[1]) if candidates else None
            elif torch.is_tensor(output):
                out = output[:, self.text_length:, :] if kind == "single" else output
            else:
                out = None
            if torch.is_tensor(out):
                self.activations[name][self._step] = out.detach().to(torch.float16).cpu()
        return fn

    def attach(self) -> "FluxBlockActivationExtractor":
        for name, (kind, index) in self.block_config.items():
            block = get_flux_block(self.transformer, kind, index)
            self._handles.append(
                block.register_forward_hook(self._hook(name, kind), with_kwargs=True)
            )
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def set_step(self, step: int) -> None:
        self._step = step

    def clear(self) -> None:
        self.activations.clear()
        self._step = 0


# ============================================================
# 3. FLUX extractor / driver (SD3InternalsExtractor analog)
# ============================================================

class FluxInternalsExtractor:
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-schnell",
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 height: int = 512, width: int = 512, max_sequence_length: int = 256,
                 load_in_8bit: bool = False):
        print(f"Loading {model_id} ...")
        if load_in_8bit:
            # enable_sequential_cpu_offload keeps every parameter resident in
            # HOST RAM (not just GPU VRAM) between offload cycles, so on a
            # memory-constrained shared box the full bf16 pipeline (T5-XXL +
            # the 12B transformer, ~40GB resident) can get OOM-killed even
            # with plenty of free GPU memory. int8-quantizing the two big
            # components roughly halves that host RAM footprint. Quantized
            # weights load straight onto the GPU during quantization, but
            # enable_sequential_cpu_offload below still moves them back to
            # CPU between forward passes -- int8 bytes, not bf16 ones.
            print("  loading transformer + T5 text encoder in int8 (bitsandbytes)...")
            transformer = FluxTransformer2DModel.from_pretrained(
                model_id, subfolder="transformer",
                quantization_config=DiffusersBnbConfig(load_in_8bit=True),
                torch_dtype=dtype,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder_2",
                quantization_config=TransformersBnbConfig(load_in_8bit=True),
                torch_dtype=dtype,
            )
            self.pipe = FluxPipeline.from_pretrained(
                model_id, transformer=transformer, text_encoder_2=text_encoder_2,
                torch_dtype=dtype,
            )
        else:
            self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        # enable_sequential_cpu_offload (not enable_model_cpu_offload) --
        # FLUX's transformer alone is large enough that whole-component
        # offload can consume an entire 24GB card with nothing left for
        # activations, especially on a shared GPU. Streams submodules onto
        # the GPU one at a time instead. Established fix, see flux_sae.py.
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        self.transformer = self.pipe.transformer
        self.height = height
        self.width = width
        self.max_sequence_length = max_sequence_length
        self.full_block_config = build_flux_block_config(self.transformer)
        self.block_extractor = FluxBlockActivationExtractor(
            self.transformer, self.full_block_config, text_length=max_sequence_length,
        )
        print("FLUX loaded. Block config:")
        for name, (kind, index) in self.full_block_config.items():
            print(f"  {name:14s} -> {kind} block #{index}")

    def extract(self, prompt: str, num_inference_steps: int = 4, guidance_scale: float = 0.0,
                capture_blocks: bool = True, block_names: Optional[List[str]] = None,
                generator=None) -> Dict:
        self.block_extractor.clear()
        self.block_extractor.block_config = (
            {n: self.full_block_config[n] for n in block_names}
            if block_names is not None else self.full_block_config
        )

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            # +1: callback_on_step_end(step_index=i) fires AFTER step i's
            # transformer forward (and its block hooks) already ran using
            # whatever step number was set by the PREVIOUS callback -- so
            # this sets the step for the NEXT forward pass, not this one.
            # Without it, step 0's hooks fire while _step is still its
            # initial 0, but so do step 1's (overwriting step 0's entry),
            # and the last step's activations are never read by any
            # forward pass -- every t=1.0 cell silently ends up empty.
            # See the identical fix + full trace in sd_activation_extractor.py.
            self.block_extractor.set_step(step_index + 1)
            return callback_kwargs

        if capture_blocks:
            self.block_extractor.attach()
        try:
            with torch.inference_mode():
                out = self.pipe(
                    prompt=prompt,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    max_sequence_length=self.max_sequence_length,
                    generator=generator,
                    callback_on_step_end=step_callback,
                )
            image = out.images[0]
        finally:
            if capture_blocks:
                self.block_extractor.detach()

        return {
            "image": image,
            "block_activations": dict(self.block_extractor.activations),
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }


def select_conditional_batch(tensor: torch.Tensor) -> torch.Tensor:
    """FLUX-schnell is guidance-distilled (guidance_scale=0.0): no CFG, no
    [unconditional, conditional] batch doubling like SD3 -- generation
    batch size is always 1. Returns (seq, dim)."""
    return tensor[0]


def probe_block_dim(extractor: "FluxInternalsExtractor", block_name: str) -> int:
    """Minimal 1-step generation solely to read a block's hidden size (for loading a saved SAE)."""
    result = extractor.extract("probe", num_inference_steps=1, guidance_scale=0.0,
                                capture_blocks=True, block_names=[block_name])
    act = next(iter(result["block_activations"].get(block_name, {}).values()), None)
    if act is None:
        raise RuntimeError(f"Could not probe hidden size for {block_name}.")
    return act.shape[-1]


# ============================================================
# 4. Collecting activations + training the (block, timestep) SAE grid
# ============================================================

def collect_block_activations_grid(extractor: FluxInternalsExtractor, dataset: List[Dict[str, str]],
                                    block_names: List[str], target_steps: set, cache_dir: str,
                                    seeds=(42,), num_inference_steps: int = 4, guidance_scale: float = 0.0,
                                    max_tokens_per_cell: int = 200_000) -> None:
    """
    Runs generation for every (prompt, seed), capturing image-token
    activations for every block in `block_names` at every step in
    `target_steps` -- a single generation contributes to ALL (block, step)
    cells at once. Stops once every cell has reached `max_tokens_per_cell`.

    Rather than pooling every generation's activations in memory (which
    grows linearly with how many prompts have been processed -- at the
    default budget that's ~1.2GB per cell, ~20GB across all 16 FLUX cells,
    ON TOP OF the model's own resident memory -- confirmed via dmesg as
    what actually OOM-killed this pipeline once the model-loading OOM
    itself was separately fixed via int8 quantization), each generation's
    per-cell tensor is written straight to its own small file under
    `cache_dir` and dropped from memory immediately. `train_saes_grid`
    reads a cell's files back in only when it's that cell's turn to train,
    so peak RAM only ever holds one cell's data at a time, not all 16.
    """
    counts: Dict[Tuple[str, int], int] = defaultdict(int)
    cells = [(b, s) for b in block_names for s in target_steps]
    for b, s in cells:
        os.makedirs(os.path.join(cache_dir, f"{b}__t{s}"), exist_ok=True)

    def all_full() -> bool:
        return all(counts[c] >= max_tokens_per_cell for c in cells)

    total_pairs = len(dataset) * len(seeds)
    tracker = ProgressTracker(total_pairs, label="collection pairs",
                               every=max(1, total_pairs // 200 or 1))

    file_idx = 0
    for item in dataset:
        if all_full():
            print("  [collect_block_activations_grid] token budget reached for all cells, stopping early.")
            break
        for seed in seeds:
            if all_full():
                break
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = extractor.extract(
                item["prompt"], num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                capture_blocks=True, block_names=block_names, generator=generator,
            )
            for b in block_names:
                for step, act in result["block_activations"].get(b, {}).items():
                    key = (b, step)
                    if step not in target_steps or counts[key] >= max_tokens_per_cell:
                        continue
                    cond = select_conditional_batch(act)  # (N_img, d_in)
                    cell_dir = os.path.join(cache_dir, f"{b}__t{step}")
                    torch.save(cond, os.path.join(cell_dir, f"{file_idx:06d}.pt"))
                    counts[key] += cond.shape[0]
            file_idx += 1
            tracker.step()
            gc.collect()
            torch.cuda.empty_cache()


def load_cell_activations(cache_dir: str, block: str, step: int) -> Optional[torch.Tensor]:
    cell_dir = os.path.join(cache_dir, f"{block}__t{step}")
    if not os.path.isdir(cell_dir):
        return None
    files = sorted(os.listdir(cell_dir))
    if not files:
        return None
    tensors = [torch.load(os.path.join(cell_dir, f)) for f in files]
    return torch.cat(tensors, dim=0)


def train_saes_grid(extractor: FluxInternalsExtractor, dataset: List[Dict[str, str]],
                     block_names: List[str], timestep_fractions: Tuple[float, ...], seeds=(42,),
                     num_inference_steps: int = 4, guidance_scale: float = 0.0,
                     max_tokens_per_cell: int = 200_000, d_hidden: int = 4096, l1_coeff: float = 1e-3,
                     sae_epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3,
                     device: str = "cuda", save_dir: str = "./sae_checkpoints",
                     cache_dir: Optional[str] = None, keep_cache: bool = False,
                     ) -> Dict[Tuple[str, float], SparseAutoencoder]:
    step_indices = sorted({int(round(f * (num_inference_steps - 1))) for f in timestep_fractions})
    frac_by_step = {int(round(f * (num_inference_steps - 1))): f for f in timestep_fractions}
    target_steps = set(step_indices)
    cache_dir = cache_dir or os.path.join(save_dir, "_activation_cache")

    print(f"Collecting activations from {block_names} x steps {sorted(target_steps)} "
          f"(fractions {timestep_fractions}) across {len(dataset)} prompt(s) x {len(seeds)} seed(s)...")
    collect_block_activations_grid(
        extractor, dataset, block_names, target_steps, cache_dir, seeds=seeds,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        max_tokens_per_cell=max_tokens_per_cell,
    )

    os.makedirs(save_dir, exist_ok=True)
    saes: Dict[Tuple[str, float], SparseAutoencoder] = {}
    for block in block_names:
        for step in step_indices:
            frac = frac_by_step[step]
            acts = load_cell_activations(cache_dir, block, step)
            if acts is None or acts.shape[0] == 0:
                print(f"  [skip] no activations collected for {block} @ t={frac:.2f}")
                continue
            label = f"{block}/t{frac:.2f}"
            print(f"\n  Training SAE for {label}  ({acts.shape[0]} tokens, d_in={acts.shape[1]})")
            sae = train_sae(acts, d_hidden=d_hidden, l1_coeff=l1_coeff, epochs=sae_epochs,
                             batch_size=batch_size, lr=lr, device=device)
            saes[(block, frac)] = sae
            torch.save(sae.state_dict(), os.path.join(save_dir, f"sae_{block}_t{frac:.2f}.pt"))
            # Free this cell's activations (a fresh few-hundred-MB to ~1GB
            # tensor per cell) before moving to the next one, rather than
            # letting all 16 cells' worth pile up -- the exact accumulation
            # pattern that caused the OOM this function now avoids.
            del acts
            gc.collect()
    if not keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return saes


# ============================================================
# 5. Concept evaluation -- same CSV/JSON schema as the base script
# ============================================================

def evaluate_concepts(extractor: FluxInternalsExtractor, dataset: List[Dict[str, str]],
                       saes: Dict[Tuple[str, float], SparseAutoencoder], block_names: List[str],
                       seeds=(42, 45, 12, 1000), timestep_fractions=(0.0, 0.33, 0.67, 1.0),
                       num_inference_steps: int = 4, guidance_scale: float = 0.0,
                       heatmap_out_size: int = 64, out_dir: str = "./flux_sae",
                       save_heatmap_images: bool = True, heatmap_sample_rate: float = 1.0,
                       resume: bool = False) -> List[Dict]:
    """
    For every concept x seed x requested timestep fraction x block: encodes
    that block's image-token activations at that timestep with its own
    (block, timestep) SAE, finds the concept latent, builds its spatial
    heatmap, and records the heatmap's centroid (x, y) and concentration
    score. Identical row schema and incremental-write/resume behavior to
    sd_activation_extractor.py's evaluate_concepts -- see that function's
    docstring for the full rationale.
    """
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    heatmaps_dir = os.path.join(out_dir, "heatmaps")
    if save_heatmap_images:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(heatmaps_dir, exist_ok=True)

    step_indices = sorted({int(round(f * (num_inference_steps - 1))) for f in timestep_fractions})
    frac_by_step = {int(round(f * (num_inference_steps - 1))): f for f in timestep_fractions}

    csv_path = os.path.join(out_dir, "concept_metrics.csv")
    fieldnames = ["concept_id", "prompt", "block", "seed", "step_index", "timestep_fraction",
                  "latent_idx", "centroid_x", "centroid_y", "concentration", "heatmap_overlay"]

    rows: List[Dict] = []
    done_pairs: set = set()
    if resume:
        rows, done_pairs = load_existing_results(csv_path, block_names=block_names,
                                                   timestep_fractions=timestep_fractions)
        if done_pairs:
            print(f"[evaluate_concepts] resuming: {len(done_pairs)} (concept, seed) pair(s) "
                  f"already done, {len(rows)} row(s) loaded from {csv_path}")

    writer = IncrementalCSVWriter(csv_path, fieldnames)
    writer.open(resume=resume)

    total_pairs = len(dataset) * len(seeds)
    tracker = ProgressTracker(total_pairs, label="(concept, seed) pairs",
                               every=max(1, total_pairs // 200 or 1))
    skipped = 0

    try:
        for item in dataset:
            concept_id, prompt = item["concept_id"], item["prompt"]
            concept_slug = _slug(concept_id)
            for seed in seeds:
                if (concept_id, seed) in done_pairs:
                    skipped += 1
                    tracker.step()
                    continue

                save_media = save_heatmap_images and _should_save_heatmap(
                    concept_id, seed, heatmap_sample_rate)

                generator = torch.Generator(device="cpu").manual_seed(seed)
                result = extractor.extract(
                    prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                    capture_blocks=True, block_names=block_names, generator=generator,
                )
                image = result["image"]
                if save_media:
                    image.save(os.path.join(images_dir, f"{concept_slug}_seed{seed}.png"))

                for block in block_names:
                    block_slug = _slug(block)
                    per_step = result["block_activations"].get(block, {})
                    for step in step_indices:
                        act = per_step.get(step)
                        frac = frac_by_step[step]
                        sae = saes.get((block, frac))
                        if act is None or sae is None:
                            continue
                        cond = select_conditional_batch(act)
                        try:
                            heat, latent_idx = sae_concept_heatmap(sae, cond, out_size=heatmap_out_size)
                        except ValueError:
                            continue
                        cx, cy = heatmap_centroid(heat)
                        conc = attention_concentration(heat)

                        row = {
                            "concept_id": concept_id, "prompt": prompt, "block": block, "seed": seed,
                            "step_index": step, "timestep_fraction": frac,
                            "latent_idx": latent_idx, "centroid_x": cx, "centroid_y": cy,
                            "concentration": conc, "heatmap_overlay": "",
                        }
                        if save_media:
                            full_heat = F.interpolate(
                                heat.unsqueeze(0).unsqueeze(0), size=image.size[::-1],
                                mode="bilinear", align_corners=False,
                            ).squeeze(0).squeeze(0)
                            overlay_path = os.path.join(
                                heatmaps_dir, f"{concept_slug}_seed{seed}_{block_slug}_t{frac:.2f}.png",
                            )
                            save_heatmap_overlay(image, full_heat, overlay_path)
                            row["heatmap_overlay"] = overlay_path

                        writer.write(row)
                        rows.append(row)

                tracker.step(extra=f"last: {concept_id[:30]!r} seed={seed}")
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        writer.close()

    with open(os.path.join(out_dir, "concept_metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nDone. {len(rows)} total row(s) in {csv_path} ({skipped} pair(s) skipped via --resume).")
    if save_heatmap_images:
        print(f"Saved images -> {images_dir}/  and heatmap overlays -> {heatmaps_dir}/ "
              f"(sampled at rate={heatmap_sample_rate})")
    return rows


# ============================================================
# 6. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FLUX SAE-based concept centroid/concentration analysis across a "
                     "(block, timestep) grid, dataset, and seeds. Direct FLUX port of "
                     "sd_activation_extractor.py's --mode sae pipeline -- same output "
                     "format (concept_metrics.csv/.json, centroid_alignment_summary.json). "
                     "See the module docstring for what's different and why."
    )
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--load-in-8bit", action="store_true",
                         help="Load the transformer and T5 text encoder int8-quantized "
                              "(bitsandbytes). Roughly halves host RAM use under "
                              "enable_sequential_cpu_offload (~40GB -> ~20-25GB for schnell) "
                              "at the cost of studying a quantized model's activations "
                              "rather than the original bf16 model's.")

    parser.add_argument("--dataset", type=str, required=True,
                         help="Path to a .txt/.json/.csv concept/prompt dataset.")
    parser.add_argument("--sae-blocks", type=str,
                         default="double_early,double_late,single_early,single_late",
                         help=f"Comma-separated FLUX block names. Choices: {', '.join(FLUX_BLOCK_NAMES)}.")
    parser.add_argument("--seeds", type=str, default="42,45,12,1000")
    parser.add_argument("--timesteps", type=str, default="0.0,0.33,0.67,1.0",
                         help="Fractions of the denoising trajectory to train/evaluate a "
                              "separate SAE at (0=first/noisiest step, 1=last/cleanest step). "
                              "One SAE is trained per (block, timestep) pair.")
    parser.add_argument("--sae-hidden", type=int, default=4096)
    parser.add_argument("--sae-l1", type=float, default=1e-3)
    parser.add_argument("--sae-epochs", type=int, default=10)
    parser.add_argument("--sae-batch-size", type=int, default=1024)
    parser.add_argument("--sae-lr", type=float, default=1e-3)
    parser.add_argument("--max-tokens-per-cell", type=int, default=200_000,
                         help="Token budget PER (block, timestep) cell during SAE training "
                              "collection (analogous to the base script's "
                              "--max-tokens-per-block, renamed since each cell now trains "
                              "its own SAE instead of sharing one per block).")
    parser.add_argument("--sae-dir", type=str, default="./sae_checkpoints")
    parser.add_argument("--load-sae", action="store_true",
                         help="Load existing SAE checkpoints from --sae-dir instead of retraining.")
    parser.add_argument("--heatmap-out-size", type=int, default=64)
    parser.add_argument("--no-heatmap-images", action="store_true")
    parser.add_argument("--heatmap-sample-rate", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    block_names = [b.strip() for b in args.sae_blocks.split(",")]
    unknown = set(block_names) - set(FLUX_BLOCK_NAMES)
    if unknown:
        raise ValueError(f"Unknown block name(s) {unknown}; choose from {FLUX_BLOCK_NAMES}")

    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    timestep_fractions = tuple(float(t.strip()) for t in args.timesteps.split(","))
    out_dir = args.out_dir or "./flux_sae"

    raw_dataset = load_concept_dataset(args.dataset)
    shuffle_seed = None if args.shuffle_seed < 0 else args.shuffle_seed
    dataset = prepare_dataset(raw_dataset, limit=args.limit, shuffle_seed=shuffle_seed)
    print(f"[main] using {len(dataset)}/{len(raw_dataset)} dataset entries "
          f"(shuffle_seed={shuffle_seed}, limit={args.limit})")

    extractor = FluxInternalsExtractor(
        args.model_id, device=args.device, height=args.height, width=args.width,
        max_sequence_length=args.max_sequence_length, load_in_8bit=args.load_in_8bit,
    )

    step_indices = sorted({int(round(f * (args.steps - 1))) for f in timestep_fractions})
    frac_by_step = {int(round(f * (args.steps - 1))): f for f in timestep_fractions}

    if args.load_sae:
        print(f"Loading SAE checkpoints from {args.sae_dir} ...")
        saes = {}
        for block in block_names:
            for step in step_indices:
                frac = frac_by_step[step]
                ckpt_path = os.path.join(args.sae_dir, f"sae_{block}_t{frac:.2f}.pt")
                if not os.path.exists(ckpt_path):
                    print(f"  [skip] no checkpoint found for {block}/t{frac:.2f} at {ckpt_path}")
                    continue
                d_in = probe_block_dim(extractor, block)
                sae = SparseAutoencoder(d_in, args.sae_hidden, l1_coeff=args.sae_l1).to(args.device)
                sae.load_state_dict(torch.load(ckpt_path, map_location=args.device))
                saes[(block, frac)] = sae
    else:
        saes = train_saes_grid(
            extractor, dataset, block_names, timestep_fractions, seeds=seeds,
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            max_tokens_per_cell=args.max_tokens_per_cell, d_hidden=args.sae_hidden,
            l1_coeff=args.sae_l1, sae_epochs=args.sae_epochs, batch_size=args.sae_batch_size,
            lr=args.sae_lr, device=args.device, save_dir=args.sae_dir,
        )

    if not saes:
        raise RuntimeError("No SAEs available (training produced none and/or nothing was loaded).")

    total_generations = len(dataset) * len(seeds)
    print(f"\n[main] evaluation phase: {len(dataset)} concepts x {len(seeds)} seeds "
          f"= {total_generations} generations, x up to {len(saes)} (block, timestep) SAE(s) "
          f"= up to {total_generations * len(saes)} metric rows.\n")

    rows = evaluate_concepts(
        extractor, dataset, saes, block_names, seeds=seeds, timestep_fractions=timestep_fractions,
        num_inference_steps=args.steps, guidance_scale=args.guidance,
        heatmap_out_size=args.heatmap_out_size, out_dir=out_dir,
        save_heatmap_images=not args.no_heatmap_images,
        heatmap_sample_rate=args.heatmap_sample_rate, resume=args.resume,
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
