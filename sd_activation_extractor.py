"""
sd3_activation_extractor.py
===========================
Extract internal activations and image<->text attention from Stable
Diffusion 3 (SD3, MMDiT), then interpret a generated image: per-word
attribution heatmaps, centroids, and concentration scores.

Fixes two bugs present in an earlier version of this file:

1. OOM ("Killed", no traceback): SD3's joint attention operates over the
   *concatenated* [image_tokens ; text_tokens] sequence, so the raw
   attention matrix is (image+text) x (image+text) -- at 1024x1024 that's
   roughly 4250 x 4250 per head. Storing that unmodified for every one of
   ~24 transformer blocks at every one of ~28 denoising steps works out to
   multiple terabytes of host RAM, which is what gets your process killed
   by the Linux OOM killer. This version keeps ONLY the image-queries ->
   text-keys slice, averages/downsamples it immediately inside the hook,
   and stores in fp16 -- a >10,000x reduction, with no loss of the
   information you actually need (per-word spatial attribution).

2. Correctness: SD3's real joint-attention processor uses separate
   projection weights for the text/context stream (`add_q_proj` /
   `add_k_proj` / `add_v_proj`), not the image stream's `to_q`/`to_k`/`to_v`.
   Reusing the image projections for text (as before) doesn't reflect the
   model's actual computation. This version uses the dedicated projections
   when present, falling back gracefully otherwise.

Requirements:
    pip install -U diffusers transformers accelerate safetensors pillow matplotlib
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention

# Poem/story `prompt` fields can comfortably exceed Python csv's default
# per-field parse limit (131072 bytes), which raises
# "_csv.Error: field larger than field limit" when reading a large
# concept_metrics.csv back in (e.g. for --resume). Raise it up front, with
# the standard OverflowError-safe fallback for platforms where sys.maxsize
# doesn't fit the underlying C long.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


# ============================================================
# 1. Generic module lookup
# ============================================================

def get_by_path(root: torch.nn.Module, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def list_transformer_blocks(transformer) -> List[str]:
    """SD3 has no down/mid/up blocks -- just a flat stack of MMDiT blocks."""
    return [f"transformer_blocks.{i}" for i in range(len(transformer.transformer_blocks))]


# ============================================================
# 2. Transformer block activation extractor (with memory controls)
# ============================================================

class TransformerActivationExtractor:
    """
    Forward hooks on named transformer blocks. `step_stride` and
    `block_names` let you deliberately subsample what's captured -- with
    24 blocks x 28 steps of full hidden states (batch x ~4250 x 1536),
    capturing everything is ~15-20 GB even before attention is considered,
    so don't turn this on by default for a plain interpret run.
    """

    def __init__(self, transformer, block_names: Optional[List[str]] = None,
                 step_stride: int = 1):
        self.transformer = transformer
        self.block_names = block_names or list_transformer_blocks(transformer)
        self.step_stride = max(1, step_stride)
        self.activations: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        self._step = 0
        self._handles = []

    def _hook(self, name: str):
        def fn(module, inputs, output):
            if self._step % self.step_stride != 0:
                return
            if isinstance(output, tuple):
                # diffusers' JointTransformerBlock returns
                # (encoder_hidden_states, hidden_states) i.e. (text, image)
                # for every block except the last (context_pre_only=True),
                # which returns just `hidden_states`. Rather than hard-code
                # that ordering (fragile if it ever changes upstream), pick
                # whichever tensor has the longer sequence length -- image
                # tokens (thousands, at typical resolutions) always vastly
                # outnumber text tokens (~150-300), so this is a robust,
                # self-verifying way to grab the image stream specifically.
                candidates = [t for t in output if torch.is_tensor(t)]
                out = max(candidates, key=lambda t: t.shape[1]) if candidates else None
            else:
                out = output
            if torch.is_tensor(out):
                self.activations[name][self._step] = out.detach().to(torch.float16).cpu()
        return fn

    def attach(self) -> "TransformerActivationExtractor":
        for name in self.block_names:
            module = get_by_path(self.transformer, name)
            self._handles.append(module.register_forward_hook(self._hook(name)))
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
# 2b. SAE feature ablation/amplification hook -- causal intervention
# ============================================================

class SAEFeaturePatcher:
    """
    Modify-in-place forward hook (unlike TransformerActivationExtractor,
    which only observes) that zeroes or amplifies ONE SAE latent's
    contribution to ONE transformer block's output, at ONE specific
    denoising step, on the conditional half of the batch only (under CFG
    the unconditional half is untouched). Every other forward pass --
    other blocks, other steps -- runs completely unmodified, so any change
    in the final image is attributable to exactly this single
    intervention. This is the causal necessity/sufficiency mechanism in
    writeups/ablation_protocol.html: subtract the latent's own decoder
    direction, scaled by its own per-token activation, to ablate it;
    add a multiple of that same delta to amplify it.

    latent_idx and target_step should come straight from a
    concept_metrics.csv row (or top_concepts_top32.csv), so the patch is
    applied at exactly the (block, step) the metric itself measured --
    not recomputed, to guarantee the causal test is about the same
    latent the metric actually named.
    """

    def __init__(self, sae: "SparseAutoencoder", latent_idx: int, block_name: str,
                 target_step: int, mode: str = "ablate", factor: float = 3.0,
                 do_cfg: bool = True):
        assert mode in ("ablate", "amplify"), f"mode must be 'ablate' or 'amplify', got {mode!r}"
        self.sae = sae
        self.latent_idx = latent_idx
        self.block_name = block_name
        self.target_step = target_step
        self.mode = mode
        self.factor = factor
        self.do_cfg = do_cfg
        self._step = 0
        self._handle = None
        # Set True the one time the hook actually patches something -- lets
        # a caller assert the intervention really fired rather than silently
        # no-op'ing because target_step/block_name never matched.
        self.applied = False

    def set_step(self, step: int) -> None:
        self._step = step

    def _patch_tensor(self, image_hidden: torch.Tensor) -> torch.Tensor:
        # image_hidden: (batch, N_img, d_in); batch=2 as [uncond, cond] under CFG.
        sae_device = next(self.sae.parameters()).device
        cond = image_hidden[1:2] if self.do_cfg else image_hidden[0:1]  # (1, N_img, d_in)
        orig_dtype = cond.dtype
        with torch.no_grad():
            x = cond.to(sae_device).float()
            z = self.sae.encode(x)                              # (1, N_img, d_hidden)
            z_i = z[..., self.latent_idx]                        # (1, N_img) -- per-token activation
            d_i = self.sae.decoder.weight[:, self.latent_idx]    # (d_in,) -- this latent's fixed direction
            delta = z_i.unsqueeze(-1) * d_i                      # (1, N_img, d_in)
            if self.mode == "ablate":
                patched = x - delta
            else:
                patched = x + (self.factor - 1) * delta
            patched = patched.to(orig_dtype).to(image_hidden.device)
        self.applied = True
        if self.do_cfg:
            return torch.cat([image_hidden[0:1], patched], dim=0)
        return patched

    def _hook(self, module, inputs, output):
        if self._step != self.target_step:
            return None  # not our step -- leave output untouched
        if isinstance(output, tuple):
            # Same "pick the longer sequence" rule TransformerActivationExtractor
            # uses to find the image stream among (text, image) or (image,).
            candidates = [(i, t) for i, t in enumerate(output) if torch.is_tensor(t)]
            if not candidates:
                return None
            img_idx, img_tensor = max(candidates, key=lambda it: it[1].shape[1])
            patched = self._patch_tensor(img_tensor)
            new_output = list(output)
            new_output[img_idx] = patched
            return tuple(new_output)
        else:
            return self._patch_tensor(output)

    def attach(self, transformer) -> "SAEFeaturePatcher":
        module = get_by_path(transformer, self.block_name)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ============================================================
# 3. SD3 joint-attention capture -- the actual OOM fix lives here
# ============================================================

class SD3JointAttentionCapture:
    """
    Captures image<->text attention from one SD3 joint-attention layer.

    Unlike the naive version, this NEVER holds the full
    (image+text) x (image+text) matrix in memory after the hook returns:
    it slices out only image-queries -> text-keys, collapses heads
    immediately (unless `keep_heads=True`), downsamples the spatial map to
    `heatmap_size` x `heatmap_size`, and stores as fp16. Per (layer, step)
    that's a few hundred KB instead of several GB.
    """

    def __init__(self, store: Dict[str, List[torch.Tensor]], name: str,
                 heatmap_size: int = 32, keep_heads: bool = False):
        self.store = store
        self.name = name
        self.heatmap_size = heatmap_size
        self.keep_heads = keep_heads

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Image stream: standard to_q/to_k/to_v.
        q_img = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k_img = attn.head_to_batch_dim(attn.to_k(hidden_states))
        v_img = attn.head_to_batch_dim(attn.to_v(hidden_states))

        # Text/context stream: SD3's real processor uses separate weights
        # for this. Use them if present; fall back only if this checkpoint
        # genuinely shares projections.
        proj_q = getattr(attn, "add_q_proj", attn.to_q)
        proj_k = getattr(attn, "add_k_proj", attn.to_k)
        proj_v = getattr(attn, "add_v_proj", attn.to_v)
        q_txt = attn.head_to_batch_dim(proj_q(encoder_hidden_states))
        k_txt = attn.head_to_batch_dim(proj_k(encoder_hidden_states))
        v_txt = attn.head_to_batch_dim(proj_v(encoder_hidden_states))

        joint_q = torch.cat([q_img, q_txt], dim=1)
        joint_k = torch.cat([k_img, k_txt], dim=1)
        joint_v = torch.cat([v_img, v_txt], dim=1)

        image_length = q_img.shape[1]
        scale = q_img.shape[-1] ** -0.5

        attn_scores = torch.bmm(joint_q, joint_k.transpose(1, 2)) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # ---- THE FIX: slice + collapse + downsample BEFORE storing -------
        image_to_text = attn_probs[:, :image_length, image_length:]  # (heads, N_img, N_text)
        if not self.keep_heads:
            image_to_text = image_to_text.mean(dim=0, keepdim=True)  # (1, N_img, N_text)

        num_text_tokens = image_to_text.shape[-1]
        side = int(round(image_length ** 0.5))
        if side * side == image_length:
            heat = image_to_text.permute(0, 2, 1).reshape(-1, num_text_tokens, side, side)
            heat = F.interpolate(heat, size=(self.heatmap_size, self.heatmap_size),
                                  mode="bilinear", align_corners=False)
        else:
            # Non-square patch grid (shouldn't normally happen for SD3) --
            # store the (still small) raw slice rather than guessing a shape.
            heat = image_to_text
        self.store[self.name].append(heat.detach().to(torch.float16).cpu())
        # --------------------------------------------------------------

        hidden_states = attn.batch_to_head_dim(torch.bmm(attn_probs, joint_v))
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        image_output = hidden_states[:, :image_length]
        text_output = hidden_states[:, image_length:]
        return image_output, text_output


# ============================================================
# 4. SD3 extractor / driver
# ============================================================

class SD3InternalsExtractor:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 device: str = "cuda", dtype: torch.dtype = torch.float16):
        print(f"Loading {model_id} ...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        # enable_model_cpu_offload (not .to(device)) -- these are shared, not
        # dedicated, GPUs: other users' jobs routinely hold 10-14GB, and this
        # extractor's block/attention hooks need headroom on top of the base
        # pipeline. Same OOM root cause and fix already applied to this
        # project's generate_pixArt.py. Offload handles device placement
        # internally, so `device` no longer needs an explicit .to() here --
        # it's still used below for the generator and SAE/tensor placement.
        self.pipe.enable_model_cpu_offload()
        self.transformer = self.pipe.transformer
        self.block_extractor = TransformerActivationExtractor(self.transformer)
        self.attention_maps: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.original_processors = dict(self.transformer.attn_processors)

    def _attach_attention_hooks(self, layer_names: Optional[List[str]] = None,
                                 heatmap_size: int = 32, keep_heads: bool = False) -> None:
        wanted = set(layer_names) if layer_names else None
        processors = {}
        for name, original in self.original_processors.items():
            if wanted is None or name in wanted:
                processors[name] = SD3JointAttentionCapture(
                    self.attention_maps, name, heatmap_size=heatmap_size, keep_heads=keep_heads
                )
            else:
                processors[name] = original
        self.transformer.set_attn_processor(processors)

    def _restore_attention_processors(self) -> None:
        # Restore the exact original processors (e.g. JointAttnProcessor2_0),
        # not a generic default -- mirrors the SD1.5 extractor's fix for the
        # same class of "unrecognized processor type" restore error.
        self.transformer.set_attn_processor(dict(self.original_processors))

    def extract(self, prompt: str, num_inference_steps: int = 28, guidance_scale: float = 7.0,
                capture_blocks: bool = False, capture_attention: bool = True,
                block_names: Optional[List[str]] = None,
                attn_layer_names: Optional[List[str]] = None, heatmap_size: int = 32,
                keep_heads: bool = False, block_step_stride: int = 1,
                generator=None, patcher: Optional["SAEFeaturePatcher"] = None) -> Dict:
        self.block_extractor.clear()
        self.attention_maps.clear()
        self.block_extractor.step_stride = max(1, block_step_stride)
        if block_names is not None:
            self.block_extractor.block_names = block_names

        if capture_blocks:
            self.block_extractor.attach()
        if capture_attention:
            self._attach_attention_hooks(attn_layer_names, heatmap_size, keep_heads)
        if patcher is not None:
            patcher.set_step(0)
            patcher.attach(self.transformer)

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            # callback_on_step_end fires AFTER step `step_index`'s transformer
            # forward (and its block hooks) already ran -- using that forward
            # pass, self.block_extractor._step still held the PREVIOUS step's
            # number (or 0, initially). So this sets the step number for the
            # NEXT forward pass, not this one: without the +1, every step's
            # activations get mislabeled as the step before it (step 0 gets
            # overwritten by step 1's data), and the last step's activations
            # are recorded under a step number no forward pass ever reads --
            # i.e. captured but never used, so every "t=1.0" (final step) row
            # silently disappears from evaluate_concepts. The dangling
            # set_step(step_index + 1) call after the final step is likewise
            # harmless: no further forward pass reads it. patcher shares the
            # exact same step-numbering convention so target_step lines up
            # with the step_index already recorded in concept_metrics.csv.
            self.block_extractor.set_step(step_index + 1)
            if patcher is not None:
                patcher.set_step(step_index + 1)
            return callback_kwargs

        try:
            out = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=step_callback,
            )
            image = out.images[0]
        finally:
            if capture_blocks:
                self.block_extractor.detach()
            if capture_attention:
                self._restore_attention_processors()
            if patcher is not None:
                patcher.detach()

        return {
            "image": image,
            "block_activations": dict(self.block_extractor.activations),
            "attention_maps": dict(self.attention_maps),
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }


# ============================================================
# 5. Post-processing: aggregate across layers/steps, centroid, concentration
# ============================================================

def aggregate_token_heatmap(attention_maps: Dict[str, List[torch.Tensor]], token_idx: int,
                             out_size: int = 1024, layer_names: Optional[List[str]] = None,
                             step_indices: Optional[List[int]] = None) -> torch.Tensor:
    """
    DAAM-style aggregation: averages the (already-downsampled) per-token
    heatmap across the chosen layers/timesteps, then upsamples once to
    `out_size` for a final, high-resolution attribution map.
    """
    layer_names = layer_names or list(attention_maps.keys())
    total, count = None, 0
    for name in layer_names:
        steps_list = attention_maps.get(name, [])
        indices = step_indices if step_indices is not None else range(len(steps_list))
        for t in indices:
            if t >= len(steps_list):
                continue
            heat = steps_list[t][:, token_idx].float()  # (heads_or_1, h, w)
            heat = heat.mean(dim=0)                       # (h, w)
            total = heat if total is None else total + heat
            count += 1
    if count == 0:
        raise ValueError("No attention maps found for the requested layers/steps.")
    heat = (total / count).unsqueeze(0).unsqueeze(0)
    heat = F.interpolate(heat, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return heat.squeeze(0).squeeze(0)


def heatmap_centroid(heatmap: torch.Tensor) -> Tuple[float, float]:
    heatmap = heatmap.clamp(min=0)
    total = heatmap.sum()
    h, w = heatmap.shape
    if total <= 0:
        return (w / 2.0, h / 2.0)
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=heatmap.dtype), torch.arange(w, dtype=heatmap.dtype), indexing="ij"
    )
    cx = (xs * heatmap).sum() / total
    cy = (ys * heatmap).sum() / total
    return (cx.item(), cy.item())


def attention_concentration(heatmap: torch.Tensor) -> float:
    cx, cy = heatmap_centroid(heatmap)
    h, w = heatmap.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=heatmap.dtype), torch.arange(w, dtype=heatmap.dtype), indexing="ij"
    )
    weights = heatmap.clamp(min=0)
    total = weights.sum()
    if total <= 0:
        return 0.0
    variance = (((xs - cx) ** 2 + (ys - cy) ** 2) * weights).sum() / total
    max_variance = (w ** 2 + h ** 2) / 4.0
    return float(max(0.0, 1.0 - variance.item() / max_variance))


def save_heatmap_overlay(image: Image.Image, heatmap: torch.Tensor, path: str,
                          alpha: float = 0.55, cmap: str = "inferno") -> None:
    import matplotlib.cm as cm
    heat = heatmap.detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    colored = (cm.get_cmap(cmap)(heat)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored).resize(image.size)
    Image.blend(image.convert("RGB"), heat_img, alpha=alpha).save(path)


def save_transformer_activations(activations: Dict[str, Dict[int, torch.Tensor]],
                                  out_dir: str = "./sd3_activations") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for block_name, per_step in activations.items():
        steps = sorted(per_step.keys())
        stacked = torch.stack([per_step[s] for s in steps], dim=0)
        safe_name = block_name.replace(".", "_")
        torch.save({"steps": steps, "activations": stacked}, os.path.join(out_dir, f"{safe_name}.pt"))


def select_conditional_batch(tensor: torch.Tensor, do_cfg: bool = True) -> torch.Tensor:
    """
    Block/attention captures have shape (batch, seq, dim). Under classifier-
    free guidance, batch=2 as [unconditional, conditional] (diffusers
    concatenates negative embeddings first). Returns just the conditional
    half as (seq, dim) -- the activations actually driven by the prompt.
    """
    if not do_cfg:
        return tensor[0]
    assert tensor.shape[0] == 2, f"expected batch=2 under CFG, got {tensor.shape[0]}"
    return tensor[1]


# ============================================================
# 6b. Sparse Autoencoder
# ============================================================

class SparseAutoencoder(torch.nn.Module):
    """
    A standard single-layer SAE: Linear -> ReLU encoder, Linear decoder,
    L1 sparsity penalty on the hidden code. Decoder columns are kept
    unit-norm (call `normalize_decoder_()` after every optimizer step) --
    without this, the model can trivially cheat the L1 penalty by shrinking
    decoder weights and inflating hidden activations to compensate.
    """

    def __init__(self, d_in: int, d_hidden: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.encoder = torch.nn.Linear(d_in, d_hidden)
        self.decoder = torch.nn.Linear(d_hidden, d_in, bias=False)
        self.l1_coeff = l1_coeff
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f)

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x: torch.Tensor):
        x_hat, f = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        sparsity = f.abs().mean()
        return recon + self.l1_coeff * sparsity, recon, sparsity

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            w = self.decoder.weight  # (d_in, d_hidden); normalize each column
            norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.data = w / norms


# ============================================================
# 6c. Dataset loading
# ============================================================

def load_concept_dataset(path: str) -> List[Dict[str, str]]:
    """
    Supported formats:
      .txt  -- one prompt per line; concept_id = zero-padded line index
      .json -- list of strings, or list of {"concept": ..., "prompt": ...}
      .csv  -- flexible columns (auto-detects 'prompt'/'content'/'text' for
               the text field and 'concept'/'name_of_work'/'title' for the
               id, falling back to the first two columns), with automatic
               repair of two problems that are common in real-world
               poem/story CSV exports (see `_load_csv_dataset`):
                 1. rows where unescaped commas/newlines in the raw text
                    split ONE entry across several CSV rows
                 2. exact duplicate (title, content) rows
    Returns [{"concept_id": str, "prompt": str}, ...]
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [{"concept_id": f"c{i:04d}", "prompt": p} for i, p in enumerate(lines)]
    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        items = []
        for i, entry in enumerate(data):
            if isinstance(entry, str):
                items.append({"concept_id": f"c{i:04d}", "prompt": entry})
            else:
                items.append({"concept_id": str(entry.get("concept", f"c{i:04d}")),
                              "prompt": entry["prompt"]})
        return items
    if ext == ".csv":
        return _load_csv_dataset(path)
    raise ValueError(f"Unsupported dataset format: {ext} (use .txt, .json, or .csv)")


# A genuine title either starts with an uppercase letter ("Humpty Dumpty",
# no author needed), or -- for the intentionally-lowercase-styled titles
# some poets use (e.g. e.e. cummings, Rupi Kaur) -- ends with a proper
# "by <Capitalized Author Name>" attribution clause.
_BY_AUTHOR_END = re.compile(r"\bby\s+[A-Z][\w.\u2019'-]*(\s+[A-Z][\w.\u2019'-]*)*\s*$")


def _looks_like_real_title(title) -> bool:
    if not isinstance(title, str):
        return False
    t = title.strip()
    if not t:
        return False
    if t[0].isupper():
        return True
    return bool(_BY_AUTHOR_END.search(t))


def _load_csv_dataset(path: str) -> List[Dict[str, str]]:
    """
    Repairs the two failure modes seen in real poem/story CSV exports:

    1. Overflow rows: unescaped commas or embedded newlines in the raw
       text can make a spreadsheet export split ONE entry across several
       CSV rows, spilling fragments of the body text into the title column
       and into extra "Unnamed" overflow columns. Any row that either (a)
       has data in an overflow column, or (b) has a title that doesn't
       look like a real title (see `_looks_like_real_title`), is treated
       as a stray continuation of the *previous* row and merged back into
       it -- every merge is printed so you can audit the decisions rather
       than trust the heuristic blindly. One known false-positive: a title
       ending in an inconsistently-lowercased author surname (e.g. a stray
       "cummings" instead of "Cummings" elsewhere in the same file) can
       get merged incorrectly -- check the merge log if a concept's text
       looks longer than expected.
    2. Exact duplicate (title, content) rows are dropped, keeping the
       first occurrence.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if df.empty:
        return []

    id_col = next((c for c in ("concept", "name_of_work", "title") if c in df.columns), df.columns[0])
    text_col = next((c for c in ("prompt", "content", "text") if c in df.columns), df.columns[1])
    # Only pandas' own auto-generated "Unnamed: N" columns count as overflow
    # signal -- those only appear when a raw row has more comma-separated
    # fields than the header declares (the spillover case this function
    # repairs). A deliberately-named metadata column (e.g. "id", "source")
    # is always populated by design, not a sign of a corrupted row, so
    # treating it as overflow would flag every row as a continuation.
    extra_cols = [c for c in df.columns if c not in (id_col, text_col) and str(c).startswith("Unnamed")]

    records: List[Dict[str, str]] = []
    merge_log: List[Tuple[int, str, str]] = []
    for i, row in df.iterrows():
        title, text = row[id_col], row[text_col]
        has_overflow = any(pd.notna(row[c]) for c in extra_cols)
        is_continuation = has_overflow or not _looks_like_real_title(title)

        if is_continuation and records:
            fragment = " ".join(str(row[c]) for c in [id_col, text_col] + extra_cols if pd.notna(row[c]))
            merge_log.append((i, records[-1]["concept_id"], fragment[:60]))
            records[-1]["prompt"] = (records[-1]["prompt"] + " " + fragment).strip()
            continue
        if pd.isna(title) and pd.isna(text):
            continue  # fully blank row
        records.append({
            "concept_id": str(title) if pd.notna(title) else f"c{len(records):04d}",
            "prompt": str(text) if pd.notna(text) else "",
        })

    if merge_log:
        print(f"[load_concept_dataset] merged {len(merge_log)} corrupted/continuation row(s):")
        for i, target, frag in merge_log:
            print(f"    row {i}  ->  '{target[:40]}'  |  fragment: {frag!r}")

    seen, deduped = set(), []
    for r in records:
        key = (r["concept_id"], r["prompt"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    if len(deduped) < len(records):
        print(f"[load_concept_dataset] dropped {len(records) - len(deduped)} exact duplicate row(s)")

    print(f"[load_concept_dataset] {len(df)} raw rows -> {len(deduped)} clean concept entries")
    return deduped


# ============================================================
# 6d. Collecting activations + training SAEs
# ============================================================

def collect_block_activations(extractor: "SD3InternalsExtractor", dataset: List[Dict[str, str]],
                               block_names: List[str], cache_dir: str, seeds=(42,),
                               num_inference_steps: int = 28, guidance_scale: float = 7.0,
                               timestep_stride: int = 4, max_tokens_per_block: int = 200_000,
                               device: str = "cuda") -> None:
    """
    Runs generation for every (prompt, seed), capturing image-token
    activations at `block_names` every `timestep_stride`-th denoising step.
    Stops once every block has reached `max_tokens_per_block`.

    Rather than pooling every generation's activations in memory (which
    grows without bound as more prompts are processed -- confirmed via
    dmesg, on a large dataset this is what actually OOM-kills the process,
    independent of and in addition to the base model's own memory use),
    each generation's per-block tensor is written straight to its own small
    file under `cache_dir` and dropped from memory immediately.
    `train_sae_on_blocks` reads a block's files back in only when it's that
    block's turn to train, so peak RAM only ever holds one block's data at
    a time, not all of them.

    NOTE: pass in a shuffled `dataset` (see `prepare_dataset`) for large
    collections -- otherwise, since this stops as soon as the token budget
    is full, the training set ends up drawn entirely from whatever order
    the dataset happens to be in rather than a representative sample.
    """
    counts: Dict[str, int] = defaultdict(int)
    do_cfg = guidance_scale > 1.0
    for b in block_names:
        os.makedirs(os.path.join(cache_dir, b.replace(".", "_")), exist_ok=True)

    total_pairs = len(dataset) * len(seeds)
    tracker = ProgressTracker(total_pairs, label="collection pairs",
                               every=max(1, total_pairs // 200 or 1))

    file_idx = 0
    for item in dataset:
        if all(counts[b] >= max_tokens_per_block for b in block_names):
            print("  [collect_block_activations] token budget reached for all blocks, stopping early.")
            break
        for seed in seeds:
            if all(counts[b] >= max_tokens_per_block for b in block_names):
                break
            generator = torch.Generator(device=device).manual_seed(seed)
            result = extractor.extract(
                item["prompt"], num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                capture_blocks=True, capture_attention=False, block_names=block_names,
                block_step_stride=timestep_stride, generator=generator,
            )
            for b in block_names:
                if counts[b] >= max_tokens_per_block:
                    continue
                for act in result["block_activations"].get(b, {}).values():
                    cond = select_conditional_batch(act, do_cfg=do_cfg)  # (N_img, d_in)
                    block_dir = os.path.join(cache_dir, b.replace(".", "_"))
                    torch.save(cond, os.path.join(block_dir, f"{file_idx:06d}.pt"))
                    counts[b] += cond.shape[0]
            file_idx += 1
            tracker.step()


def load_block_activations(cache_dir: str, block: str) -> Optional[torch.Tensor]:
    block_dir = os.path.join(cache_dir, block.replace(".", "_"))
    if not os.path.isdir(block_dir):
        return None
    files = sorted(os.listdir(block_dir))
    if not files:
        return None
    tensors = [torch.load(os.path.join(block_dir, f)) for f in files]
    return torch.cat(tensors, dim=0)


def train_sae(activations: torch.Tensor, d_hidden: int = 4096, l1_coeff: float = 1e-3,
              epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3,
              device: str = "cuda") -> SparseAutoencoder:
    d_in = activations.shape[-1]
    sae = SparseAutoencoder(d_in, d_hidden, l1_coeff=l1_coeff).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    activations = activations.to(torch.float32)
    n = activations.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        tot_loss = tot_recon = tot_sparsity = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = activations[idx].to(device)
            loss, recon, sparsity = sae.loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder_()
            tot_loss += loss.item() * len(idx)
            tot_recon += recon.item() * len(idx)
            tot_sparsity += sparsity.item() * len(idx)
        print(f"    epoch {epoch + 1}/{epochs}  loss={tot_loss / n:.4f}  "
              f"recon={tot_recon / n:.4f}  sparsity={tot_sparsity / n:.4f}")
    return sae


def train_sae_on_blocks(extractor: "SD3InternalsExtractor", dataset: List[Dict[str, str]],
                         block_names: List[str], seeds=(42,), num_inference_steps: int = 28,
                         guidance_scale: float = 7.0, timestep_stride: int = 4,
                         max_tokens_per_block: int = 200_000, d_hidden: int = 4096,
                         l1_coeff: float = 1e-3, sae_epochs: int = 10, batch_size: int = 1024,
                         lr: float = 1e-3, device: str = "cuda",
                         save_dir: str = "./sae_checkpoints",
                         cache_dir: Optional[str] = None, keep_cache: bool = False,
                         ) -> Dict[str, SparseAutoencoder]:
    cache_dir = cache_dir or os.path.join(save_dir, "_activation_cache")
    print(f"Collecting activations from {block_names} across {len(dataset)} prompt(s) "
          f"x {len(seeds)} seed(s) (every {timestep_stride} step(s))...")
    collect_block_activations(
        extractor, dataset, block_names, cache_dir, seeds=seeds, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, timestep_stride=timestep_stride,
        max_tokens_per_block=max_tokens_per_block, device=device,
    )
    os.makedirs(save_dir, exist_ok=True)
    saes: Dict[str, SparseAutoencoder] = {}
    for block in block_names:
        acts = load_block_activations(cache_dir, block)
        if acts is None or acts.shape[0] == 0:
            print(f"  [skip] no activations collected for {block}")
            continue
        print(f"\n  Training SAE for {block}  ({acts.shape[0]} tokens, d_in={acts.shape[1]})")
        sae = train_sae(acts, d_hidden=d_hidden, l1_coeff=l1_coeff, epochs=sae_epochs,
                         batch_size=batch_size, lr=lr, device=device)
        saes[block] = sae
        # Free this block's activations before moving to the next one,
        # rather than letting every block's tokens pile up simultaneously.
        del acts
        gc.collect()
    if not keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
        torch.save(sae.state_dict(), os.path.join(save_dir, f"sae_{block.replace('.', '_')}.pt"))
    return saes


def probe_block_dim(extractor: "SD3InternalsExtractor", block_name: str, device: str = "cuda") -> int:
    """Minimal 1-step generation solely to read a block's hidden size (for loading a saved SAE)."""
    extractor.block_extractor.block_names = [block_name]
    result = extractor.extract("probe", num_inference_steps=1, guidance_scale=1.0,
                                capture_blocks=True, capture_attention=False, block_names=[block_name])
    act = next(iter(result["block_activations"].get(block_name, {}).values()), None)
    if act is None:
        raise RuntimeError(f"Could not probe hidden size for {block_name}.")
    return act.shape[-1]


# ============================================================
# 6e. Concept latent -> spatial heatmap, and concept evaluation
# ============================================================

def sae_concept_heatmap(sae: SparseAutoencoder, block_activation_img: torch.Tensor,
                         out_size: int = 64) -> Tuple[torch.Tensor, int]:
    """
    block_activation_img: (N_img, d_in) image-token activations for one
    (prompt, seed, timestep). Encodes with the SAE, finds the latent with
    the highest mean activation across image tokens -- a proxy for 'the
    feature this block is using to represent this concept' -- and reshapes
    that latent's per-token activation into a spatial heatmap, exactly like
    a cross-attention map.
    """
    device = next(sae.parameters()).device
    with torch.no_grad():
        codes = sae.encode(block_activation_img.to(device).float())  # (N_img, d_hidden)
    latent_idx = int(codes.mean(dim=0).argmax().item())
    per_token = codes[:, latent_idx]

    n_img = per_token.shape[0]
    side = int(round(n_img ** 0.5))
    if side * side != n_img:
        raise ValueError(f"Image-token count {n_img} isn't a perfect square; can't form a spatial grid.")
    heat = per_token.reshape(1, 1, side, side)
    heat = F.interpolate(heat, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return heat.squeeze(0).squeeze(0).cpu(), latent_idx


def _fmt_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class ProgressTracker:
    """Elapsed-time / ETA tracker for long-running dataset x seed loops --
    matters once you're running thousands of concepts, not just a handful."""

    def __init__(self, total: int, label: str = "items", every: int = 1):
        self.total = total
        self.done = 0
        self.label = label
        self.every = max(1, every)
        self.start = time.time()

    def step(self, n: int = 1, extra: str = "") -> None:
        self.done += n
        if self.done % self.every != 0 and self.done != self.total:
            return
        elapsed = time.time() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        pct = 100 * self.done / self.total if self.total else 100
        msg = (f"  [{self.done}/{self.total} {self.label}  {pct:5.1f}%]  "
               f"elapsed={_fmt_duration(elapsed)}  eta={_fmt_duration(remaining)}")
        print(msg + (f"  {extra}" if extra else ""))


def prepare_dataset(dataset: List[Dict[str, str]], limit: Optional[int] = None,
                     shuffle_seed: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Applies an optional deterministic shuffle -- important at dataset sizes
    like 4200 rows, since without it, SAE-training's `max_tokens_per_block`
    budget gets spent entirely on the first alphabetical/source-ordered
    entries rather than a representative sample -- and an optional cap, for
    a quick smoke test on a slice before committing to the full run.
    """
    dataset = list(dataset)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(dataset)
    if limit is not None:
        dataset = dataset[:limit]
    return dataset


def _should_save_heatmap(concept_id: str, seed: int, rate: float) -> bool:
    """
    Deterministic (stable across runs and --resume) sampling of which
    (concept, seed) pairs get their image + heatmap overlays saved to disk.
    At 4200 concepts x 4 seeds x 5 blocks x 3 timesteps, saving everything
    (rate=1.0) means up to ~250K PNGs -- fine for a 50-row test, not fine
    for the full run. Use e.g. rate=0.02 to keep ~2% as visual spot-checks
    while still getting every row's numeric centroid/concentration.
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.md5(f"{concept_id}::{seed}".encode()).hexdigest()
    return (int(digest[:8], 16) % 1_000_000) / 1_000_000 < rate


class IncrementalCSVWriter:
    """
    Appends rows to a CSV as they're produced (with a flush after every
    row) instead of holding everything in memory and writing once at the
    end -- so a crash partway through a multi-hour/day run of thousands of
    concepts never loses completed work. Combine with `--resume` to skip
    already-written (concept_id, seed) pairs on a restart.
    """

    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None

    def open(self, resume: bool) -> None:
        file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        mode = "a" if (resume and file_exists) else "w"
        self._file = open(self.path, mode, newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if mode == "w":
            self._writer.writeheader()

    def write(self, row: Dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


def load_existing_results(path: str, block_names: Optional[List[str]] = None,
                           timestep_fractions: Optional[Tuple[float, ...]] = None
                           ) -> Tuple[List[Dict], set]:
    """
    For --resume: reads a previously-written concept_metrics.csv and
    returns (rows, done_pairs).

    If `block_names`/`timestep_fractions` are given (the current run's
    settings), a (concept, seed) pair is only marked done if the file
    already has rows for EVERY (block, timestep) combination currently
    requested -- not just if the pair appears at all. This matters if you
    reuse the same --out-dir after changing --sae-blocks or --timesteps
    (or after an earlier --limit smoke test): without this check, a pair
    with only partial/stale coverage from a differently-configured prior
    run would be wrongly treated as fully done and silently skipped,
    which is the most common cause of a --resume run writing far fewer
    rows than expected. If either argument is omitted, falls back to the
    coarser "present at all" check.
    """
    if not os.path.exists(path):
        return [], set()
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            row["step_index"] = int(row["step_index"])
            row["timestep_fraction"] = float(row["timestep_fraction"])
            row["latent_idx"] = int(row["latent_idx"])
            row["centroid_x"] = float(row["centroid_x"])
            row["centroid_y"] = float(row["centroid_y"])
            row["concentration"] = float(row["concentration"])
            rows.append(row)

    if block_names is None or timestep_fractions is None:
        return rows, {(r["concept_id"], r["seed"]) for r in rows}

    required = {(b, round(f, 6)) for b in block_names for f in timestep_fractions}
    coverage: Dict[Tuple[str, int], set] = defaultdict(set)
    for r in rows:
        coverage[(r["concept_id"], r["seed"])].add((r["block"], round(r["timestep_fraction"], 6)))
    done_pairs = {pair for pair, have in coverage.items() if required.issubset(have)}
    return rows, done_pairs



def _slug(s: str, max_len: int = 40) -> str:
    """Filesystem-safe short slug for concept ids / titles used in filenames."""
    s = re.sub(r"[^\w\-]+", "_", str(s)).strip("_")
    return s[:max_len] or "item"


def _latent_filename(concept_id: str, seed: int) -> str:
    # slug + short hash: the 40-char slug alone can collide across long,
    # similarly-prefixed poem titles; the hash keeps every concept's file distinct.
    digest = hashlib.md5(str(concept_id).encode()).hexdigest()[:8]
    return f"{_slug(concept_id)}_{digest}_seed{seed}.npz"


def summarize_latents(sae: SparseAutoencoder, block_activation_img: torch.Tensor,
                       top_k: int = 64, map_size: int = 32) -> Dict[str, np.ndarray]:
    """
    Everything the Concept Fidelity Score needs from one (prompt, block,
    timestep), small enough to keep for every prompt: the mean activation
    of EVERY latent over image tokens (for contrastive concept->latent
    mapping and presence), plus the spatial maps of the `top_k` latents by
    mean activation, downsampled to `map_size` (for coherence/binding).
    ~130 KB per (block, timestep) in fp16 instead of the ~32 MB a full
    per-token code would take.
    """
    device = next(sae.parameters()).device
    with torch.no_grad():
        codes = sae.encode(block_activation_img.to(device).float())   # (N_img, d_hidden)
        mean_vec = codes.mean(dim=0)                                    # (d_hidden,)
        n_img = codes.shape[0]
        side = int(round(n_img ** 0.5))
        k = min(top_k, codes.shape[1])
        top_idx = mean_vec.topk(k).indices
        out = {
            "mean": mean_vec.to(torch.float16).cpu().numpy(),
            "top_idx": top_idx.to(torch.int32).cpu().numpy(),
            "l0": (codes > 0).float().sum(dim=1).mean().item(),        # avg active latents per token
        }
        if side * side == n_img:
            maps = codes[:, top_idx].T.reshape(1, k, side, side)
            maps = F.interpolate(maps, size=(map_size, map_size), mode="bilinear", align_corners=False)[0]
            out["maps"] = maps.to(torch.float16).cpu().numpy()          # (k, map_size, map_size)
    return out


def evaluate_concepts(extractor: "SD3InternalsExtractor", dataset: List[Dict[str, str]],
                       saes: Dict[str, SparseAutoencoder], seeds=(42, 45, 12, 1000),
                       timestep_fractions=(0.0, 0.5, 1.0), num_inference_steps: int = 28,
                       guidance_scale: float = 7.0, heatmap_out_size: int = 64,
                       out_dir: str = "./sae", device: str = "cuda",
                       save_heatmap_images: bool = True, heatmap_sample_rate: float = 1.0,
                       resume: bool = False, latents_dir: Optional[str] = None,
                       latent_top_k: int = 64, latent_map_size: int = 32) -> List[Dict]:
    """
    For every concept x seed x requested timestep fraction: generates,
    encodes the chosen block(s)' image-token activations with their SAE,
    finds the concept latent, builds its spatial heatmap, and records the
    heatmap's centroid (x, y) and concentration score. This is the raw
    per-row table that Centroid Alignment (deviation across seeds) and
    Cross-Concept Separation (distance across concepts) are both computed
    from downstream -- see `summarize_centroid_alignment`.

    Built for large datasets (thousands of concepts):
      - rows are appended to concept_metrics.csv incrementally (flushed
        after every row), so a crash partway through never loses completed
        work -- rerun with `resume=True` to pick up where it left off.
      - `heatmap_sample_rate` controls what fraction of (concept, seed)
        pairs get their image + heatmap overlays saved to disk (see
        `_should_save_heatmap`); the numeric CSV/JSON always contains every
        row regardless of this setting.
      - progress is reported with elapsed time and an ETA.

    If `save_heatmap_images` is True, for sampled (concept, seed) pairs writes:
      out_dir/images/<concept>_seed<seed>.png                       -- the generated image
      out_dir/heatmaps/<concept>_seed<seed>_<block>_t<fraction>.png -- heatmap overlay
    """
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    heatmaps_dir = os.path.join(out_dir, "heatmaps")
    if save_heatmap_images:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(heatmaps_dir, exist_ok=True)
    if latents_dir is not None:
        os.makedirs(latents_dir, exist_ok=True)

    block_names = list(saes.keys())
    do_cfg = guidance_scale > 1.0
    step_indices = sorted({int(round(f * (num_inference_steps - 1))) for f in timestep_fractions})
    frac_by_step = {int(round(f * (num_inference_steps - 1))): f for f in timestep_fractions}
    extractor.block_extractor.block_names = block_names

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

                # CPU generator (not `device`) -- matches imageMetric/pipeline_common.py's
                # convention for the images this project's CLIP/BLIP2/VQAScore benchmark
                # was scored on. A CUDA generator with the same seed produces a DIFFERENT
                # noise sequence (and thus a different image) than a CPU one, so this is
                # what makes seed=42 here reproduce the literal already-scored images
                # rather than a same-prompt-different-image approximation.
                generator = torch.Generator(device="cpu").manual_seed(seed)
                result = extractor.extract(
                    prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                    capture_blocks=True, capture_attention=False, block_names=block_names,
                    block_step_stride=1, generator=generator,  # need exact steps here
                )
                image = result["image"]
                if save_media:
                    image.save(os.path.join(images_dir, f"{concept_slug}_seed{seed}.png"))

                pair_rows: List[Dict] = []
                latent_store: Dict[str, np.ndarray] = {}
                for block in block_names:
                    block_slug = _slug(block)
                    per_step = result["block_activations"].get(block, {})
                    for step in step_indices:
                        act = per_step.get(step)
                        if act is None:
                            continue
                        cond = select_conditional_batch(act, do_cfg=do_cfg)
                        try:
                            heat, latent_idx = sae_concept_heatmap(saes[block], cond, out_size=heatmap_out_size)
                        except ValueError:
                            continue
                        cx, cy = heatmap_centroid(heat)
                        conc = attention_concentration(heat)

                        row = {
                            "concept_id": concept_id, "prompt": prompt, "block": block, "seed": seed,
                            "step_index": step, "timestep_fraction": frac_by_step[step],
                            "latent_idx": latent_idx, "centroid_x": cx, "centroid_y": cy,
                            "concentration": conc, "heatmap_overlay": "",
                        }
                        if save_media:
                            full_heat = F.interpolate(
                                heat.unsqueeze(0).unsqueeze(0), size=image.size[::-1],
                                mode="bilinear", align_corners=False,
                            ).squeeze(0).squeeze(0)
                            overlay_path = os.path.join(
                                heatmaps_dir,
                                f"{concept_slug}_seed{seed}_{block_slug}_t{frac_by_step[step]:.2f}.png",
                            )
                            save_heatmap_overlay(image, full_heat, overlay_path)
                            row["heatmap_overlay"] = overlay_path
                        pair_rows.append(row)

                        if latents_dir is not None:
                            summary = summarize_latents(saes[block], cond, top_k=latent_top_k,
                                                        map_size=latent_map_size)
                            prefix = f"{block_slug}|t{frac_by_step[step]:.2f}|"
                            for key, val in summary.items():
                                latent_store[prefix + key] = val

                # The latent file is written BEFORE this pair's rows, so a crash in
                # between leaves an orphan .npz (harmless, overwritten on resume)
                # rather than rows that --resume would treat as done with no
                # latent file behind them.
                if latents_dir is not None and latent_store:
                    latent_file = _latent_filename(concept_id, seed)
                    np.savez_compressed(os.path.join(latents_dir, latent_file), **latent_store)
                    with open(os.path.join(latents_dir, "latents_index.csv"), "a", newline="") as f:
                        csv.writer(f).writerow([concept_id, seed, latent_file])
                for row in pair_rows:
                    writer.write(row)
                    rows.append(row)

                tracker.step(extra=f"last: {concept_id[:30]!r} seed={seed}")
    finally:
        writer.close()

    with open(os.path.join(out_dir, "concept_metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nDone. {len(rows)} total row(s) in {csv_path} "
          f"({skipped} pair(s) skipped via --resume).")
    if save_heatmap_images:
        print(f"Saved images -> {images_dir}/  and heatmap overlays -> {heatmaps_dir}/ "
              f"(sampled at rate={heatmap_sample_rate})")
    return rows


def summarize_centroid_alignment(rows: List[Dict]) -> List[Dict]:
    """
    Groups rows by (concept_id, block, timestep_fraction) and measures how
    much the concept's centroid moves across seeds -- operationalizing
    Centroid Alignment: low deviation means the model represents this
    concept in the same place regardless of random seed (i.e. it's
    representing the concept, not guessing); high deviation means the
    opposite.
    """
    groups: Dict[Tuple[str, str, float], List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        key = (r["concept_id"], r["block"], r["timestep_fraction"])
        groups[key].append((r["centroid_x"], r["centroid_y"]))

    summary = []
    for (concept_id, block, frac), pts in groups.items():
        pts_t = torch.tensor(pts)
        mean_pt = pts_t.mean(dim=0)
        deviation = (pts_t - mean_pt).norm(dim=1).mean().item()
        summary.append({
            "concept_id": concept_id, "block": block, "timestep_fraction": frac,
            "num_seeds": len(pts), "mean_centroid": mean_pt.tolist(),
            "centroid_deviation": deviation,
        })
    return summary


# ============================================================
# 6. get prompt token spans (for mapping token_idx -> word)
# ============================================================

def get_prompt_token_spans(pipe, prompt: str) -> List[Tuple[int, str]]:
    """CLIP-tokenizer content tokens (excludes BOS/EOS/pad) with their index."""
    tokenizer = pipe.tokenizer  # CLIP-L tokenizer; T5 stream is handled separately by the pipe
    ids = tokenizer(prompt, padding="max_length", truncation=True,
                     max_length=tokenizer.model_max_length, return_tensors="pt").input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    return [(i, tok.replace("</w>", "")) for i, (tid, tok) in enumerate(zip(ids.tolist(), tokens))
            if tid not in special]


# ============================================================
# 7. End-to-end: generate + interpret
# ============================================================

def interpret_prompt(extractor: SD3InternalsExtractor, prompt: str,
                      num_inference_steps: int = 28, guidance_scale: float = 7.0,
                      out_dir: str = "./sd3_interpretation",
                      layer_names: Optional[List[str]] = None,
                      heatmap_size: int = 32, out_image_size: int = 1024,
                      tokens_to_interpret: Optional[List[str]] = None,
                      capture_blocks: bool = False, generator=None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    result = extractor.extract(
        prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        capture_blocks=capture_blocks, capture_attention=True,
        attn_layer_names=layer_names, heatmap_size=heatmap_size, generator=generator,
    )
    image = result["image"]
    image.save(os.path.join(out_dir, "generated_image.png"))

    token_spans = get_prompt_token_spans(extractor.pipe, prompt)
    if tokens_to_interpret:
        wanted = {w.lower() for w in tokens_to_interpret}
        token_spans = [(i, t) for i, t in token_spans if t.lower() in wanted]

    report = {"prompt": prompt, "num_inference_steps": num_inference_steps, "tokens": {}}
    print(f"\nInterpreting {len(token_spans)} token(s) across "
          f"{len(result['attention_maps'])} captured layers...\n")

    for idx, word in token_spans:
        try:
            heat = aggregate_token_heatmap(
                result["attention_maps"], token_idx=idx, out_size=out_image_size, layer_names=layer_names
            )
        except (ValueError, IndexError):
            continue  # token index out of range for this layer's text-token count (e.g. T5 padding)
        cx, cy = heatmap_centroid(heat)
        conc = attention_concentration(heat)
        overlay_path = os.path.join(out_dir, f"attn_{idx:02d}_{word}.png")
        save_heatmap_overlay(image, heat, overlay_path)
        report["tokens"][word] = {"token_index": idx, "centroid": [cx, cy],
                                   "concentration": conc, "heatmap_overlay": overlay_path}
        print(f"  '{word:>15s}'  centroid=({cx:7.1f},{cy:7.1f})  concentration={conc:.3f}")

    if capture_blocks:
        save_transformer_activations(result["block_activations"], os.path.join(out_dir, "block_activations"))

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved image + {len(report['tokens'])} overlays + report.json -> {out_dir}")
    return report


# ============================================================
# 8. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SD3 interpretability: single-prompt attention maps (--mode interpret), "
                     "or SAE-based concept centroid/concentration analysis across a dataset, "
                     "seeds, and timesteps (--mode sae)."
    )
    parser.add_argument("--mode", choices=["interpret", "sae"], default="interpret")
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default=None)

    # --- interpret mode ---
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--tokens", type=str, default=None,
                         help="Comma-separated words to interpret (default: all content tokens).")
    parser.add_argument("--heatmap-size", type=int, default=32)
    parser.add_argument("--layers", type=str, default=None,
                         help="Comma-separated transformer-block indices to capture attention from.")
    parser.add_argument("--save-block-activations", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # --- sae mode ---
    parser.add_argument("--dataset", type=str, default=None,
                         help="Path to a .txt/.json/.csv concept/prompt dataset (required for --mode sae).")
    parser.add_argument("--sae-blocks", type=str, default="4,8,12,16,20",
                         help="Comma-separated transformer-block indices to train/evaluate SAEs on.")
    parser.add_argument("--seeds", type=str, default="42,45,12,1000")
    parser.add_argument("--timesteps", type=str, default="0.0,0.5,1.0",
                         help="Fractions of the denoising trajectory to evaluate at "
                              "(0=first step, 1=last step).")
    parser.add_argument("--sae-hidden", type=int, default=4096)
    parser.add_argument("--sae-l1", type=float, default=1e-3)
    parser.add_argument("--sae-epochs", type=int, default=10)
    parser.add_argument("--sae-batch-size", type=int, default=1024)
    parser.add_argument("--sae-lr", type=float, default=1e-3)
    parser.add_argument("--train-timestep-stride", type=int, default=4,
                         help="Sample every Nth denoising step when collecting SAE training data.")
    parser.add_argument("--max-tokens-per-block", type=int, default=200_000)
    parser.add_argument("--sae-dir", type=str, default="./sae_checkpoints")
    parser.add_argument("--load-sae", action="store_true",
                         help="Load existing SAE checkpoints from --sae-dir instead of retraining.")
    parser.add_argument("--heatmap-out-size", type=int, default=64)
    parser.add_argument("--no-heatmap-images", action="store_true",
                         help="Skip saving generated images + heatmap overlay PNGs entirely "
                              "(only write the numeric centroid/concentration CSV/JSON).")
    parser.add_argument("--heatmap-sample-rate", type=float, default=1.0,
                         help="Fraction of (concept, seed) pairs to save images/heatmaps for "
                              "(0.0-1.0). At thousands of concepts, 1.0 means hundreds of "
                              "thousands of PNGs -- use e.g. 0.02-0.05 for spot-checks. The "
                              "numeric CSV/JSON always has every row regardless of this setting.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap the dataset to the first N entries after shuffling -- use "
                              "for a smoke test before committing to a full multi-thousand-row run.")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                         help="Deterministic shuffle seed applied to the dataset before both SAE "
                              "training and evaluation (matters for large datasets, since SAE "
                              "training stops once its token budget is full -- without a shuffle "
                              "it would only ever see the first alphabetical/source-ordered "
                              "entries). Pass -1 to keep the dataset's original order.")
    parser.add_argument("--resume", action="store_true",
                         help="Resume evaluation from an existing out_dir/concept_metrics.csv, "
                              "skipping (concept, seed) pairs already completed. Essential for "
                              "multi-thousand-row runs that might get interrupted.")
    parser.add_argument("--save-latents", action="store_true",
                         help="Also write out_dir/latents/<concept>_seed<seed>.npz per (concept, seed): "
                              "the mean activation of every SAE latent plus the top-k latents' spatial "
                              "maps, per (block, timestep) -- the raw material for the Concept Fidelity "
                              "Score's contrastive mapping, presence, coherence and binding.")
    parser.add_argument("--latent-top-k", type=int, default=64)
    parser.add_argument("--latent-map-size", type=int, default=32)
    args = parser.parse_args()

    extractor = SD3InternalsExtractor(args.model_id, device=args.device)

    if args.mode == "interpret":
        prompt = args.prompt or input("Enter a prompt: ").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        tokens_to_interpret = [t.strip() for t in args.tokens.split(",")] if args.tokens else None
        layer_names = None
        if args.layers:
            layer_names = [f"transformer_blocks.{i.strip()}.attn.processor" for i in args.layers.split(",")]
        generator = (torch.Generator(device=args.device).manual_seed(args.seed)
                     if args.seed is not None else None)
        interpret_prompt(
            extractor, prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
            out_dir=args.out_dir or "./sd3_interpretation", layer_names=layer_names,
            heatmap_size=args.heatmap_size, tokens_to_interpret=tokens_to_interpret,
            capture_blocks=args.save_block_activations, generator=generator,
        )
        return

    # --- mode == "sae" ---
    if not args.dataset:
        raise ValueError("--dataset is required for --mode sae")
    raw_dataset = load_concept_dataset(args.dataset)
    shuffle_seed = None if args.shuffle_seed < 0 else args.shuffle_seed
    dataset = prepare_dataset(raw_dataset, limit=args.limit, shuffle_seed=shuffle_seed)
    print(f"[main] using {len(dataset)}/{len(raw_dataset)} dataset entries "
          f"(shuffle_seed={shuffle_seed}, limit={args.limit})")

    block_names = [f"transformer_blocks.{i.strip()}" for i in args.sae_blocks.split(",")]
    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    timestep_fractions = tuple(float(t.strip()) for t in args.timesteps.split(","))
    out_dir = args.out_dir or "./sae"

    if args.load_sae:
        print(f"Loading SAE checkpoints from {args.sae_dir} ...")
        saes = {}
        for block in block_names:
            ckpt_path = os.path.join(args.sae_dir, f"sae_{block.replace('.', '_')}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [skip] no checkpoint found for {block} at {ckpt_path}")
                continue
            d_in = probe_block_dim(extractor, block, device=args.device)
            sae = SparseAutoencoder(d_in, args.sae_hidden, l1_coeff=args.sae_l1).to(args.device)
            sae.load_state_dict(torch.load(ckpt_path, map_location=args.device))
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


if __name__ == "__main__":
    main()