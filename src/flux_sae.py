# ================================================================
# FLUX.1-DEV — SAE-BASED METAPHOR ANALYSIS
# ================================================================
#
# Adapts flux.py's concrete/metaphor attention analysis to the
# SAE-based mechanistic-interpretability methodology of:
#
#   Tinaz, Fabian, Soltanolkotabi. "Emergence and Evolution of
#   Interpretable Concepts in Diffusion Models." NeurIPS 2025.
#   arXiv:2504.15473
#
# ----------------------------------------------------------------
# WHAT CHANGED RELATIVE TO flux.py, AND WHY
# ----------------------------------------------------------------
#
# The paper's object of study is not the raw cross-attention map,
# but a k-sparse (TopK) autoencoder trained per (architectural
# block, diffusion timestep) on the residual update the block
# writes to the image stream (Sec 3.1-3.2: Δ_{l,t} = the
# difference between a block's output and input). Interpretable
# "concepts" are columns of the SAE's decoder (W_dec), and a
# concept's spatial footprint is how strongly its latent fires
# across image tokens.
#
# This script keeps flux.py's 10-sample concrete/metaphor dataset
# and its metric functions (calculate_centroid, attention_entropy,
# attention_concentration_index, centroid_deviation,
# cross_concept_index) completely unchanged, and only swaps out
# WHAT feeds them:
#
#   flux.py:      concept spatial map = averaged raw cross-attention
#                 from image tokens to that concept's T5 tokens.
#
#   flux_sae.py:  concept spatial map = the activation of a single
#                 SAE latent (CID), chosen per (block, timestep
#                 bucket) as the latent whose decoder direction is
#                 most aligned with the image-token activations the
#                 concept's text tokens attend to.
#
# Per the paper, this is done SEPARATELY for each (block, timestep-
# bucket) combination -- concepts at different blocks/times are not
# directly comparable, since a different SAE is trained for each
# combination (Sec 3.2, Fig 9-11 captions).
#
# ----------------------------------------------------------------
# SCOPE (deliberately narrower than the full paper)
# ----------------------------------------------------------------
#
#  1. FLUX has no U-Net down/mid/up blocks. Its transformer has two
#     kinds of blocks: double-stream (image and text attend to each
#     other via separate QKV projections, joined for attention) and
#     single-stream (image and text tokens live in one concatenated
#     sequence, later in the network). We pick an early/mid/late
#     block from EACH stream as our block set -- SAE_BLOCK_CONFIG
#     below -- as the closest structural analog to the paper's
#     down_block / mid_block / up_block choice.
#
#  2. The paper's concept dictionary (Sec 3.3) is built from ~40k
#     generations, RAM (image tagging), GroundingDINO (open-set
#     detection) and SAM (segmentation) -- entirely vision-driven
#     and decoupled from any specific prompt. That pipeline is out
#     of scope here. Instead, concepts here are the same ones
#     flux.py already used (defined by concrete/abstract text spans
#     in each of the 10 prompts), and CID assignment is done via
#     the existing text-token attention as a lightweight seed,
#     documented in `assign_concept_to_cid` below.
#
#  3. No causal interventions (Sec 3.5). This script is read-only /
#     analysis-only, matching flux.py's own scope.
#
#  4. Given only 10 samples, each SAE is intentionally small (see
#     SAE_EXPANSION_FACTOR / SAE_TOPK below) -- this is a research
#     prototype, not a claim that these SAEs are as well-converged
#     as the paper's (trained on 200k LAION-COCO prompts).
#
# ================================================================


# ================================================================
# 1. IMPORTS
# ================================================================

import gc
import json
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse everything from flux.py that isn't specific to raw-attention
# extraction: the dataset, config, tokenization, and metric functions
# are unchanged by the switch to SAEs.
from flux import (
    MODEL_ID,
    RESULTS_DIR,
    IMAGE_DIR,
    METRIC_DIR,
    SEED,
    HEIGHT,
    WIDTH,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    MAX_SEQUENCE_LENGTH,
    DTYPE,
    DATASET,
    concepts_dict_for_sample,
    get_token_indices,
    normalize_attention,
    calculate_centroid,
    centroid_deviation,
    cross_concept_index,
    attention_entropy,
    attention_concentration_index,
    token_attention_to_spatial,
    compute_metrics,
    slugify,
    RecordingFluxAttnProcessor,
)

from diffusers import FluxPipeline


# ================================================================
# 2. SAE-SPECIFIC OUTPUT DIRECTORIES
# ================================================================

SAE_RESULTS_DIR = os.path.join(RESULTS_DIR, "sae")
SAE_WEIGHTS_DIR = os.path.join(SAE_RESULTS_DIR, "weights")
SAE_MAP_DIR = os.path.join(SAE_RESULTS_DIR, "concept_maps")
SAE_METRIC_DIR = os.path.join(SAE_RESULTS_DIR, "metrics")

for directory in [SAE_RESULTS_DIR, SAE_WEIGHTS_DIR, SAE_MAP_DIR, SAE_METRIC_DIR]:
    os.makedirs(directory, exist_ok=True)


# ================================================================
# 3. SAE HYPERPARAMETERS (Sec 3.1)
# ================================================================
#
# The paper uses n_features >> d ("expansive" encoding, motivated by
# the superposition hypothesis) and TopK activation (their k=10/20,
# see Table 1). We keep the same TopK-activation architecture, but
# scale n_features off each block's actual hidden size rather than
# hardcoding it, and use a much smaller expansion + smaller training
# set, appropriate for 10 prompts instead of 200k.

SAE_EXPANSION_FACTOR = 4     # n_features = SAE_EXPANSION_FACTOR * d
SAE_TOPK = 32                # k in TopK(ReLU(...))
SAE_EPOCHS = 300
SAE_BATCH_SIZE = 512
SAE_LR = 1e-3


# ================================================================
# 4. TIMESTEP BUCKETS (Sec 3.2: t in {0.0, 0.5, 1.0})
# ================================================================
#
# The paper samples DDIM steps at t=1.0 (near-pure noise), t=0.5
# (middle of the trajectory), t=0.0 (final, near-clean image), and
# trains one SAE per timestep. FLUX's flow-matching schedule has
# NUM_INFERENCE_STEPS steps running the same direction (noise -> 0),
# so we bucket by step index instead of DDIM index.

TIMESTEP_BUCKET_STEPS = {
    "t1.0": 0,
    "t0.5": NUM_INFERENCE_STEPS // 2,
    "t0.0": NUM_INFERENCE_STEPS - 1,
}

TARGET_STEPS = set(TIMESTEP_BUCKET_STEPS.values())
STEP_TO_BUCKET = {v: k for k, v in TIMESTEP_BUCKET_STEPS.items()}


# ================================================================
# 5. BLOCK SELECTION (structural analog of down/mid/up_block)
# ================================================================
#
# Populated at runtime from the loaded pipeline, since block counts
# vary by FLUX checkpoint. "double" = pipe.transformer.transformer_blocks
# (joint image/text attention, separate QKV per stream). "single" =
# pipe.transformer.single_transformer_blocks (concatenated sequence,
# later in the network, pre_only attention as noted in flux.py).

def build_block_config(transformer):
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


# ================================================================
# 6. TOPK SPARSE AUTOENCODER (Sec 3.1)
# ================================================================

class TopKSAE(nn.Module):
    """
    z = TopK(ReLU(W_enc (x - b))),  xhat = W_dec z + b

    Matches the paper's formulation exactly (Sec 3.1): a single
    shared bias b (subtracted before encoding, added back after
    decoding), TopK activation instead of an L1 penalty, and
    columns of W_dec as the "concept vectors" f_i.
    """

    def __init__(self, d_in, n_features, k):
        super().__init__()

        self.d_in = d_in
        self.n_features = n_features
        self.k = k

        self.b = nn.Parameter(torch.zeros(d_in))

        # Standard TopK-SAE init: decoder columns as random unit
        # vectors, encoder tied to decoder's transpose at init.
        w_dec = torch.randn(d_in, n_features)
        w_dec = w_dec / w_dec.norm(dim=0, keepdim=True)

        self.W_dec = nn.Parameter(w_dec)
        self.W_enc = nn.Parameter(w_dec.t().clone())

    def encode(self, x):
        pre = F.relu((x - self.b) @ self.W_enc.t())

        if self.k >= self.n_features:
            return pre

        topk = torch.topk(pre, self.k, dim=-1)

        z = torch.zeros_like(pre)
        z.scatter_(-1, topk.indices, topk.values)

        return z

    def decode(self, z):
        return z @ self.W_dec.t() + self.b

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


def train_topk_sae(x, n_features, k, epochs, batch_size, lr, device, label=""):
    """
    Train one TopK SAE on activations x: [N, d_in].

    Loss is plain MSE reconstruction -- TopK activation enforces
    sparsity directly (no L1 term needed, consistent with the
    paper's choice of TopK over the ReLU+L1 baseline, Sec 2).
    """

    x = x.to(device)

    d_in = x.shape[-1]
    n_features = int(n_features)

    sae = TopKSAE(d_in, n_features, k).to(device)

    # Initialize bias to the data mean -- standard practice for
    # SAEs, keeps the pre-activation centered at init.
    with torch.no_grad():
        sae.b.copy_(x.mean(dim=0))

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    n = x.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)

        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = x[idx]

            xhat, z = sae(batch)
            loss = F.mse_loss(xhat, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.shape[0]

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"    [{label}] epoch {epoch:4d}  MSE {epoch_loss / n:.6f}")

    sae.eval()

    with torch.no_grad():
        xhat, _ = sae(x)
        residual_var = (x - xhat).var().item()
        total_var = x.var().item()
        explained_variance = 1.0 - residual_var / (total_var + 1e-12)

    print(f"    [{label}] explained variance: {explained_variance * 100:.1f}%")

    return sae, explained_variance


# ================================================================
# 7. STEP TRACKING (needed since FLUX's callback reports the
#    current denoising step, unlike flux.py's per-layer attention
#    processor which had no notion of "step" to filter on)
# ================================================================

class StepFilteredList:
    """
    Drop-in replacement for the plain list flux.py's
    RecordingFluxAttnProcessor appends to. Only keeps records whose
    step is in `target_steps`, and tags each record with the step
    it was recorded at -- both to bound memory (57 layers x 28 steps
    of full attention is large) and because we need per-bucket
    attention below.
    """

    def __init__(self, target_steps, step_ref):
        self.target_steps = target_steps
        self.step_ref = step_ref
        self.records = []

    def append(self, item):
        step = self.step_ref[0]

        if step in self.target_steps:
            item = dict(item)
            item["step"] = step
            self.records.append(item)

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


# ================================================================
# 8. BLOCK-LEVEL DELTA (Δ_{l,t}) HOOKS
# ================================================================
#
# NOTE ON DIFFUSERS VERSION DIFFERENCES: FluxTransformerBlock.forward
# (double-stream) returns (encoder_hidden_states, hidden_states) --
# text first, image second. FluxSingleTransformerBlock has changed
# across diffusers versions:
#
#   - Current diffusers (added for First Block Cache support, see
#     github.com/huggingface/diffusers/issues/12071): takes
#     `hidden_states` (image-only) and `encoder_hidden_states`
#     (text-only) as SEPARATE args, concatenates them internally,
#     and returns the same (encoder_hidden_states, hidden_states)
#     tuple as the double-stream blocks -- image tokens already
#     isolated in output[1], no slicing needed.
#
#   - Older diffusers: the block receives one pre-concatenated
#     [text; image] tensor as `hidden_states` and returns a single
#     raw tensor in that same layout (matching flux.py's original
#     RecordingFluxAttnProcessor comments) -- needs text_length
#     slicing on both ends.
#
# BlockDeltaRecorder below detects which behavior it's looking at
# from the actual shapes/types at runtime rather than assuming one,
# so it works either way -- but if results still look wrong, this is
# the first thing to check by hand against your installed version.

class BlockDeltaRecorder:
    """
    Records Δ = image_stream_output - image_stream_input for one
    block, at the diffusion steps in `target_steps`.
    """

    def __init__(self, block_kind, target_steps, step_ref, text_length):
        self.block_kind = block_kind          # "double" or "single"
        self.target_steps = target_steps
        self.step_ref = step_ref
        self.text_length = text_length

        # step -> Tensor[num_image_tokens, d]
        self.deltas = {}
        self._cached_input = None

    def pre_hook(self, module, args, kwargs):
        step = self.step_ref[0]

        if step not in self.target_steps:
            self._cached_input = None
            return

        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

        if self.block_kind == "double" or encoder_hidden_states is not None:
            # hidden_states is already image-only; text is (or would
            # be, for a double block) a separate argument. See NOTE.
            self._cached_input = hidden_states.detach()[0].clone()
        else:
            # Older diffusers: single-stream block received one
            # pre-concatenated [text; image] sequence.
            self._cached_input = (
                hidden_states.detach()[0, self.text_length:, :].clone()
            )

    def post_hook(self, module, args, kwargs, output):
        step = self.step_ref[0]

        if step not in self.target_steps or self._cached_input is None:
            return output

        if isinstance(output, tuple):
            # (encoder_hidden_states, hidden_states) -- image stream
            # is the second element, for both double-stream blocks
            # and (in current diffusers) single-stream blocks too.
            # See NOTE above.
            image_out = output[1].detach()[0].clone()
        else:
            # Older diffusers: single-stream block returns one raw
            # concatenated tensor.
            image_out = output.detach()[0, self.text_length:, :].clone()

        # Guard against a version mismatch producing mismatched
        # shapes (e.g. cached input sliced but output wasn't, or vice
        # versa) rather than silently recording a wrong delta.
        if image_out.shape != self._cached_input.shape:
            raise RuntimeError(
                f"Block delta shape mismatch for a {self.block_kind} block: "
                f"input {tuple(self._cached_input.shape)} vs "
                f"output {tuple(image_out.shape)}. This means the "
                f"diffusers-version auto-detection in BlockDeltaRecorder "
                f"guessed wrong -- see the NOTE above BlockDeltaRecorder."
            )

        delta = (image_out - self._cached_input).float().cpu()
        self.deltas[step] = delta
        self._cached_input = None

        return output


def install_block_hooks(transformer, block_config, step_ref, text_length):
    recorders = {}
    handles = []

    for name, (kind, index) in block_config.items():
        block = (
            transformer.transformer_blocks[index]
            if kind == "double"
            else transformer.single_transformer_blocks[index]
        )

        recorder = BlockDeltaRecorder(kind, TARGET_STEPS, step_ref, text_length)
        recorders[name] = recorder

        handles.append(block.register_forward_pre_hook(recorder.pre_hook, with_kwargs=True))
        handles.append(block.register_forward_hook(recorder.post_hook, with_kwargs=True))

    return recorders, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ================================================================
# 9. FULL EXTRACTION: ONE GENERATION -> ATTENTION + BLOCK DELTAS
# ================================================================

def extract_flux_activations(pipe, prompt, num_steps, seed, max_sequence_length,
                              block_config, text_length):
    """
    Runs one FLUX generation, recording:
      - joint image<->text attention at TARGET_STEPS only (for
        concept seeding, Sec 9 below), via flux.py's own
        RecordingFluxAttnProcessor -- unmodified.
      - per-block image-stream Δ at TARGET_STEPS, for each block in
        `block_config` (for SAE training/inference).
    """

    transformer = pipe.transformer

    # ------------------------------------------------------------
    # Step tracking, shared by both hook systems below.
    # ------------------------------------------------------------

    step_ref = [0]

    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        step_ref[0] = step_index
        return callback_kwargs

    # ------------------------------------------------------------
    # Attention processors (concept-seeding signal) -- same
    # processor class as flux.py, just fed a step-filtered storage.
    # ------------------------------------------------------------

    original_processors = dict(transformer.attn_processors)

    attention_storage = StepFilteredList(TARGET_STEPS, step_ref)

    recording_processors = {
        name: RecordingFluxAttnProcessor(layer_name=name, storage=attention_storage)
        for name in transformer.attn_processors
    }

    transformer.set_attn_processor(recording_processors)

    # ------------------------------------------------------------
    # Block-level Δ hooks (SAE input signal).
    # ------------------------------------------------------------

    recorders, handles = install_block_hooks(transformer, block_config, step_ref, text_length)

    # ------------------------------------------------------------
    # Generate.
    # ------------------------------------------------------------

    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=num_steps,
            guidance_scale=GUIDANCE_SCALE,
            max_sequence_length=max_sequence_length,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )

    # ------------------------------------------------------------
    # Restore.
    # ------------------------------------------------------------

    remove_hooks(handles)

    for name, processor in original_processors.items():
        transformer.set_attn_processor(processor)

    block_deltas = {
        name: recorder.deltas
        for name, recorder in recorders.items()
    }

    return result.images[0], attention_storage.records, block_deltas


# ================================================================
# 10. CONCEPT -> CID ASSIGNMENT
# ================================================================
#
# Paper's concept dictionary (Sec 3.3) labels each CID via vision
# models applied to thousands of generations, decoupled from any
# one prompt. Out of scope here (see header). Instead: within a
# single sample, use the concept's own T5-token attention (already
# recorded, and already correct/debugged in flux.py) as a cheap,
# text-grounded seed for which image tokens the concept touches,
# then pick the SAE latent whose decoder direction best explains
# the block's activity at exactly those tokens.

def concept_seed_weights(attention_records, step, concept_indices, num_image_tokens):
    """
    Average image-token attention to `concept_indices`'s T5 tokens,
    over every double-stream layer recorded at `step`. Returns a
    normalized [num_image_tokens] weight vector, or None if the
    concept's tokens never occurred in the prompt or no attention
    was recorded at this step.
    """

    if len(concept_indices) == 0:
        return None

    maps = []

    for record in attention_records:
        if record["step"] != step:
            continue

        attention = record["attention"]

        if attention.ndim == 3:
            attention = attention[0]

        if attention.shape[0] != num_image_tokens:
            # Different blocks can have different image-token counts
            # under some FLUX configs; skip mismatches defensively.
            continue

        concept_attention = attention[:, concept_indices].mean(dim=-1)
        maps.append(concept_attention.numpy())

    if len(maps) == 0:
        return None

    averaged = np.mean(maps, axis=0)

    return normalize_attention(averaged)


def assign_concept_to_cid(seed_weights, deltas, sae):
    """
    concept_prototype = weighted average of per-image-token deltas,
    weighted by the concept's attention seed. The assigned CID is
    the SAE decoder column (concept vector) most cosine-similar to
    that prototype -- i.e. the latent direction that best explains
    the residual update at the tokens this concept's text attends to.
    """

    w = torch.from_numpy(seed_weights).float()
    prototype = (w.unsqueeze(-1) * deltas).sum(dim=0)  # [d]

    prototype_norm = prototype / (prototype.norm() + 1e-12)
    w_dec = sae.W_dec.detach().cpu()
    w_dec_norm = w_dec / (w_dec.norm(dim=0, keepdim=True) + 1e-12)

    similarities = prototype_norm @ w_dec_norm  # [n_features]

    cid = int(torch.argmax(similarities).item())
    score = float(similarities[cid].item())

    return cid, score


# ================================================================
# 11. MAIN
# ================================================================

def main():
    print()
    print("=" * 70)
    print("STARTING FLUX SAE METAPHOR EXPERIMENT")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    device = "cuda"

    print("Loading FLUX...")

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

    # enable_model_cpu_offload() moves each top-level component (text
    # encoders, transformer, vae) onto the GPU AS A WHOLE when its
    # forward is called. FLUX.1-dev's transformer alone is ~12B params
    # (~24GB in bf16), so that single allocation can consume the
    # entire GPU with nothing left over for activations -- especially
    # on a shared GPU. enable_sequential_cpu_offload() streams
    # individual submodules onto the GPU one at a time instead
    # (slower, but the standard fix for FLUX-dev on 24GB cards).
    pipe.enable_sequential_cpu_offload(gpu_id=0)

    # Trim additional activation memory -- cheap to enable, no
    # quality cost for this analysis pipeline.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("FLUX loaded.")
    print()

    block_config = build_block_config(pipe.transformer)

    print("Block config (structural analog of down/mid/up_block):")
    for name, (kind, index) in block_config.items():
        print(f"  {name:14s} -> {kind} block #{index}")
    print()

    # ============================================================
    # PASS 1: generate every sample once, recording attention +
    # per-block deltas at TARGET_STEPS.
    # ============================================================

    sample_records = []

    # (block_name, bucket) -> list of Tensor[num_image_tokens, d]
    training_pool = {}

    tokenizer = pipe.tokenizer_2  # T5 -- see flux.py's own note on why.

    for sample in DATASET:
        slug = f"sample_{sample['id']:02d}_{slugify(sample['title'])}"

        prompt = sample["prompt"]
        concepts = concepts_dict_for_sample(sample)
        seed = SEED + sample["id"]

        print("=" * 70)
        print(f"SAMPLE {sample['id']} [{sample['source']}] {sample['title']}")
        print("=" * 70)
        print()

        concept_token_indices = {
            concept: get_token_indices(tokenizer, prompt, concept, max_length=MAX_SEQUENCE_LENGTH)
            for concept in concepts
        }

        for concept, indices in concept_token_indices.items():
            print(f"  {concept:20s}", indices)
        print()

        image, attention_records, block_deltas = extract_flux_activations(
            pipe,
            prompt,
            NUM_INFERENCE_STEPS,
            seed=seed,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            block_config=block_config,
            text_length=MAX_SEQUENCE_LENGTH,
        )

        image_path = os.path.join(IMAGE_DIR, f"{slug}_sae.png")
        image.save(image_path)
        print("Image saved:", image_path)

        sample_records.append({
            "sample": sample,
            "concepts": concepts,
            "concept_token_indices": concept_token_indices,
            "attention_records": attention_records,
            "block_deltas": block_deltas,
            "slug": slug,
        })

        for name in block_config:
            for step, delta in block_deltas[name].items():
                bucket = STEP_TO_BUCKET[step]
                key = (name, bucket)
                training_pool.setdefault(key, []).append(delta)

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # TRAIN ONE SAE PER (block, timestep bucket)
    # ============================================================

    print()
    print("=" * 70)
    print("TRAINING SAEs")
    print("=" * 70)
    print()

    saes = {}
    sae_diagnostics = {}

    for key, tensors in training_pool.items():
        block_name, bucket = key
        label = f"{block_name}/{bucket}"

        x = torch.cat(tensors, dim=0)
        d = x.shape[-1]
        n_features = SAE_EXPANSION_FACTOR * d

        print(f"  Training SAE for {label}: N={x.shape[0]}, d={d}, "
              f"n_features={n_features}, k={SAE_TOPK}")

        sae, explained_variance = train_topk_sae(
            x, n_features, SAE_TOPK, SAE_EPOCHS, SAE_BATCH_SIZE, SAE_LR, device, label=label,
        )

        saes[key] = sae
        sae_diagnostics[label] = {
            "n_train_vectors": int(x.shape[0]),
            "d_in": int(d),
            "n_features": int(n_features),
            "k": SAE_TOPK,
            "explained_variance": explained_variance,
        }

        torch.save(
            sae.state_dict(),
            os.path.join(SAE_WEIGHTS_DIR, f"sae_{block_name}_{bucket}.pt"),
        )

    with open(os.path.join(SAE_RESULTS_DIR, "sae_diagnostics.json"), "w") as f:
        json.dump(sae_diagnostics, f, indent=4)

    print()
    print("SAE diagnostics saved:", os.path.join(SAE_RESULTS_DIR, "sae_diagnostics.json"))
    print()

    # ============================================================
    # PASS 2: per sample, per (block, bucket) -- assign concepts to
    # CIDs, build SAE-derived concept maps, compute flux.py's
    # unchanged metrics on them.
    # ============================================================

    print()
    print("=" * 70)
    print("BUILDING SAE CONCEPT MAPS + METRICS")
    print("=" * 70)
    print()

    all_results = []

    for record in sample_records:
        sample = record["sample"]
        concepts = record["concepts"]
        concept_token_indices = record["concept_token_indices"]
        attention_records = record["attention_records"]
        block_deltas = record["block_deltas"]
        slug = record["slug"]

        sample_results = {}

        for key, sae in saes.items():
            block_name, bucket = key
            step = TIMESTEP_BUCKET_STEPS[bucket]

            if step not in block_deltas[block_name]:
                continue  # this block/step wasn't recorded for this sample

            deltas = block_deltas[block_name][step]  # [num_image_tokens, d]
            num_image_tokens = deltas.shape[0]

            with torch.no_grad():
                z = sae.encode(deltas.to(next(sae.parameters()).device)).cpu()

            concept_maps = {}
            cid_assignments = {}

            for concept, indices in concept_token_indices.items():
                weights = concept_seed_weights(
                    attention_records, step, indices, num_image_tokens
                )

                if weights is None:
                    concept_maps[concept] = None
                    continue

                cid, score = assign_concept_to_cid(weights, deltas, sae)

                cid_assignments[concept] = {"cid": cid, "cosine_similarity": score}

                concept_maps[concept] = token_attention_to_spatial(
                    z[:, cid].numpy()
                )

            metrics = compute_metrics(concept_maps, concepts)
            metrics["sae"] = {
                "block": block_name,
                "timestep_bucket": bucket,
                "cid_assignments": cid_assignments,
            }

            sample_results[f"{block_name}__{bucket}"] = metrics

            # Save raw concept maps for this (sample, block, bucket).
            map_dir = os.path.join(SAE_MAP_DIR, slug, f"{block_name}_{bucket}")
            os.makedirs(map_dir, exist_ok=True)

            for concept, amap in concept_maps.items():
                if amap is None:
                    continue
                np.save(os.path.join(map_dir, f"{concept}.npy"), amap)

        sample_results["experiment"] = {
            "sample_id": sample["id"],
            "source": sample["source"],
            "title": sample["title"],
            "model": MODEL_ID,
            "seed": SEED + sample["id"],
            "prompt": sample["prompt"],
        }

        metrics_path = os.path.join(SAE_METRIC_DIR, f"{slug}.json")
        with open(metrics_path, "w") as f:
            json.dump(sample_results, f, indent=4)

        print("Metrics saved:", metrics_path)

        all_results.append(sample_results)

    combined_path = os.path.join(SAE_METRIC_DIR, "all_sae_metrics.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print()
    print("Combined SAE metrics saved:", combined_path)
    print()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 70)
    print("SAE EXPERIMENT COMPLETE")
    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()