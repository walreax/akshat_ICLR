# ================================================================
# FLUX.1-DEV — MECHANISTIC METAPHOR ANALYSIS
# ================================================================
#
# Model:
#   black-forest-labs/FLUX.1-dev
#
# Precision:
#   BF16
#
# Quantization:
#   NONE
#
# GPU:
#   Physical GPU 1
#
# Hardware:
#   RTX A5000 24 GB
#
# Metrics:
#
#   1. Concept Centroid Deviation
#   2. Cross-Concept Index
#   3. Attention Concentration Index
#   4. Attention Entropy
#
# ================================================================


# ================================================================
# 1. GPU CONFIGURATION
# ================================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ[
    "PYTORCH_CUDA_ALLOC_CONF"
] = "expandable_segments:True"


# ================================================================
# 2. IMPORTS
# ================================================================

import gc
import json
import math
import time

import numpy as np

import torch

from PIL import Image

from diffusers import FluxPipeline


# ================================================================
# 3. CONFIGURATION
# ================================================================

MODEL_ID = (
    "black-forest-labs/FLUX.1-schnell"
)


PROJECT_DIR = (
    "/DATA/swagata/akshat_ICLR"
)


RESULTS_DIR = (
    os.path.join(
        PROJECT_DIR,
        "flux_results"
    )
)


IMAGE_DIR = (
    os.path.join(
        RESULTS_DIR,
        "images"
    )
)


ATTENTION_DIR = (
    os.path.join(
        RESULTS_DIR,
        "attention_maps"
    )
)


METRIC_DIR = (
    os.path.join(
        RESULTS_DIR,
        "metrics"
    )
)


for directory in [
    RESULTS_DIR,
    IMAGE_DIR,
    ATTENTION_DIR,
    METRIC_DIR
]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ================================================================
# 4. EXPERIMENT CONFIGURATION
# ================================================================

SEED = 42


HEIGHT = 512


WIDTH = 512


# FLUX.1-schnell is a distilled, guidance-free model: its intended
# operating point is 4 steps with guidance_scale=0 (not dev's 28 /
# 3.5) -- passing dev's settings runs schnell "off-label" and does
# not match how it was trained/distilled.
NUM_INFERENCE_STEPS = 4


GUIDANCE_SCALE = 0.0


# FLUX's transformer only ever sees the T5-encoded sequence -- CLIP
# contributes a single pooled vector and nothing per-token. This has
# to match whatever `max_sequence_length` the pipe() call uses (see
# extract_flux_attention), or concept token indices are computed
# against a differently-padded sequence than the one attention is
# actually recorded over.
#
# schnell caps at 256 tokens (dev supports up to 512) -- passing a
# larger value raises a ValueError.
MAX_SEQUENCE_LENGTH = 256


DTYPE = torch.bfloat16


# ================================================================
# 5. METAPHOR DATASET
# ================================================================
#
# 10 samples: 5 sourced as short poem-style metaphors ("PoemSum"),
# 5 written as short story vignettes ("Visual Storytelling"). Each
# has two concrete (directly visualizable) and two abstract
# (metaphorical) concepts.
#
# IMPORTANT: every concrete/abstract concept string must appear
# verbatim inside its sample's `prompt` -- get_token_indices() finds
# a concept by matching its own token-id sequence as a contiguous
# subsequence of the prompt's token ids. A concept whose exact text
# never appears in the prompt will just come back with an empty
# index list, and every metric for it silently degrades to `None`.
# ================================================================

STYLE_SUFFIX = (
    ", surrealist cinematic artwork, "
    "dark atmospheric lighting, "
    "highly detailed, "
    "dramatic composition"
)


DATASET = [

    # ------------------------------------------------------------
    # PoemSum (5)
    # ------------------------------------------------------------

    {
        "id": 1,
        "source": "PoemSum",
        "title": "The Cage of Time",
        "prompt": (
            "a gilded hourglass trapping a bird, "
            "symbolizing fleeting time and mortality"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["hourglass", "bird"],
        "abstract_concepts": ["time", "mortality"],
    },

    {
        "id": 2,
        "source": "PoemSum",
        "title": "The Shattered Mirror",
        "prompt": (
            "a cracked mirror reflecting a stormy ocean, "
            "signifying inner turmoil and identity"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["mirror", "ocean"],
        "abstract_concepts": ["turmoil", "identity"],
    },

    {
        "id": 3,
        "source": "PoemSum",
        "title": "The Thread of Fate",
        "prompt": (
            "a golden thread weaving through dark labyrinth walls, "
            "embodying guidance and destiny"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["thread", "labyrinth"],
        "abstract_concepts": ["guidance", "destiny"],
    },

    {
        "id": 4,
        "source": "PoemSum",
        "title": "The Weight of Silence",
        "prompt": (
            "a rusted scale balanced by a single feather, "
            "symbolizing crushing silence and impossible burden"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["scale", "feather"],
        "abstract_concepts": ["silence", "burden"],
    },

    {
        "id": 5,
        "source": "PoemSum",
        "title": "The River of Memory",
        "prompt": (
            "a winding river flowing through ancient stone, "
            "representing fading memory and quiet forgetting"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["river", "stone"],
        "abstract_concepts": ["memory", "forgetting"],
    },

    # ------------------------------------------------------------
    # Visual Storytelling (5) -- written as short story vignettes
    # ------------------------------------------------------------

    {
        "id": 6,
        "source": "Visual Storytelling",
        "title": "The Broken Anchor",
        "prompt": (
            "a rusted anchor underwater connected to a glowing heart, "
            "representing heavy grief and hope"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["anchor", "heart"],
        "abstract_concepts": ["grief", "hope"],
    },

    {
        "id": 7,
        "source": "Visual Storytelling",
        "title": "The Melting Fortress",
        "prompt": (
            "a fortress made of ice melting under a burning sun, "
            "depicting pride and downfall"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["fortress", "sun"],
        "abstract_concepts": ["pride", "downfall"],
    },

    {
        "id": 8,
        "source": "Visual Storytelling",
        "title": "The Sailor's Fading Ship",
        "prompt": (
            "an old sailor stands alone on a weathered dock as his ship "
            "dissolves into the fog, capturing his fading memory and "
            "deep loneliness"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["sailor", "ship"],
        "abstract_concepts": ["memory", "loneliness"],
    },

    {
        "id": 9,
        "source": "Visual Storytelling",
        "title": "The Child and the Lantern",
        "prompt": (
            "a small child carries a flickering lantern through a dark "
            "forest, embodying fragile innocence facing quiet fear"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["child", "lantern"],
        "abstract_concepts": ["innocence", "fear"],
    },

    {
        "id": 10,
        "source": "Visual Storytelling",
        "title": "The Soldier's Broken Shield",
        "prompt": (
            "a wounded soldier leans on a cracked shield beneath a "
            "setting sun, portraying quiet courage and deep exhaustion"
            + STYLE_SUFFIX
        ),
        "concrete_concepts": ["soldier", "shield"],
        "abstract_concepts": ["courage", "exhaustion"],
    },

]


def concepts_dict_for_sample(sample):
    """
    Build the {concept: {"type": "concrete" | "metaphorical"}} dict
    that get_token_indices()/compute_metrics() expect, from a
    dataset sample's concrete_concepts / abstract_concepts lists.
    """

    concepts = {}

    for concept in sample["concrete_concepts"]:
        concepts[concept] = {"type": "concrete"}

    for concept in sample["abstract_concepts"]:
        concepts[concept] = {"type": "metaphorical"}

    return concepts


# ================================================================
# 6. TOKENIZATION UTILITIES
# ================================================================

def get_token_indices(
    tokenizer,
    prompt,
    concept,
    max_length=None
):

    """
    Find the token positions corresponding
    to a concept.

    `max_length` should match whatever padding
    length the sequence being indexed into was
    actually produced with (see MAX_SEQUENCE_LENGTH).
    Defaults to the tokenizer's own model_max_length
    if not given.
    """

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=(
            max_length
            if max_length is not None
            else tokenizer.model_max_length
        ),
        truncation=True
    )

    input_ids = (
        encoded.input_ids[0]
    )

    concept_ids = tokenizer(
        concept,
        add_special_tokens=False
    ).input_ids

    prompt_ids = (
        input_ids.tolist()
    )


    matches = []


    concept_length = (
        len(concept_ids)
    )


    for i in range(
        len(prompt_ids)
        -
        concept_length
        +
        1
    ):

        if (
            prompt_ids[
                i:i + concept_length
            ]
            ==
            concept_ids
        ):

            matches.extend(
                range(
                    i,
                    i + concept_length
                )
            )


    return sorted(
        list(
            set(matches)
        )
    )


# ================================================================
# 7. NORMALIZE ATTENTION MAP
# ================================================================

def normalize_attention(
    attention
):

    """
    Normalize an attention map
    into a probability distribution.
    """

    attention = (
        attention.astype(
            np.float64
        )
    )


    total = (
        attention.sum()
    )


    if total <= 0:

        return (
            np.ones_like(
                attention
            )
            /
            attention.size
        )


    return (
        attention
        /
        total
    )


# ================================================================
# 8. CONCEPT CENTROID
# ================================================================

def calculate_centroid(
    attention
):

    """
    Calculate the spatial center of
    an attention distribution.

    Coordinates are normalized to [0, 1].
    """

    probability = (
        normalize_attention(
            attention
        )
    )


    height, width = (
        probability.shape
    )


    y_coordinates = (
        np.linspace(
            0,
            1,
            height
        )
    )


    x_coordinates = (
        np.linspace(
            0,
            1,
            width
        )
    )


    xx, yy = np.meshgrid(
        x_coordinates,
        y_coordinates
    )


    centroid_x = (
        probability * xx
    ).sum()


    centroid_y = (
        probability * yy
    ).sum()


    return np.array([
        centroid_x,
        centroid_y
    ])


# ================================================================
# 9. CENTROID DEVIATION
# ================================================================

def centroid_deviation(
    concrete_map,
    metaphorical_map
):

    """
    Euclidean distance between the
    concrete concept centroid and
    metaphorical concept centroid.
    """

    concrete_centroid = (
        calculate_centroid(
            concrete_map
        )
    )


    metaphorical_centroid = (
        calculate_centroid(
            metaphorical_map
        )
    )


    distance = np.linalg.norm(
        concrete_centroid
        -
        metaphorical_centroid
    )


    return {

        "concrete_centroid":
            concrete_centroid.tolist(),

        "metaphorical_centroid":
            metaphorical_centroid.tolist(),

        "distance":
            float(distance)

    }


# ================================================================
# 10. CROSS-CONCEPT INDEX
# ================================================================

def cross_concept_index(
    map_a,
    map_b
):

    """
    Cosine similarity between two
    spatial attention distributions.
    """

    a = (
        map_a
        .astype(np.float64)
        .flatten()
    )


    b = (
        map_b
        .astype(np.float64)
        .flatten()
    )


    norm_a = np.linalg.norm(
        a
    )


    norm_b = np.linalg.norm(
        b
    )


    if (
        norm_a == 0
        or
        norm_b == 0
    ):

        return 0.0


    return float(
        np.dot(
            a,
            b
        )
        /
        (
            norm_a
            *
            norm_b
        )
    )


# ================================================================
# 11. ATTENTION ENTROPY
# ================================================================

def attention_entropy(
    attention
):

    """
    Normalized spatial attention entropy.
    """

    probability = (
        normalize_attention(
            attention
        )
    )


    entropy = -np.sum(
        probability
        *
        np.log(
            probability
            +
            1e-12
        )
    )


    maximum_entropy = np.log(
        probability.size
    )


    return float(
        entropy
        /
        (
            maximum_entropy
            +
            1e-12
        )
    )


# ================================================================
# 12. ATTENTION CONCENTRATION INDEX
# ================================================================

def attention_concentration_index(
    attention
):

    """
    ACI = 1 - normalized entropy.

    ACI close to 1:
        concentrated attention

    ACI close to 0:
        diffuse attention
    """

    entropy = (
        attention_entropy(
            attention
        )
    )


    return float(
        1.0 - entropy
    )


# ================================================================
# 13. ATTENTION HOOK
# ================================================================

class FluxAttentionRecorder:

    """
    Records attention information from
    the FLUX transformer.

    The exact internals of FLUX differ
    from Stable Diffusion because FLUX
    uses joint image/text attention.
    """

    def __init__(self):

        self.records = []

        self.current_step = 0

        self.current_block = 0


    def reset(self):

        self.records = []

        self.current_step = 0

        self.current_block = 0


    def record(
        self,
        attention
    ):

        self.records.append({

            "step":
                self.current_step,

            "block":
                self.current_block,

            "attention":
                attention.detach().cpu()

        })


# ================================================================
# 14. FLUX ATTENTION EXTRACTION
# ================================================================

def extract_flux_attention(
    pipe,
    prompt,
    concept_tokens,
    num_steps,
    seed,
    max_sequence_length
):

    """
    Extract image-query -> text-token
    attention from FLUX.

    NOTE:
    FLUX uses joint attention, so the
    sequence contains both text and image
    representations.

    We extract the text columns corresponding
    to the supplied concept tokens.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "ATTENTION EXTRACTION"
    )

    print(
        "=" * 70
    )

    print()


    # ------------------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------------------
    #
    # We install a temporary processor that
    # records the attention matrix.
    #
    # The actual processor is implemented
    # below.
    # ------------------------------------------------------------


    attention_records = []


    original_processors = {}


    transformer = (
        pipe.transformer
    )


    # ------------------------------------------------------------
    # Save original processors
    # ------------------------------------------------------------

    for name, processor in (
        transformer.attn_processors.items()
    ):

        original_processors[
            name
        ] = processor


    # ------------------------------------------------------------
    # Install recording processors
    # ------------------------------------------------------------

    # `set_attn_processor(single_processor)` installs that ONE
    # processor on every attention layer in the transformer -- it
    # does not target just `name`. Calling it once per name in a
    # loop was overwriting all 57 layers on every iteration, so
    # every block ended up sharing the *last* loop iteration's
    # processor (and its layer_name). Building the full dict and
    # setting it once is the correct usage: it maps each named layer
    # to its own processor instance.

    recording_processors = {
        name: RecordingFluxAttnProcessor(
            layer_name=name,
            storage=attention_records,
        )
        for name in transformer.attn_processors
    }

    transformer.set_attn_processor(recording_processors)


    # ------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------

    generator = (
        torch.Generator(
            device="cpu"
        )
        .manual_seed(
            seed
        )
    )


    with torch.inference_mode():

        result = pipe(

            prompt=prompt,

            height=HEIGHT,

            width=WIDTH,

            num_inference_steps=num_steps,

            guidance_scale=(
                GUIDANCE_SCALE
            ),

            max_sequence_length=(
                max_sequence_length
            ),

            generator=generator

        )


    # ------------------------------------------------------------
    # Restore processors
    # ------------------------------------------------------------

    for name, processor in (
        original_processors.items()
    ):

        transformer.set_attn_processor(
            processor
        )


    return (
        result.images[0],
        attention_records
    )


# ================================================================
# 15. RECORDING FLUX ATTENTION PROCESSOR
# ================================================================

class RecordingFluxAttnProcessor:

    """
    FLUX attention processor.

    This processor mirrors the attention
    calculation while retaining the
    image -> text portion needed for
    mechanistic analysis.
    """

    def __init__(
        self,
        layer_name,
        storage
    ):

        self.layer_name = (
            layer_name
        )

        self.storage = (
            storage
        )


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs
    ):

        # --------------------------------------------------------
        # WHAT WAS WRONG BEFORE (kept here as a note, since the
        # traceback only showed the first of several bugs):
        #
        # 1. norm_q/norm_k were applied to the [batch, seq,
        #    inner_dim] tensor, before it was split into heads.
        #    attn.norm_q / attn.norm_k are RMSNorm(head_dim) --
        #    they need [..., head_dim] as the last axis, not
        #    [..., inner_dim]. That's the crash you hit.
        #
        # 2. `hidden_states` for a double-stream block is ONLY the
        #    image tokens -- the text tokens live in
        #    `encoder_hidden_states` and go through their own
        #    add_q_proj/add_k_proj/add_v_proj weights. The old code
        #    never called those, so `query`/`key`/`value` never
        #    contained the text tokens at all: this was computing
        #    image self-attention and slicing a meaningless chunk
        #    out of it, not real image-to-text attention.
        #
        # 3. `image_rotary_emb` was accepted as a parameter but
        #    never applied. FLUX has no positional info without it;
        #    every generation run through this processor (even
        #    outside of the recording steps) was structurally
        #    broken.
        #
        # 4. This branch must return a (hidden_states,
        #    encoder_hidden_states) TUPLE when encoder_hidden_states
        #    is not None -- FluxTransformerBlock.forward() unpacks
        #    two values. The old code always returned one tensor,
        #    which would have failed on the very next line once (1)
        #    was fixed.
        # --------------------------------------------------------

        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # --------------------------------------------------------
        # Query / Key / Value (image stream)
        # --------------------------------------------------------

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = query.shape[-1]
        num_heads = attn.heads
        head_dim = inner_dim // num_heads

        # Split into heads FIRST, then normalize per-head.
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)

        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        # --------------------------------------------------------
        # Query / Key / Value (text stream), double-stream blocks
        # only. Projected separately, normalized separately, then
        # joined with the image stream so both attend over one
        # shared sequence.
        # --------------------------------------------------------

        text_length = 0

        if encoder_hidden_states is not None:

            enc_query = attn.add_q_proj(encoder_hidden_states)
            enc_key = attn.add_k_proj(encoder_hidden_states)
            enc_value = attn.add_v_proj(encoder_hidden_states)

            enc_query = enc_query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            enc_key = enc_key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            enc_value = enc_value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            if getattr(attn, "norm_added_q", None) is not None:
                enc_query = attn.norm_added_q(enc_query)

            if getattr(attn, "norm_added_k", None) is not None:
                enc_key = attn.norm_added_k(enc_key)

            text_length = enc_query.shape[2]

            # Text tokens first, then image tokens -- this is the
            # ordering FLUX uses for the joint sequence everywhere
            # (rotary embeddings, the single-stream blocks, etc).
            query = torch.cat([enc_query, query], dim=2)
            key = torch.cat([enc_key, key], dim=2)
            value = torch.cat([enc_value, value], dim=2)

        # --------------------------------------------------------
        # Rotary position embeddings.
        # --------------------------------------------------------

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # --------------------------------------------------------
        # Attention scores (computed manually so the probability
        # matrix exists to record from -- fused SDPA never
        # materializes it).
        # --------------------------------------------------------

        scale = head_dim ** -0.5

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # --------------------------------------------------------
        # Store attention: image-query rows against text-key
        # columns only, averaged over heads.
        # [batch, image_tokens, text_length]
        # --------------------------------------------------------

        if encoder_hidden_states is not None:

            image_queries = attention_probs[:, :, text_length:, :]
            image_to_text = image_queries[..., :text_length]
            image_to_text = image_to_text.mean(dim=1)

            self.storage.append({

                "layer":
                    self.layer_name,

                "attention":
                    image_to_text.detach()
                    .float()
                    .cpu()

            })

        # --------------------------------------------------------
        # Produce normal attention output
        # --------------------------------------------------------

        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        output = output.to(query.dtype)

        if encoder_hidden_states is not None:

            # Split the joint output back into its text and image
            # portions and run each through its own output
            # projection -- to_out for the image stream, to_add_out
            # for the text stream.

            encoder_hidden_states_out, hidden_states_out = (
                output[:, :text_length],
                output[:, text_length:],
            )

            hidden_states_out = attn.to_out[0](hidden_states_out)

            if len(attn.to_out) > 1 and attn.to_out[1] is not None:
                hidden_states_out = attn.to_out[1](hidden_states_out)

            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

            return hidden_states_out, encoder_hidden_states_out

        else:

            # Single-stream blocks build their attention with
            # `pre_only=True` -- per FluxAttention.__init__, `to_out`
            # is only created `if not self.pre_only`, so it does not
            # exist here. FluxSingleTransformerBlock projects the
            # output itself (via its own `proj_out`, after
            # concatenating this with its parallel MLP branch), so
            # we just hand back the raw reshaped attention output --
            # exactly what the stock FluxAttnProcessor does in this
            # branch.

            return output


# ================================================================
# 16. CONVERT TOKEN ATTENTION TO SPATIAL MAP
# ================================================================

def token_attention_to_spatial(
    attention
):

    """
    Convert image-token attention into
    a 2D spatial map.

    FLUX operates over image latent tokens.
    """

    # attention:
    #
    # 1D array of length num_image_tokens


    image_token_attention = (
        np.asarray(
            attention
        )
    )


    # ------------------------------------------------------------
    # Number of image tokens
    # ------------------------------------------------------------

    num_image_tokens = (
        image_token_attention.shape[0]
    )


    # ------------------------------------------------------------
    # Infer spatial dimensions
    # ------------------------------------------------------------

    spatial_size = int(
        math.sqrt(
            num_image_tokens
        )
    )


    if (
        spatial_size
        *
        spatial_size
        !=
        num_image_tokens
    ):

        raise ValueError(
            "Image token count is not square: "
            +
            str(
                num_image_tokens
            )
        )


    spatial_map = (
        image_token_attention
        .reshape(
            spatial_size,
            spatial_size
        )
    )


    return spatial_map


# ================================================================
# 17. AGGREGATE CONCEPT ATTENTION
# ================================================================

def aggregate_concept_attention(
    attention_records,
    concept_token_indices
):

    """
    Combine token-level attention belonging
    to the same concept.

    Returns one spatial map per concept.
    """

    concept_maps = {}


    for concept, indices in (
        concept_token_indices.items()
    ):

        maps = []


        for record in (
            attention_records
        ):

            attention = (
                record[
                    "attention"
                ]
            )


            # ----------------------------------------------------
            # [batch, image_tokens, text_tokens]
            # ----------------------------------------------------

            if attention.ndim == 3:

                attention = (
                    attention[0]
                )


            # ----------------------------------------------------
            # Select concept tokens
            # ----------------------------------------------------

            if len(indices) == 0:

                continue


            concept_attention = (
                attention[
                    :,
                    indices
                ]
                .mean(
                    dim=-1
                )
            )


            maps.append(
                concept_attention
                .numpy()
            )


        if len(maps) == 0:

            concept_maps[
                concept
            ] = None

            continue


        # --------------------------------------------------------
        # Average across layers, then reshape the flat per-image-
        # token vector into a 2D spatial map. Every downstream
        # metric (calculate_centroid, attention_entropy, ACI, CCI)
        # expects a (height, width) grid, not a flat vector -- this
        # reshape step was defined (token_attention_to_spatial) but
        # never actually called, which is why calculate_centroid's
        # `height, width = probability.shape` failed: probability
        # was still 1D.
        # --------------------------------------------------------

        averaged = np.mean(
            maps,
            axis=0
        )

        concept_maps[
            concept
        ] = token_attention_to_spatial(
            averaged
        )


    return concept_maps


# ================================================================
# 18. COMPUTE ALL METRICS
# ================================================================

def compute_metrics(
    concept_maps,
    concepts
):

    print()

    print(
        "=" * 70
    )

    print(
        "COMPUTING METRICS"
    )

    print(
        "=" * 70
    )

    print()


    metrics = {

        "concept_metrics": {},

        "centroid_deviation": {},

        "cross_concept_index": {},

    }


    # ============================================================
    # Individual concept metrics
    # ============================================================

    for concept, amap in (
        concept_maps.items()
    ):

        if amap is None:

            continue


        centroid = (
            calculate_centroid(
                amap
            )
        )


        entropy = (
            attention_entropy(
                amap
            )
        )


        aci = (
            attention_concentration_index(
                amap
            )
        )


        metrics[
            "concept_metrics"
        ][concept] = {

            "centroid":
                centroid.tolist(),

            "attention_entropy":
                entropy,

            "attention_concentration_index":
                aci

        }


    # ============================================================
    # Concrete / metaphorical pairs
    # ============================================================

    concrete = [
        concept
        for concept, info
        in concepts.items()
        if info["type"]
        ==
        "concrete"
    ]


    metaphorical = [
        concept
        for concept, info
        in concepts.items()
        if info["type"]
        ==
        "metaphorical"
    ]


    # ============================================================
    # Centroid deviation
    # ============================================================

    for c in concrete:

        if (
            c not in concept_maps
            or
            concept_maps[c] is None
        ):

            continue


        for m in metaphorical:

            if (
                m not in concept_maps
                or
                concept_maps[m] is None
            ):

                continue


            result = (
                centroid_deviation(
                    concept_maps[c],
                    concept_maps[m]
                )
            )


            metrics[
                "centroid_deviation"
            ][
                f"{c}__{m}"
            ] = result


    # ============================================================
    # Cross Concept Index
    # ============================================================

    for c in concrete:

        if (
            c not in concept_maps
            or
            concept_maps[c] is None
        ):

            continue


        for m in metaphorical:

            if (
                m not in concept_maps
                or
                concept_maps[m] is None
            ):

                continue


            cci = (
                cross_concept_index(
                    concept_maps[c],
                    concept_maps[m]
                )
            )


            metrics[
                "cross_concept_index"
            ][
                f"{c}__{m}"
            ] = cci


    return metrics


# ================================================================
# 19. MAIN
# ================================================================

def slugify(text):
    return "".join(
        c if c.isalnum() else "_"
        for c in text.lower()
    ).strip("_")


def main():

    print()

    print("=" * 70)

    print(
        "STARTING FLUX METAPHOR EXPERIMENT"
    )

    print("=" * 70)

    print()

    print(
        f"{len(DATASET)} samples "
        f"({sum(1 for s in DATASET if s['source'] == 'PoemSum')} PoemSum, "
        f"{sum(1 for s in DATASET if s['source'] == 'Visual Storytelling')} "
        "Visual Storytelling)"
    )

    print()


    # ============================================================
    # CUDA
    # ============================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )


    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


    print()


    # ============================================================
    # LOAD PIPELINE (once, reused across every sample)
    # ============================================================

    print(
        "Loading FLUX..."
    )


    pipe = FluxPipeline.from_pretrained(

        MODEL_ID,

        torch_dtype=DTYPE

    )


    # ============================================================
    # CPU OFFLOAD
    # ============================================================

    pipe.enable_model_cpu_offload(
        gpu_id=0
    )


    print(
        "FLUX loaded."
    )


    print()


    # ============================================================
    # RUN EVERY SAMPLE
    # ============================================================

    all_results = []


    for sample in DATASET:

        slug = (
            f"sample_{sample['id']:02d}_"
            f"{slugify(sample['title'])}"
        )

        prompt = sample["prompt"]
        concepts = concepts_dict_for_sample(sample)
        seed = SEED + sample["id"]

        print(
            "=" * 70
        )

        print(
            f"SAMPLE {sample['id']} "
            f"[{sample['source']}] "
            f"{sample['title']}"
        )

        print(
            "=" * 70
        )

        print()


        # --------------------------------------------------------
        # TOKENIZE CONCEPTS
        #
        # Use tokenizer_2 (T5), not tokenizer (CLIP): FLUX's
        # transformer only ever attends over the T5-encoded
        # sequence -- CLIP contributes just one pooled vector and
        # is never seen per-token by the attention we're recording.
        # Indices computed from the CLIP tokenizer would point at
        # positions in a sequence the transformer never sees.
        # `max_length` is pinned to MAX_SEQUENCE_LENGTH so this
        # matches exactly what extract_flux_attention's pipe() call
        # below actually encodes the prompt with.
        # --------------------------------------------------------

        print(
            "Tokenizing concepts (T5)..."
        )

        print()


        tokenizer = (
            pipe.tokenizer_2
        )


        concept_token_indices = {}


        for concept in concepts:

            indices = (
                get_token_indices(
                    tokenizer,
                    prompt,
                    concept,
                    max_length=MAX_SEQUENCE_LENGTH
                )
            )


            concept_token_indices[
                concept
            ] = indices


            print(
                f"  {concept:20s}",
                indices
            )


        print()


        # --------------------------------------------------------
        # GENERATE + RECORD ATTENTION
        # --------------------------------------------------------

        image, attention_records = (
            extract_flux_attention(

                pipe,

                prompt,

                concept_token_indices,

                NUM_INFERENCE_STEPS,

                seed=seed,

                max_sequence_length=MAX_SEQUENCE_LENGTH

            )
        )


        # --------------------------------------------------------
        # SAVE IMAGE
        # --------------------------------------------------------

        image_path = (
            os.path.join(
                IMAGE_DIR,
                f"{slug}.png"
            )
        )


        image.save(
            image_path
        )


        print(
            "Image saved:",
            image_path
        )


        # --------------------------------------------------------
        # AGGREGATE CONCEPT MAPS
        # --------------------------------------------------------

        concept_maps = (
            aggregate_concept_attention(

                attention_records,

                concept_token_indices

            )
        )


        # --------------------------------------------------------
        # SAVE ATTENTION MAPS (per-sample subfolder)
        # --------------------------------------------------------

        sample_attention_dir = (
            os.path.join(
                ATTENTION_DIR,
                slug
            )
        )

        os.makedirs(
            sample_attention_dir,
            exist_ok=True
        )


        for concept, amap in (
            concept_maps.items()
        ):

            if amap is None:

                continue


            path = (
                os.path.join(
                    sample_attention_dir,
                    f"{concept}.npy"
                )
            )


            np.save(
                path,
                amap
            )


        # --------------------------------------------------------
        # COMPUTE METRICS
        # --------------------------------------------------------

        metrics = (
            compute_metrics(

                concept_maps,

                concepts

            )
        )


        # --------------------------------------------------------
        # ADD EXPERIMENT METADATA
        # --------------------------------------------------------

        metrics[
            "experiment"
        ] = {

            "sample_id":
                sample["id"],

            "source":
                sample["source"],

            "title":
                sample["title"],

            "model":
                MODEL_ID,

            "precision":
                "BF16",

            "quantization":
                "NONE",

            "gpu":
                torch.cuda.get_device_name(0),

            "physical_gpu":
                1,

            "prompt":
                prompt,

            "seed":
                seed,

            "height":
                HEIGHT,

            "width":
                WIDTH,

            "steps":
                NUM_INFERENCE_STEPS,

            "guidance_scale":
                GUIDANCE_SCALE,

            "max_sequence_length":
                MAX_SEQUENCE_LENGTH

        }


        # --------------------------------------------------------
        # SAVE PER-SAMPLE METRICS
        # --------------------------------------------------------

        metrics_path = (
            os.path.join(
                METRIC_DIR,
                f"{slug}.json"
            )
        )


        with open(
            metrics_path,
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )


        print(
            "Metrics saved:",
            metrics_path
        )

        print()


        all_results.append(
            metrics
        )


        # --------------------------------------------------------
        # Free per-sample attention records before the next run --
        # these can be large (heads x image_tokens x text_length
        # per recorded layer x step).
        # --------------------------------------------------------

        del attention_records
        del concept_maps

        gc.collect()

        torch.cuda.empty_cache()


    # ============================================================
    # SAVE COMBINED SUMMARY
    # ============================================================

    combined_path = (
        os.path.join(
            METRIC_DIR,
            "all_metrics.json"
        )
    )


    with open(
        combined_path,
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4
        )


    print(
        "Combined metrics saved:",
        combined_path
    )


    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print()

    print(
        "=" * 70
    )

    print(
        "ALL SAMPLES COMPLETE"
    )

    print(
        "=" * 70
    )

    print()


    # ============================================================
    # CLEANUP
    # ============================================================

    del pipe

    gc.collect()

    torch.cuda.empty_cache()


    print(
        "Experiment complete."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()