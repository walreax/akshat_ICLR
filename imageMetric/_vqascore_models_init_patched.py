# PATCHED by the imageMetric project: the original t2v_metrics package
# unconditionally imports 18 VQA model implementations here (LLaVA,
# InstructBLIP, GPT4V, Qwen2VL, Gemini, etc.), each dragging in its own
# heavy/fragile dependencies (lavis, omegaconf, xformers...) even though
# evaluate_models.py only ever uses CLIP_T5_MODELS ("clip-flant5-xl").
# Stripped down to just that path. Original backed up alongside this file
# as __init__.py.orig on the remote node.

from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model

from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [
    CLIP_T5_MODELS,
]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]

def get_vqascore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models()
    if model_name in CLIP_T5_MODELS:
        return CLIPT5Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()
