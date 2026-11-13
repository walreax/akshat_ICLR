# PATCHED by the imageMetric project: stripped to only CLIP_MODELS
# ("openai:ViT-L-14"), the one evaluate_models.py actually uses.
# BLIP2-ITC, HPSv2, PickScore, and the video CLIP variants each drag in
# their own heavy/fragile deps we never touch. Original backed up as
# __init__.py.orig on the remote node.

from .clip_model import CLIP_MODELS, CLIPScoreModel
from ...constants import HF_CACHE_DIR

ALL_CLIP_MODELS = [
    CLIP_MODELS,
]

def list_all_clipscore_models():
    return [model for models in ALL_CLIP_MODELS for model in models]

def get_clipscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_clipscore_models()
    if model_name in CLIP_MODELS:
        return CLIPScoreModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()
