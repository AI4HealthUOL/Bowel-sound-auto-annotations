from __future__ import annotations

import json
import warnings
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file
from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from pipeline.config import Settings

# Optional ensemble import (kept exactly like your original logic)
try:
    from m4_ensemble import build_m4_from_local_checkpoints
except Exception:
    build_m4_from_local_checkpoints = None


def get_device() -> str:
    """Single point of truth for device selection."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_feature_extractor(model_dir: Optional[str]) -> ASTFeatureExtractor:
    """
    Load the AST feature extractor.
    If a directory load fails, fall back to the default extractor.
    """
    src = model_dir
    try:
        return ASTFeatureExtractor.from_pretrained(src)
    except Exception:
        warnings.warn(f"Could not load feature extractor from {src}; using default ASTFeatureExtractor().")
        return ASTFeatureExtractor()


def load_ast_model_single(settings: Settings) -> torch.nn.Module:
    """Load a single AST classifier from config.json + model.safetensors."""
    device = get_device()

    with open(settings.model_config, "r") as f:
        config_dict = json.load(f)
    config = ASTConfig.from_dict(config_dict)

    model = ASTForAudioClassification(config)
    state = load_file(settings.model_weights)
    model.load_state_dict(state)

    model.eval().to(device)
    return model


def load_model_according_to_mode(
    mode: str,
    settings: Settings,
    m1_dir: Optional[str] = None,
    m2_dir: Optional[str] = None,
    m3_dir: Optional[str] = None,
) -> Tuple[ASTFeatureExtractor, torch.nn.Module]:
    """
    Return (feature_extractor, model) based on the requested mode.

    Modes:
      - single : your original AST model (config + safetensors)
      - m4     : ensemble built from three local checkpoints (M1/M2/M3)

    Contract:
      - model(**inputs) returns an object with `.logits` OR a dict with "logits"
        (compatible with SequenceClassifierOutput).
    """
    device = get_device()

    if mode == "single":
        fe = load_feature_extractor(settings.model_dir)
        model = load_ast_model_single(settings)
        return fe, model

    if mode == "m4":
        if build_m4_from_local_checkpoints is None:
            raise RuntimeError(
                "m4_ensemble.py import failed. Ensure it's available and exports build_m4_from_local_checkpoints()."
            )
        if not (m1_dir and m2_dir and m3_dir):
            raise ValueError("--m1_dir, --m2_dir, and --m3_dir are required when --model_mode m4")

        # Assume same frontend; use feature extractor from M1.
        fe = load_feature_extractor(m1_dir)

        # Build ensemble (expected to output SequenceClassifierOutput-like object)
        m4 = build_m4_from_local_checkpoints(m1_dir, m2_dir, m3_dir, device=device)
        m4.eval()
        return fe, m4

    raise ValueError(f"Unknown model_mode: {mode}")
