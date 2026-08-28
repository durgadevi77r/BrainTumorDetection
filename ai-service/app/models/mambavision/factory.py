"""
factory.py — Official MambaVision model construction.

This file does not re-implement MambaVision. It only loads the official model
code and weights and adapts the classifier head for 4 brain tumor classes.

Head replacement note
---------------------
The nvidia/MambaVision-T-1K HF model uses a custom architecture class whose
head (`model.head`) is a plain `nn.Linear(640, 1000)`.  The standard HF
`ignore_mismatched_sizes=True` kwarg does not replace this head automatically
because it is not registered in the model's `_keys_to_ignore_on_load_*` lists.

We therefore:
1. Load the full pretrained backbone (1000-class head included).
2. Freeze all backbone parameters.
3. Replace `model.head` with a new 2-layer classification head sized to
   `cfg.num_classes` (4 for brain tumour classification).

This ensures:
- All backbone weights are the official NVIDIA ImageNet-1K pretrained weights.
- Only the new 4-class head is randomly initialised and trained from scratch
  in Phase 1.
- Phase 2 can optionally unfreeze top backbone layers for fine-tuning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification

from app.core.logging import logger
from app.models.mambavision.config import MambaVisionHFConfig


# Feature dimension of the MambaVision-T backbone output (before the head)
_MAMBAVISION_T_FEATURE_DIM = 640


def _replace_head(model: nn.Module, in_features: int, num_classes: int) -> nn.Module:
    """
    Replace model.head with a new Linear(in_features, num_classes).

    Using a single Linear layer (matching the original head architecture) so
    that save_pretrained / from_pretrained can load the weights back correctly.
    The old head weights are discarded; the new head is randomly initialised.
    """
    new_head = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(new_head.weight, std=0.02)
    if new_head.bias is not None:
        nn.init.zeros_(new_head.bias)
    model.head = new_head
    logger.info(
        f"[factory] Replaced model.head: Linear({in_features}, 1000) → "
        f"Linear({in_features}, {num_classes})"
    )
    return model


def _freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters except the classification head so Phase-1
    only trains the new head.

    For MambaVisionModelForImageClassification the structure is:
      model.model.head  ← the actual classifier head (model.model is MambaVision)

    Parameter names follow PyTorch's naming convention:
      "model.head.weight", "model.head.bias"  (for the inner model.head)

    We keep unfrozen anything whose name starts with "model.head." or "head."
    so this works regardless of whether the head is at the top level or nested.
    """
    frozen = 0
    for name, param in model.named_parameters():
        # Keep head trainable; freeze everything else
        is_head = name.startswith("head.") or name.startswith("model.head.")
        if not is_head:
            param.requires_grad_(False)
            frozen += param.numel()
        else:
            param.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[factory] Backbone frozen | frozen_params={frozen:,} "
        f"trainable_params={trainable:,}"
    )


def build_mambavision_model(
    *,
    cfg: Optional[MambaVisionHFConfig] = None,
    pretrained: bool = True,
) -> AutoModelForImageClassification:
    """
    Create a MambaVision image-classification model for transfer learning.

    Behavior
    --------
    - Loads official ImageNet-1K pretrained weights from Hugging Face hub
      (or the local HF cache).
    - Replaces the 1000-class head with a new `num_classes`-class head.
    - Freezes the backbone so Phase-1 training only updates the new head.

    Parameters
    ----------
    cfg : MambaVisionHFConfig | None
        Configuration.  Defaults to ``MambaVisionHFConfig()`` (4 classes).
    pretrained : bool
        Load pretrained backbone weights.  Set False only for unit tests.

    Returns
    -------
    AutoModelForImageClassification
        MambaVision-T with a 4-class head, backbone frozen.
    """
    cfg = cfg or MambaVisionHFConfig()

    if pretrained:
        logger.info(
            f"[factory] Loading MambaVision pretrained backbone from {cfg.repo_id} …"
        )
        # Load with the original 1000 classes — ignore_mismatched_sizes is
        # irrelevant here because we replace the head ourselves below.
        model = AutoModelForImageClassification.from_pretrained(
            cfg.repo_id,
            trust_remote_code=cfg.trust_remote_code,
        )
    else:
        base_config = AutoConfig.from_pretrained(
            cfg.repo_id,
            trust_remote_code=cfg.trust_remote_code,
        )
        model = AutoModelForImageClassification.from_config(
            base_config,
            trust_remote_code=cfg.trust_remote_code,
        )
        logger.info(
            f"[factory] Built MambaVision from config (no pretrained weights) | "
            f"repo={cfg.repo_id}"
        )

    # Locate the actual head used in forward().
    # MambaVisionModelForImageClassification wraps a MambaVision instance at
    # model.model (the inner MambaVision). The forward path is:
    #   model.forward() → model.model.forward() → model.model.head(features)
    # So the head we need to replace is model.model.head, NOT model.head.
    actual_model = getattr(model, "model", model)   # the inner MambaVision
    if hasattr(actual_model, "head") and isinstance(actual_model.head, nn.Linear):
        in_features = actual_model.head.in_features
        # Replace the correct head
        new_head = nn.Linear(in_features, cfg.num_classes)
        nn.init.trunc_normal_(new_head.weight, std=0.02)
        if new_head.bias is not None:
            nn.init.zeros_(new_head.bias)
        actual_model.head = new_head
        logger.info(
            f"[factory] Replaced model.model.head: "
            f"Linear({in_features}, 1000) → Linear({in_features}, {cfg.num_classes})"
        )
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model = _replace_head(model, in_features, cfg.num_classes)
    else:
        in_features = _MAMBAVISION_T_FEATURE_DIM
        logger.warning(
            f"[factory] Could not detect head; using default dim={in_features}"
        )
        actual_model = getattr(model, "model", model)
        new_head = nn.Linear(in_features, cfg.num_classes)
        if hasattr(actual_model, "head"):
            actual_model.head = new_head
        else:
            model.head = new_head

    _freeze_backbone(model)

    # Sanity check — verify the forward path produces num_classes outputs
    _actual = getattr(model, "model", model)
    final_head = getattr(_actual, "head", None) or getattr(model, "head", None)
    final_out = final_head.out_features if isinstance(final_head, nn.Linear) else -1
    if final_out != cfg.num_classes:
        raise RuntimeError(
            f"[factory] Head replacement failed: "
            f"expected {cfg.num_classes} outputs, got {final_out}"
        )

    logger.info(
        f"[factory] MambaVision-T ready | "
        f"head=Linear({in_features}, {cfg.num_classes}) | "
        f"backbone_frozen=True"
    )
    return model

