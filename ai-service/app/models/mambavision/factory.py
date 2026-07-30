"""
factory.py — Official MambaVision model construction.

This file does not re-implement MambaVision. It only loads the official model
code and weights and adapts the classifier head for 4 brain tumor classes.
"""

from __future__ import annotations

from typing import Optional

from transformers import AutoConfig, AutoModelForImageClassification

from app.core.logging import logger
from app.models.mambavision.config import MambaVisionHFConfig


def build_mambavision_model(
    *,
    cfg: Optional[MambaVisionHFConfig] = None,
    pretrained: bool = True,
) -> AutoModelForImageClassification:
    """
    Create a MambaVision image-classification model for transfer learning.

    Behavior
    --------
    - When `pretrained=True`, loads official ImageNet pretrained weights.
    - The classifier head is resized to `cfg.num_classes` (4) using
      `ignore_mismatched_sizes=True`, which is the standard HF transfer-learning
      approach for classification heads.
    """

    cfg = cfg or MambaVisionHFConfig()

    if pretrained:
        logger.info(f"Loading official MambaVision pretrained weights from {cfg.repo_id} …")
        model = AutoModelForImageClassification.from_pretrained(
            cfg.repo_id,
            trust_remote_code=cfg.trust_remote_code,
            num_labels=cfg.num_classes,
            ignore_mismatched_sizes=True,
        )
        return model

    base_config = AutoConfig.from_pretrained(
        cfg.repo_id,
        trust_remote_code=cfg.trust_remote_code,
    )
    base_config.num_labels = cfg.num_classes
    model = AutoModelForImageClassification.from_config(
        base_config,
        trust_remote_code=cfg.trust_remote_code,
    )
    logger.info(f"Built MambaVision from config (no pretrained weights) | repo={cfg.repo_id}")
    return model

