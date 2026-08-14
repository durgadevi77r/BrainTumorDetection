"""
predict.py — Single-image inference pipeline.

Loads the requested model (cached after first call), runs the full
preprocessing stack, returns class label + confidence + per-class
probabilities, then triggers Grad-CAM generation.

Usage
-----
    from app.models.predict import predict
    result = predict(image_bytes, model_name="mambavision")
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.models.load_model import load_model
from app.preprocessing.preprocess import preprocess_for_inference


def predict(
    source: str | bytes | Path,
    model_name: Optional[str] = None,
    *,
    generate_gradcam: bool = True,
    image_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single MRI image.

    Parameters
    ----------
    source : str | bytes | Path
        File path *or* raw image bytes (JPEG / PNG).
    model_name : str | None
        Architecture key. Falls back to ``settings.active_model``.
    generate_gradcam : bool
        Whether to produce a Grad-CAM heatmap overlay.  Disabled
        automatically when the model weights have not been saved yet.
    image_id : str | None
        Identifier used to name the Grad-CAM output file.
        A UUID is generated when *None*.

    Returns
    -------
    dict
        {
          "class":         str,    # top-1 predicted label
          "confidence":    float,  # probability 0–1 (4 d.p.)
          "probabilities": {label: float, ...},  # all class scores
          "gradcam_path":  str | None,  # absolute path to overlay PNG
          "model_used":    str,
        }

    Raises
    ------
    FileNotFoundError
        When no saved weights are found for the requested model.
    ValueError
        When the image cannot be decoded.
    """
    name     = (model_name or settings.active_model).lower()
    img_id   = image_id or str(uuid.uuid4())
    classes  = settings.classes

    # ── Diagnostic: checkpoint metadata ──────────────────────────────────────
    from app.models.load_model import get_model_info
    model_info = get_model_info(name)
    logger.info(
        "[predict:diag] model=%s | checkpoint_classes=%s | "
        "checkpoint_num_classes=%s | runtime_classes=%s",
        name,
        model_info.get("class_names", "unknown"),
        model_info.get("num_classes", "unknown"),
        classes,
    )

    # ── Load model (cached) ───────────────────────────────────────────────────
    model = load_model(name)

    # ── Preprocess ────────────────────────────────────────────────────────────
    # preprocess_for_inference produces NHWC float32 in [0, 1] with
    # apply_denoise=False and apply_clahe=False (DEFAULT_CONFIG defaults),
    # matching the build_eval_transform used during training exactly.
    # TorchImageClassifier._to_tensor() then applies the ImageNet z-score.
    tensor = preprocess_for_inference(source, expand_dims=True)   # (1, H, W, C)

    # ── Diagnostic: input tensor stats ───────────────────────────────────────
    logger.info(
        "[predict:diag] image_id=%s | tensor_shape=%s | "
        "tensor_min=%.4f | tensor_max=%.4f | tensor_mean=%.4f",
        img_id,
        tensor.shape,
        float(tensor.min()),
        float(tensor.max()),
        float(tensor.mean()),
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    raw_preds: np.ndarray = model.predict(tensor, verbose=1)  # (1, num_classes)
    probs: np.ndarray     = raw_preds[0]                      # (num_classes,)

    top_idx    = int(np.argmax(probs))
    top_label  = classes[top_idx]
    confidence = round(float(probs[top_idx]), 4)

    probabilities = {
        label: round(float(probs[i]), 4)
        for i, label in enumerate(classes)
    }

    # ── Diagnostic: raw model output ─────────────────────────────────────────
    logger.info(
        "[predict:diag] image_id=%s | raw_probs=%s | "
        "top_idx=%d | top_label=%s | confidence=%.4f",
        img_id,
        [round(float(p), 4) for p in probs],
        top_idx,
        top_label,
        confidence,
    )

    logger.info(
        "Prediction | model=%s class=%s confidence=%.4f image_id=%s",
        name, top_label, confidence, img_id,
    )

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_path: Optional[str] = None
    if generate_gradcam:
        try:
            from app.utils.gradcam import generate_gradcam as _gradcam
            result = _gradcam(
                source,
                model_name=name,
                class_index=top_idx,
                image_id=img_id,
            )
            gradcam_path = result.get("gradcam_path")
        except Exception as exc:
            # Grad-CAM failure is non-fatal — log and continue
            logger.warning("Grad-CAM generation failed for %s: %s", img_id, exc)

    return {
        "class":         top_label,
        "confidence":    confidence,
        "probabilities": probabilities,
        "gradcam_path":  gradcam_path,
        "model_used":    name,
    }
