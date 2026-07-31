"""
gradcam.py — Grad-CAM heatmap generation for MambaVision (and other CNN/hybrid models).

Overview
--------
MambaVision (nvidia/MambaVision-T-1K) is a hybrid CNN-SSM architecture:

    patch_embed (Conv2d stem)
    └── levels[0,1]  — ConvBlock stages (Conv2d + BN)
    └── levels[2,3]  — Transformer/Mamba stages (window-attn + SSM blocks)
    └── norm          — BatchNorm2d(640)   ← BEST Grad-CAM hook point
    └── avgpool       — AdaptiveAvgPool2d(1)
    └── head          — Linear(640 → num_classes)

The ``norm`` layer (``model.model.norm``) is a BatchNorm2d that receives the
last spatial 4-D feature map ``(B, 640, H, W)`` just before average pooling.
Hooking here gives the richest spatial gradient signal for Grad-CAM.

Layer-selection strategy (``_find_gradcam_target``)
----------------------------------------------------
1.  MambaVision-specific: look for ``model.model.norm`` (``BatchNorm2d``).
2.  Generic fallback A: last ``BatchNorm2d`` in the entire model whose output
    has at least 2 spatial dimensions (i.e. it is a 2-D feature map).
3.  Generic fallback B: last ``Conv2d`` layer.
4.  If none found, raise ``ValueError``.

Outputs (per call to ``generate_gradcam``)
------------------------------------------
All artefacts are saved under ``settings.gradcam_mambavision_dir``
(``saved_models/mambavision/gradcam/``) in a per-image sub-directory:

    <gradcam_dir>/<image_id>/
        original.png        — pre-processed RGB source image (uint8)
        heatmap.png         — raw Grad-CAM heat-map (JET-coloured, uint8)
        overlay.png         — heatmap blended onto original (the main output)
        metadata.json       — prediction label, confidence, class index, paths,
                              target layer name, heatmap stats, image dimensions

Public API
----------
``generate_gradcam(source, model_name, class_index, image_id, alpha)``
    Main entry point.  Returns a dict with keys:
        gradcam_path, original_path, heatmap_path, metadata_path,
        class_index, class_name, confidence, image_id

``compute_gradcam_heatmap(wrapped_model, tensor_nhwc, class_index)``
    Pure computation — no I/O.  Returns ``np.ndarray`` float32 in [0, 1].

``overlay_heatmap(display_rgb, heatmap, alpha)``
    Blend heatmap onto RGB image.  Returns BGR uint8 (cv2 convention).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings
from app.core.logging import logger
from app.preprocessing.preprocess import preprocess_for_gradcam


# ─── Layer selection ──────────────────────────────────────────────────────────

def _find_gradcam_target(model: nn.Module) -> Tuple[str, nn.Module]:
    """
    Locate the best module to hook for Grad-CAM inside *model*.

    Search order
    ------------
    1. ``model.model.norm``  — the MambaVision-specific BatchNorm2d that sits
       just before average-pooling (the canonical choice for this architecture).
    2. Last ``BatchNorm2d`` at any depth whose output has spatial dimensions
       (4-D tensor: B × C × H × W).  This covers other CNN architectures.
    3. Last ``Conv2d`` layer (final fallback for any CNN backbone).

    Returns
    -------
    (name, module) tuple.

    Raises
    ------
    ValueError
        When no suitable layer is found.
    """
    # ── Strategy 1: MambaVision canonical target ──────────────────────────────
    # The HF wrapper is MambaVisionModelForImageClassification.
    # Its .model attribute is the raw MambaVision nn.Module.
    inner = getattr(model, "model", None)
    if inner is not None:
        norm = getattr(inner, "norm", None)
        if norm is not None and isinstance(norm, nn.BatchNorm2d):
            logger.debug("Grad-CAM target: model.model.norm (MambaVision canonical)")
            return "model.model.norm", norm

    # ── Strategy 2: last BatchNorm2d ──────────────────────────────────────────
    last_bn_name: Optional[str] = None
    last_bn_module: Optional[nn.Module] = None
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            last_bn_name = name
            last_bn_module = m
    if last_bn_name is not None:
        logger.debug(f"Grad-CAM target: last BatchNorm2d — {last_bn_name!r}")
        return last_bn_name, last_bn_module  # type: ignore[return-value]

    # ── Strategy 3: last Conv2d ───────────────────────────────────────────────
    last_conv_name: Optional[str] = None
    last_conv_module: Optional[nn.Module] = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv_name = name
            last_conv_module = m
    if last_conv_name is not None:
        logger.debug(f"Grad-CAM target: last Conv2d — {last_conv_name!r}")
        return last_conv_name, last_conv_module  # type: ignore[return-value]

    raise ValueError(
        "Could not locate a suitable Grad-CAM target layer. "
        "The model has no BatchNorm2d or Conv2d modules."
    )


# ─── Core Grad-CAM computation ────────────────────────────────────────────────

def compute_gradcam_heatmap(
    wrapped_model: Any,
    tensor_nhwc: np.ndarray,
    class_index: int,
) -> Tuple[np.ndarray, str]:
    """
    Compute a Grad-CAM heat-map for a single image.

    Parameters
    ----------
    wrapped_model : TorchImageClassifier
        Adapter from ``app.models.mambavision.predictor``.  Must expose:
        - ``.model``      — the raw ``nn.Module``
        - ``.device``     — ``torch.device``
        - ``.to_tensor()`` — NHWC float32 → NCHW normalised tensor
    tensor_nhwc : np.ndarray
        Float32 ``(1, H, W, C)`` batch in [0, 1], straight from
        ``preprocess_for_gradcam``.
    class_index : int
        Target class index whose score is backpropagated.

    Returns
    -------
    (heatmap, target_layer_name)
        heatmap : np.ndarray — float32 in [0, 1], shape ``(H', W')``
            where H', W' are the spatial dims of the feature map
            (not necessarily equal to the input resolution).
        target_layer_name : str

    Raises
    ------
    TypeError
        When ``wrapped_model`` is not a TorchImageClassifier-like adapter.
    RuntimeError
        When no activations or gradients are captured (e.g. graph issue).
    """
    if not hasattr(wrapped_model, "model") or not hasattr(wrapped_model, "to_tensor"):
        raise TypeError(
            "Grad-CAM requires a TorchImageClassifier adapter with "
            ".model and .to_tensor() attributes."
        )

    raw_model: nn.Module = wrapped_model.model
    device: torch.device = wrapped_model.device
    raw_model.eval()

    target_layer_name, target_layer = _find_gradcam_target(raw_model)

    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def _fwd_hook(_module: nn.Module, _inp: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor):
            activations["value"] = output.detach().clone()

    def _bwd_hook(_module: nn.Module, _grad_in: Any, grad_output: Any) -> None:
        if isinstance(grad_output, (tuple, list)) and len(grad_output) > 0:
            g = grad_output[0]
            if g is not None:
                gradients["value"] = g.detach().clone()

    h_fwd = target_layer.register_forward_hook(_fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        x = wrapped_model.to_tensor(tensor_nhwc).to(device)
        x.requires_grad_(True)

        raw_model.zero_grad(set_to_none=True)
        out = raw_model(x)

        # Both HF wrapper (returns dict / obj with .logits) and plain nn.Module
        if isinstance(out, dict):
            logits = out["logits"]
        elif hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, torch.Tensor):
            logits = out
        else:
            raise RuntimeError(
                f"Unexpected model output type: {type(out).__name__}. "
                "Expected dict, object with .logits, or plain Tensor."
            )

        if class_index >= logits.shape[-1]:
            raise ValueError(
                f"class_index={class_index} is out of range for "
                f"model with {logits.shape[-1]} output classes."
            )

        score = logits[0, class_index]
        score.backward()

        if "value" not in activations:
            raise RuntimeError(
                f"No activations captured from layer {target_layer_name!r}. "
                "The layer may not be reached during the forward pass."
            )
        if "value" not in gradients:
            raise RuntimeError(
                f"No gradients captured from layer {target_layer_name!r}. "
                "The layer may not be in the computational graph."
            )

        acts: torch.Tensor = activations["value"]   # (1, C, H, W)
        grads: torch.Tensor = gradients["value"]    # (1, C, H, W)

        if acts.ndim != 4:
            raise RuntimeError(
                f"Expected 4-D activations (B×C×H×W) from {target_layer_name!r}, "
                f"got shape {tuple(acts.shape)}."
            )

        # Global average pool the gradients → per-channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam: torch.Tensor = (weights * acts).sum(dim=1)  # (1, H, W)
        cam = torch.relu(cam)                             # ReLU removes negatives
        cam_np: np.ndarray = cam[0].cpu().numpy()        # (H, W)

        cam_max = float(cam_np.max())
        if cam_max > 1e-8:
            cam_np = cam_np / cam_max
        else:
            # Flat map — model is not confident; return uniform low-intensity map
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32), target_layer_name

    finally:
        h_fwd.remove()
        h_bwd.remove()


# ─── Overlay helper ───────────────────────────────────────────────────────────

def overlay_heatmap(
    display_rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize *heatmap* to match *display_rgb* and blend with JET colormap.

    Parameters
    ----------
    display_rgb : np.ndarray
        uint8 RGB image ``(H, W, 3)``.
    heatmap : np.ndarray
        float32 map in [0, 1], any spatial resolution.
    alpha : float
        Heatmap blending weight (0 = original image, 1 = pure heatmap).

    Returns
    -------
    (overlay_bgr, heatmap_bgr)
        Both are uint8 BGR images at the same resolution as *display_rgb*.
        BGR convention is used because cv2.imwrite expects BGR.
    """
    h, w = display_rgb.shape[:2]

    # Resize heat-map to match image resolution
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255.0 * np.clip(heatmap_resized, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)

    return overlay_bgr, heatmap_bgr


# ─── Public entry point ───────────────────────────────────────────────────────

def generate_gradcam(
    source: "str | bytes | Path",
    model_name: Optional[str] = None,
    class_index: Optional[int] = None,
    image_id: Optional[str] = None,
    *,
    alpha: float = 0.4,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate Grad-CAM artefacts for a single MRI image.

    Steps
    -----
    1. Load + preprocess the image via ``preprocess_for_gradcam``.
    2. Run ``model.predict()`` to determine the top-1 class (if *class_index*
       is not given).
    3. Compute the Grad-CAM heat-map via gradient backpropagation.
    4. Blend the heat-map onto the original image.
    5. Save original, heatmap, overlay PNGs and a JSON metadata file under:
           ``<output_dir>/<image_id>/``
       where *output_dir* defaults to ``settings.gradcam_mambavision_dir``
       for the MambaVision model, or ``settings.gradcam_output_dir`` for all
       other architectures.

    Parameters
    ----------
    source : str | bytes | Path
        Image file path or raw PNG/JPEG bytes.
    model_name : str | None
        Architecture key (e.g. ``"mambavision"``). Defaults to
        ``settings.active_model``.
    class_index : int | None
        Target class.  When *None* the top-1 prediction is used.
    image_id : str | None
        Caller-supplied identifier.  A UUID is generated when *None*.
    alpha : float
        Heatmap blend factor (0 = original, 1 = pure heatmap).
    output_dir : Path | None
        Override the output directory (mainly for testing).

    Returns
    -------
    dict with keys:
        ``gradcam_path``   — absolute path to overlay PNG
        ``original_path``  — absolute path to original PNG
        ``heatmap_path``   — absolute path to heatmap PNG
        ``metadata_path``  — absolute path to metadata JSON
        ``class_index``    — int
        ``class_name``     — str
        ``confidence``     — float (0–1)
        ``image_id``       — str
        ``target_layer``   — str (name of the hooked layer)

    Raises
    ------
    FileNotFoundError
        When ``source`` is a path that does not exist, or when no saved model
        weights are found for *model_name*.
    ValueError
        When the image cannot be decoded or class_index is out of range.
    IOError
        When cv2.imwrite fails to save an output file.
    """
    from app.models.load_model import load_model  # lazy — heavy import

    name = (model_name or settings.active_model).lower()
    img_id = image_id or str(uuid.uuid4())
    classes = settings.classes

    # ── Resolve output directory ──────────────────────────────────────────────
    if output_dir is None:
        if name == "mambavision":
            output_dir = settings.gradcam_mambavision_dir
        else:
            output_dir = settings.gradcam_output_dir

    img_output_dir = output_dir / img_id
    img_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    wrapped_model = load_model(name)

    # ── Preprocess image ──────────────────────────────────────────────────────
    tensor, display_rgb = preprocess_for_gradcam(source)
    # tensor: float32 (1, H, W, C) — for model.predict()
    # display_rgb: uint8 (H, W, C) — for overlay

    # ── Determine class ───────────────────────────────────────────────────────
    if class_index is None:
        probs: np.ndarray = wrapped_model.predict(tensor, verbose=0)
        class_index = int(np.argmax(probs[0]))
        confidence = float(probs[0, class_index])
    else:
        # Run predict anyway to get confidence for the requested class
        probs = wrapped_model.predict(tensor, verbose=0)
        if class_index < probs.shape[-1]:
            confidence = float(probs[0, class_index])
        else:
            confidence = 0.0

    class_name = classes[class_index] if class_index < len(classes) else str(class_index)

    # ── Compute Grad-CAM ──────────────────────────────────────────────────────
    heatmap, target_layer = compute_gradcam_heatmap(wrapped_model, tensor, class_index)

    # ── Generate overlay and heatmap images ───────────────────────────────────
    overlay_bgr, heatmap_bgr = overlay_heatmap(display_rgb, heatmap, alpha=alpha)

    # Original in BGR for cv2.imwrite
    original_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)

    # ── Save artefacts ────────────────────────────────────────────────────────
    original_path = img_output_dir / "original.png"
    heatmap_path  = img_output_dir / "heatmap.png"
    overlay_path  = img_output_dir / "overlay.png"
    meta_path     = img_output_dir / "metadata.json"

    for img_array, path in [
        (original_bgr, original_path),
        (heatmap_bgr,  heatmap_path),
        (overlay_bgr,  overlay_path),
    ]:
        ok = cv2.imwrite(str(path), img_array)
        if not ok:
            raise IOError(f"cv2.imwrite failed — could not save Grad-CAM artefact to {path}")

    # ── Build and save metadata ───────────────────────────────────────────────
    h, w = display_rgb.shape[:2]
    metadata: Dict[str, Any] = {
        "image_id":          img_id,
        "model_name":        name,
        "predicted_at":      datetime.now(timezone.utc).isoformat(),
        "class_index":       class_index,
        "class_name":        class_name,
        "confidence":        round(confidence, 6),
        "target_layer":      target_layer,
        "alpha":             alpha,
        "image_width":       w,
        "image_height":      h,
        "heatmap_min":       float(round(float(heatmap.min()), 6)),
        "heatmap_max":       float(round(float(heatmap.max()), 6)),
        "heatmap_mean":      float(round(float(heatmap.mean()), 6)),
        "original_path":     str(original_path.resolve()),
        "heatmap_path":      str(heatmap_path.resolve()),
        "gradcam_path":      str(overlay_path.resolve()),
        "metadata_path":     str(meta_path.resolve()),
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(
        f"Grad-CAM saved | image_id={img_id} class={class_name} "
        f"confidence={confidence:.4f} layer={target_layer!r} "
        f"dir={img_output_dir}"
    )

    return {
        "gradcam_path":   str(overlay_path.resolve()),
        "original_path":  str(original_path.resolve()),
        "heatmap_path":   str(heatmap_path.resolve()),
        "metadata_path":  str(meta_path.resolve()),
        "class_index":    class_index,
        "class_name":     class_name,
        "confidence":     round(confidence, 6),
        "image_id":       img_id,
        "target_layer":   target_layer,
    }
