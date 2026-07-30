"""
gradcam.py — Grad-CAM heatmap generation and overlay (PyTorch).

The AI service previously implemented Grad-CAM using TensorFlow. After migrating
the model runtime to official MambaVision (PyTorch), Grad-CAM is computed using
PyTorch autograd on the last Conv2d feature map.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.core.logging import logger
from app.models.load_model import load_keras_model
from app.preprocessing.preprocess import preprocess_image_for_gradcam


def _find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    last = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last


def _compute_gradcam_heatmap(
    wrapped_model: Any,
    tensor_nhwc: np.ndarray,
    class_index: int,
) -> np.ndarray:
    if not hasattr(wrapped_model, "model") or not hasattr(wrapped_model, "to_tensor"):
        raise TypeError("Grad-CAM requires a TorchImageClassifier-like adapter.")

    model: torch.nn.Module = wrapped_model.model
    device: torch.device = wrapped_model.device
    model.eval()

    target_layer = _find_last_conv_module(model)

    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def _forward_hook(_, __, output):
        activations["value"] = output

    def _backward_hook(_, __, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(_forward_hook)
    h2 = target_layer.register_full_backward_hook(_backward_hook)

    try:
        x = wrapped_model.to_tensor(tensor_nhwc).to(device)
        x.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        score = logits[:, class_index]
        score.backward(torch.ones_like(score))

        acts = activations["value"]
        grads = gradients["value"]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)

        cam_np = cam[0].detach().cpu().numpy()
        cam_max = float(np.max(cam_np))
        if cam_max > 0:
            cam_np = cam_np / cam_max
        return cam_np.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()


def _overlay_heatmap(
    display_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    h, w = display_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, jet, alpha, 0)
    return overlay


def generate_gradcam(
    source: str | bytes | Path,
    model_name: Optional[str] = None,
    class_index: Optional[int] = None,
    image_id: Optional[str] = None,
    *,
    alpha: float = 0.4,
) -> Dict[str, Any]:
    name     = (model_name or settings.active_model).lower()
    img_id   = image_id or str(uuid.uuid4())
    classes  = settings.classes

    model = load_keras_model(name)

    tensor, display_rgb = preprocess_image_for_gradcam(source)

    if class_index is None:
        raw_preds: np.ndarray = model.predict(tensor, verbose=0)
        class_index = int(np.argmax(raw_preds[0]))

    class_name = classes[class_index] if class_index < len(classes) else str(class_index)

    heatmap = _compute_gradcam_heatmap(model, tensor, class_index)
    overlay_bgr = _overlay_heatmap(display_rgb, heatmap, alpha=alpha)

    output_dir: Path = settings.gradcam_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{img_id}.png"

    success = cv2.imwrite(str(output_path), overlay_bgr)
    if not success:
        raise IOError(f"cv2.imwrite failed — could not save Grad-CAM to {output_path}")

    logger.info(
        f"Grad-CAM saved | image_id={img_id} class={class_name} path={output_path}"
    )

    return {
        "gradcam_path": str(output_path),
        "class_index":  class_index,
        "class_name":   class_name,
        "image_id":     img_id,
    }

