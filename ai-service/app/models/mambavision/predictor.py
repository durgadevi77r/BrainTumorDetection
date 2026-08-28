"""
predictor.py — TorchImageClassifier adapter used by the existing inference code.

The current inference pipeline expects a loaded model object to expose:

    predict(np.ndarray, verbose=0) -> np.ndarray

This adapter wraps a PyTorch/Transformers model and keeps that contract intact
so the FastAPI endpoints and pipeline logic do not have to change their request/
response formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from app.core.logging import logger
from app.preprocessing.config import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class TorchPredictConfig:
    """
    Prediction-time tensor transform configuration.

    The preprocessing pipeline returns NHWC float32 in [0, 1]. Official ImageNet
    models typically expect normalization by mean/std in RGB order.
    """

    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


class TorchImageClassifier:
    """
    Adapter that exposes a Keras-like `.predict()` for PyTorch classifiers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: Optional[torch.device] = None,
        predict_cfg: Optional[TorchPredictConfig] = None,
    ) -> None:
        self.model = model
        self.device = device or self._resolve_device("auto")
        self.predict_cfg = predict_cfg or TorchPredictConfig()

        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(pref: str) -> torch.device:
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _to_tensor(self, batch: np.ndarray) -> torch.Tensor:
        if not isinstance(batch, np.ndarray):
            raise TypeError(f"predict expects np.ndarray, got {type(batch).__name__}")

        if batch.ndim != 4:
            raise ValueError(f"Expected NHWC batch tensor, got shape={batch.shape}")

        if batch.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got shape={batch.shape}")

        x = torch.from_numpy(batch).to(dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)

        mean = torch.tensor(self.predict_cfg.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.predict_cfg.std, dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x

    def to_tensor(self, batch: np.ndarray) -> torch.Tensor:
        return self._to_tensor(batch)

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        if verbose:
            logger.debug(f"[TorchImageClassifier] predict batch shape={batch.shape}")

        x = self._to_tensor(batch).to(self.device)

        # ── Diagnostic ───────────────────────────────────────────────────────
        logger.info(
            "[TorchImageClassifier:diag] "
            "input_nhwc_shape=%s input_min=%.4f input_max=%.4f input_mean=%.4f | "
            "tensor_nchw_shape=%s tensor_min=%.4f tensor_max=%.4f tensor_mean=%.4f | "
            "device=%s model=%s",
            batch.shape,
            float(batch.min()),
            float(batch.max()),
            float(batch.mean()),
            tuple(x.shape),
            float(x.min().item()),
            float(x.max().item()),
            float(x.mean().item()),
            self.device,
            type(self.model).__name__,
        )

        with torch.inference_mode():
            out = self.model(x)
            if isinstance(out, dict):
                logits = out["logits"]
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            probs = torch.softmax(logits, dim=-1)

        # ── Diagnostic: raw logits and probabilities ──────────────────────────
        logger.info(
            "[TorchImageClassifier:diag] "
            "raw_logits=%s | softmax_probs=%s",
            [round(v, 4) for v in logits[0].detach().cpu().tolist()],
            [round(v, 4) for v in probs[0].detach().cpu().tolist()],
        )

        return probs.detach().cpu().numpy().astype(np.float32)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

