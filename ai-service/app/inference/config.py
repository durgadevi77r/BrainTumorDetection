"""
app/inference/config.py — InferenceConfig: all settings for the inference pipeline.

Central dataclass that controls every aspect of inference behaviour:
  - model selection and version pinning
  - preprocessing overrides
  - top-K, confidence threshold
  - Grad-CAM settings
  - batch sizing and parallelism
  - output formats and paths

Usage
-----
    from app.inference.config import InferenceConfig, DEFAULT_INFERENCE_CONFIG

    cfg = InferenceConfig(model_name="resnet50", top_k=3)
    cfg = InferenceConfig.from_settings()   # read defaults from app.core.config
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── Supported values ─────────────────────────────────────────────────────────
SUPPORTED_ARCHITECTURES = ("mambavision", "cnn", "vgg16", "resnet50", "efficientnet")
SUPPORTED_OUTPUT_FORMATS = ("json", "csv", "both")


@dataclass
class InferenceConfig:
    """
    Complete configuration for one inference run.

    Model
    -----
    model_name : str
        Architecture to use.  Defaults to ``settings.active_model``.
    model_version : str | None
        Opaque version tag stored in model_info.json (informational only).

    Prediction
    ----------
    top_k : int
        Number of top predictions to include in results (1 ≤ k ≤ num_classes).
    confidence_threshold : float
        Minimum confidence to mark a prediction as "high confidence" (0–1).
    class_names : list[str]
        Ordered class labels.  Read from settings when None.
    image_size : int
        Input spatial resolution (H = W).

    Grad-CAM
    --------
    generate_gradcam : bool
        Whether to produce Grad-CAM heatmaps for each prediction.
    gradcam_alpha : float
        Heatmap blend factor (0 = original image, 1 = pure heatmap).
    gradcam_output_dir : str | None
        Override for Grad-CAM output directory.

    Batch / parallel
    ----------------
    batch_size : int
        Number of images to feed to model.predict() in one call.
    max_workers : int
        Thread-pool workers for parallel I/O preprocessing.
        Set to 1 to disable parallelism.
    timeout_s : float
        Per-image preprocessing timeout in seconds (0 = no timeout).

    Output
    ------
    output_format : str
        "json" | "csv" | "both"
    output_dir : str | None
        Directory for batch result exports.  Defaults to cwd.
    save_gradcam : bool
        Whether to persist Grad-CAM overlay images to disk.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name:    str           = "mambavision"
    model_version: Optional[str] = None

    # ── Prediction ────────────────────────────────────────────────────────────
    top_k:                int        = 1
    confidence_threshold: float      = 0.5
    class_names:          List[str]  = field(
        default_factory=lambda: ["glioma", "meningioma", "notumor", "pituitary"]
    )
    image_size: int = 224

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    generate_gradcam:  bool           = False
    gradcam_alpha:     float          = 0.4
    gradcam_output_dir: Optional[str] = None

    # ── Batch / parallel ──────────────────────────────────────────────────────
    batch_size:  int   = 16
    max_workers: int   = 4
    timeout_s:   float = 30.0

    # ── Output ────────────────────────────────────────────────────────────────
    output_format: str           = "json"
    output_dir:    Optional[str] = None
    save_gradcam:  bool          = True

    def __post_init__(self) -> None:
        arch = self.model_name.lower()
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"model_name must be one of {SUPPORTED_ARCHITECTURES}, got '{arch}'"
            )
        self.model_name = arch

        if not 1 <= self.top_k <= len(self.class_names):
            raise ValueError(
                f"top_k must be between 1 and {len(self.class_names)}, got {self.top_k}"
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )

        if self.output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {SUPPORTED_OUTPUT_FORMATS}, "
                f"got '{self.output_format}'"
            )

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        return Path.cwd() / "inference_output"

    @property
    def resolved_gradcam_dir(self) -> Path:
        if self.gradcam_output_dir:
            return Path(self.gradcam_output_dir)
        from app.core.config import settings
        return settings.gradcam_output_dir

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "InferenceConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_settings(cls) -> "InferenceConfig":
        """Create an InferenceConfig seeded from the app settings singleton."""
        from app.core.config import settings
        return cls(
            model_name=settings.active_model,
            class_names=settings.classes,
            image_size=settings.image_size,
        )


# ── Module-level default (lazy — avoids circular imports at top-level) ─────────
def _make_default() -> InferenceConfig:
    try:
        return InferenceConfig.from_settings()
    except Exception:
        return InferenceConfig()


DEFAULT_INFERENCE_CONFIG: InferenceConfig = _make_default()
