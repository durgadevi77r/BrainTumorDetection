"""
config.py — PreprocessConfig: single source of truth for all pipeline parameters.

Every stage of the preprocessing pipeline reads its parameters from a
PreprocessConfig instance.  The module-level DEFAULT_CONFIG is built from
``settings`` and can be overridden per-request or per-experiment without
touching environment variables.

Usage
-----
    from app.preprocessing.config import PreprocessConfig, DEFAULT_CONFIG

    # Default config (reads from settings)
    cfg = DEFAULT_CONFIG

    # Override for an experiment
    cfg = PreprocessConfig(image_size=299, clahe_clip_limit=3.0)

    # Serialise / deserialise
    d = cfg.to_dict()
    cfg2 = PreprocessConfig.from_dict(d)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Tuple


# ─── ImageNet statistics (RGB order) ─────────────────────────────────────────
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class PreprocessConfig:
    """
    Fully configurable preprocessing pipeline parameters.

    All boolean flags can be toggled to compose exactly the transforms
    needed for each context (training / inference / Grad-CAM / preview).

    Parameters
    ----------
    image_size : int
        Target spatial dimension.  Images are resized to (image_size × image_size).
    image_channels : int
        Number of channels expected by the model (3 = RGB).

    Denoising
    ---------
    apply_denoise : bool
        Enable median-filter denoising.
    denoise_kernel_size : int
        Odd kernel size for the median filter (3 or 5).

    Contrast enhancement
    --------------------
    apply_clahe : bool
        Enable CLAHE contrast enhancement on the L channel.
    clahe_clip_limit : float
        CLAHE clip limit (higher → more contrast enhancement, more noise risk).
    clahe_tile_grid_size : tuple[int, int]
        Tile grid size for the CLAHE algorithm.

    Normalisation
    -------------
    normalise : bool
        Apply channel-wise z-score normalisation after rescaling to [0, 1].
    norm_mean : tuple[float, float, float]
        Per-channel mean (RGB order) for z-score normalisation.
    norm_std : tuple[float, float, float]
        Per-channel standard deviation (RGB order).

    Quality thresholds (used by quality.py)
    ----------------------------------------
    min_width : int
        Minimum acceptable image width in pixels.
    min_height : int
        Minimum acceptable image height in pixels.
    max_file_size_bytes : int
        Maximum acceptable raw file size (bytes).  0 = no limit.
    min_mean_intensity : float
        Minimum acceptable mean pixel intensity [0, 255].  Catches pure-black images.
    max_mean_intensity : float
        Maximum acceptable mean pixel intensity [0, 255].  Catches pure-white images.
    min_laplacian_variance : float
        Minimum variance of the Laplacian (blur detector).  Images below this
        threshold are flagged as too blurry.
    """

    # ── Spatial ───────────────────────────────────────────────────────────────
    image_size: int = 224
    image_channels: int = 3

    # ── Denoising ─────────────────────────────────────────────────────────────
    # IMPORTANT: training DataLoaders (MRIDataset + build_eval_transform) do
    # NOT apply any denoising or CLAHE.  Enabling these at inference would
    # shift the pixel distribution away from what the model was trained on and
    # cause wrong predictions.  Both flags default to False so that inference
    # matches the training pipeline exactly.
    apply_denoise: bool = False
    denoise_kernel_size: int = 3

    # ── Contrast enhancement ──────────────────────────────────────────────────
    # See note above — must stay False for inference to match training.
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # ── Normalisation ─────────────────────────────────────────────────────────
    # PyTorch models (MambaVision, EfficientNet, ResNet50, VGG16, CNN) are
    # trained with torchvision DataLoaders that apply:
    #   ToTensor() → [0, 1]  then  Normalize(IMAGENET_MEAN, IMAGENET_STD)
    #
    # TorchImageClassifier._to_tensor() reapplies the same ImageNet z-score
    # to the raw /255-rescaled NHWC array it receives from this pipeline.
    # Therefore inference normalisation must be:
    #   /255 rescale only (normalise=False) → let _to_tensor handle z-score.
    #
    # Do NOT set normalise=True here — that would double-normalise the input.
    normalise: bool = False          # /255 rescale only; z-score applied by TorchImageClassifier
    norm_mean: Tuple[float, float, float] = IMAGENET_MEAN  # kept for legacy shims
    norm_std:  Tuple[float, float, float] = IMAGENET_STD   # kept for legacy shims

    # ── Quality thresholds ────────────────────────────────────────────────────
    min_width: int = 32
    min_height: int = 32
    max_file_size_bytes: int = 20 * 1024 * 1024   # 20 MB
    min_mean_intensity: float = 5.0               # flags near-black images
    max_mean_intensity: float = 250.0             # flags near-white images
    min_laplacian_variance: float = 20.0          # flags severely blurry images

    # ── Accepted MIME types ───────────────────────────────────────────────────
    accepted_mime_types: List[str] = field(
        default_factory=lambda: ["image/jpeg", "image/png"]
    )
    accepted_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )

    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        # Convert tuples to lists for JSON compatibility
        d["clahe_tile_grid_size"] = list(self.clahe_tile_grid_size)
        d["norm_mean"]            = list(self.norm_mean)
        d["norm_std"]             = list(self.norm_std)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        """Deserialise from a plain dict."""
        data = dict(d)
        # Convert lists back to tuples for typed fields
        for key in ("clahe_tile_grid_size", "norm_mean", "norm_std"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def from_settings(cls) -> "PreprocessConfig":
        """Build a config from the app ``settings`` singleton."""
        from app.core.config import settings
        return cls(
            image_size=settings.image_size,
            image_channels=settings.image_channels,
        )


# ── Module-level default — built once at import time ─────────────────────────
# Import lazily to avoid circular dependency at module load time.
def _make_default() -> PreprocessConfig:
    try:
        return PreprocessConfig.from_settings()
    except Exception:
        return PreprocessConfig()


DEFAULT_CONFIG: PreprocessConfig = _make_default()
