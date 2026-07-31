"""
preprocess.py — Unified, context-aware image preprocessing pipeline.

This module is the single entry-point for all preprocessing in the project.
It delegates every low-level operation to the three sub-modules:

    config.py       — PreprocessConfig dataclass (all tunable parameters)
    transforms.py   — Pure stateless image functions (load, denoise, CLAHE, …)
    augmentation.py — torchvision transform builders + MRIDataset + DataLoader

Five public contexts
--------------------
``preprocess_for_inference``
    Single-image path used by ``predict.py``.
    Pipeline: load → spatial (denoise + CLAHE + resize) → RGB → normalise.
    Returns float32 (1, H, W, C) batch tensor.

``preprocess_for_gradcam``
    Same spatial pipeline as inference, but also returns the pre-normalisation
    display copy (uint8 RGB) that Grad-CAM overlays onto.
    Returns (tensor, display_rgb).

``preprocess_for_preview``
    Returns the processed uint8 RGB image without normalisation so it can be
    base64-encoded and sent back to the browser for visual inspection.

``build_generators``
    Builds PyTorch DataLoaders for the **pre-split** directory layout
    produced by the dataset module:
        processed/train/<class>/   → augmented training DataLoader
        processed/val/<class>/     → eval DataLoader (no augmentation)

``build_test_generator``
    Builds a non-shuffled eval DataLoader for the test split.

Backward-compatible shims
--------------------------
The old function names (``preprocess_image``, ``preprocess_image_for_gradcam``,
``build_data_generators``, ``normalize_image``, ``apply_median_filter``,
``apply_clahe``, ``resize_image``) are preserved as thin wrappers so that
the rest of the codebase (predict.py, gradcam.py, train.py, evaluate.py)
continues to work without any changes.

NOTE: ``build_data_generators`` (the legacy single-root + validation_split
variant) now builds a PyTorch DataLoader from the pre-split processed/
directory instead of using an in-memory random split.  Callers that relied
on ``validation_split`` should switch to ``build_generators`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.preprocessing.config import PreprocessConfig, DEFAULT_CONFIG
from app.preprocessing.transforms import (
    apply_spatial_pipeline,
    bgr_to_rgb,
    encode_image_base64,
    load_image_bgr,
    normalize_image as _normalize,
    apply_median_filter as _median,
    apply_clahe as _clahe,
    resize_image as _resize,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_spatial_pipeline(
    source: str | bytes | Path,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image and run the full spatial pipeline.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (img_rgb_uint8_resized,  img_bgr_uint8_resized)
        Both are (H, W, 3) uint8 at ``cfg.image_size × cfg.image_size``.
        ``img_rgb`` is the display-ready copy; use it for Grad-CAM overlay.
    """
    img_bgr = load_image_bgr(source)                    # load from path or bytes
    img_bgr = apply_spatial_pipeline(img_bgr, cfg)      # denoise → CLAHE → resize
    img_rgb = bgr_to_rgb(img_bgr)
    return img_rgb, img_bgr


# ─────────────────────────────────────────────────────────────────────────────
# 1. Inference
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_inference(
    source: str | bytes | Path,
    *,
    cfg: Optional[PreprocessConfig] = None,
    expand_dims: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image at inference time.

    Steps
    -----
    1. Load (path or bytes).
    2. Median-filter denoise  (if ``cfg.apply_denoise``).
    3. CLAHE contrast enhance (if ``cfg.apply_clahe``).
    4. Lanczos resize to ``cfg.image_size × cfg.image_size``.
    5. BGR → RGB.
    6. z-score normalise with ``cfg.norm_mean`` / ``cfg.norm_std``
       (or simple /255 rescale when ``cfg.normalise`` is False).
    7. Optionally add batch dimension.

    Parameters
    ----------
    source : str | bytes | Path
        File path or raw JPEG/PNG bytes.
    cfg : PreprocessConfig | None
        Pipeline config.  Uses ``DEFAULT_CONFIG`` when None.
    expand_dims : bool
        If True returns shape (1, H, W, C); if False returns (H, W, C).

    Returns
    -------
    np.ndarray
        float32 tensor ready for ``model.predict()``.

    Raises
    ------
    ValueError
        When the image cannot be decoded.
    """
    cfg = cfg or DEFAULT_CONFIG
    img_rgb, _ = _run_spatial_pipeline(source, cfg)

    arr: np.ndarray
    if cfg.normalise:
        arr = _normalize(img_rgb, mean=cfg.norm_mean, std=cfg.norm_std)
    else:
        arr = img_rgb.astype(np.float32) / 255.0

    if expand_dims:
        arr = np.expand_dims(arr, axis=0)

    logger.debug(f"preprocess_for_inference → shape={arr.shape} dtype={arr.dtype}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_gradcam(
    source: str | bytes | Path,
    *,
    cfg: Optional[PreprocessConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocessing for Grad-CAM — returns the model tensor AND the display image.

    The display image is used to overlay the Grad-CAM heatmap; it must be
    uint8 RGB at the same spatial resolution as the model input.

    Parameters
    ----------
    source : str | bytes | Path
        File path or raw JPEG/PNG bytes.
    cfg : PreprocessConfig | None
        Pipeline config.  Uses ``DEFAULT_CONFIG`` when None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        tensor      — float32 (1, H, W, C) normalised, for ``model.predict()``.
        display_rgb — uint8  (H, W, C) RGB, for heatmap overlay.
    """
    cfg = cfg or DEFAULT_CONFIG
    img_rgb, _ = _run_spatial_pipeline(source, cfg)

    # Keep a uint8 copy *before* normalisation for the overlay
    display_rgb: np.ndarray = img_rgb.copy()

    if cfg.normalise:
        arr = _normalize(img_rgb, mean=cfg.norm_mean, std=cfg.norm_std)
    else:
        arr = img_rgb.astype(np.float32) / 255.0

    tensor = np.expand_dims(arr, axis=0)

    logger.debug(
        f"preprocess_for_gradcam → tensor={tensor.shape} display={display_rgb.shape}"
    )
    return tensor, display_rgb


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preview (no normalisation — returns displayable uint8 RGB)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_preview(
    source: str | bytes | Path,
    *,
    cfg: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """
    Run the spatial pipeline and return a display-ready uint8 RGB image.

    No normalisation is applied so the result can be base64-encoded and
    sent back to the browser as-is.

    Parameters
    ----------
    source : str | bytes | Path
        File path or raw JPEG/PNG bytes.
    cfg : PreprocessConfig | None
        Pipeline config.

    Returns
    -------
    np.ndarray
        uint8 RGB (H, W, 3) at ``cfg.image_size × cfg.image_size``.
    """
    cfg = cfg or DEFAULT_CONFIG
    img_rgb, _ = _run_spatial_pipeline(source, cfg)
    return img_rgb


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training / validation DataLoaders (pre-split directories)
# ─────────────────────────────────────────────────────────────────────────────

def build_generators(
    processed_dir: str | Path,
    *,
    batch_size: int = 32,
    cfg: Optional[PreprocessConfig] = None,
    aug_cfg: Optional["AugmentationConfig"] = None,
    seed: int = 42,
    num_workers: int = 0,
) -> "Tuple[DataLoader, DataLoader]":
    """
    Build ``(train_loader, val_loader)`` from a **pre-split** processed directory.

    Expected layout (produced by ``app.dataset.splitter``)::

        processed_dir/
            train/ <class folders>
            val/   <class folders>
            test/  <class folders>

    The training DataLoader applies the full ``AugmentationConfig`` stack.
    The validation DataLoader uses resize + normalise only — no augmentation.

    Parameters
    ----------
    processed_dir : str | Path
        Root of the processed dataset.
    batch_size : int
        Mini-batch size.
    cfg : PreprocessConfig | None
        Pipeline config (used for ``image_size``).
    aug_cfg : AugmentationConfig | None
        Augmentation parameters for the training split.
    seed : int
        Random seed.
    num_workers : int
        DataLoader worker count.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    # Lazy imports: torch is only needed for training DataLoaders
    from app.preprocessing.augmentation import build_data_generators_from_split  # noqa: PLC0415

    cfg = cfg or DEFAULT_CONFIG
    processed_dir = Path(processed_dir)
    train_dir = processed_dir / "train"
    val_dir   = processed_dir / "val"

    return build_data_generators_from_split(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=cfg.image_size,
        batch_size=batch_size,
        aug_cfg=aug_cfg,
        seed=seed,
        num_workers=num_workers,
        class_names=settings.classes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Test DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_test_generator(
    test_dir: str | Path,
    *,
    cfg: Optional[PreprocessConfig] = None,
    target_size: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> "DataLoader":
    """
    Build a non-shuffled test DataLoader for evaluation.

    Accepts either:
    - A pre-split ``processed_dir/test/`` directory (preferred), or
    - Any directory containing one sub-folder per class.

    No augmentation — only resize + normalise.

    Parameters
    ----------
    test_dir : str | Path
        Root directory with one sub-folder per class.
    cfg : PreprocessConfig | None
        Pipeline config (used for image_size if target_size is None).
    target_size : int | None
        Explicit resize override (legacy kwarg — prefer cfg).
    batch_size : int
        Batch size.
    num_workers : int
        DataLoader worker count.

    Returns
    -------
    DataLoader
    """
    # Lazy imports: torch is only needed when building DataLoaders
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from app.preprocessing.augmentation import MRIDataset, build_eval_transform  # noqa: PLC0415

    cfg  = cfg or DEFAULT_CONFIG
    size = target_size or cfg.image_size
    test_dir = Path(test_dir)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    eval_transform = build_eval_transform(image_size=size)
    dataset = MRIDataset(
        test_dir,
        transform=eval_transform,
        class_names=settings.classes,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(
        f"Test DataLoader built | samples={len(dataset)} "
        f"batch={batch_size} size={size}"
    )
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible shims
# ─────────────────────────────────────────────────────────────────────────────
# These thin wrappers keep existing callers (predict.py, gradcam.py,
# train.py, evaluate.py) working without modification.

def preprocess_image(
    source: str | bytes | Path,
    *,
    target_size: Optional[int] = None,
    apply_denoise: bool = True,
    apply_contrast: bool = True,
    expand_dims: bool = True,
    cfg: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """
    Backward-compatible shim → ``preprocess_for_inference()``.
    """
    _cfg = PreprocessConfig(
        image_size=target_size or (cfg.image_size if cfg else DEFAULT_CONFIG.image_size),
        apply_denoise=apply_denoise,
        apply_clahe=apply_contrast,
        normalise=cfg.normalise if cfg else DEFAULT_CONFIG.normalise,
        norm_mean=cfg.norm_mean if cfg else DEFAULT_CONFIG.norm_mean,
        norm_std=cfg.norm_std   if cfg else DEFAULT_CONFIG.norm_std,
        clahe_clip_limit=cfg.clahe_clip_limit if cfg else DEFAULT_CONFIG.clahe_clip_limit,
        clahe_tile_grid_size=cfg.clahe_tile_grid_size if cfg else DEFAULT_CONFIG.clahe_tile_grid_size,
        denoise_kernel_size=cfg.denoise_kernel_size if cfg else DEFAULT_CONFIG.denoise_kernel_size,
    )
    return preprocess_for_inference(source, cfg=_cfg, expand_dims=expand_dims)


def preprocess_image_for_gradcam(
    source: str | bytes | Path,
    *,
    target_size: Optional[int] = None,
    apply_denoise: bool = True,
    apply_contrast: bool = True,
    cfg: Optional[PreprocessConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible shim → ``preprocess_for_gradcam()``."""
    _cfg = PreprocessConfig(
        image_size=target_size or (cfg.image_size if cfg else DEFAULT_CONFIG.image_size),
        apply_denoise=apply_denoise,
        apply_clahe=apply_contrast,
        normalise=cfg.normalise if cfg else DEFAULT_CONFIG.normalise,
        norm_mean=cfg.norm_mean if cfg else DEFAULT_CONFIG.norm_mean,
        norm_std=cfg.norm_std   if cfg else DEFAULT_CONFIG.norm_std,
        clahe_clip_limit=cfg.clahe_clip_limit if cfg else DEFAULT_CONFIG.clahe_clip_limit,
        clahe_tile_grid_size=cfg.clahe_tile_grid_size if cfg else DEFAULT_CONFIG.clahe_tile_grid_size,
        denoise_kernel_size=cfg.denoise_kernel_size if cfg else DEFAULT_CONFIG.denoise_kernel_size,
    )
    return preprocess_for_gradcam(source, cfg=_cfg)


def build_data_generators(
    dataset_dir: str | Path,
    *,
    target_size: Optional[int] = None,
    batch_size: int = 32,
    validation_split: float = 0.2,
    aug_cfg: Optional["AugmentationConfig"] = None,
    seed: int = 42,
    num_workers: int = 0,
) -> "Tuple[DataLoader, DataLoader]":
    """
    Backward-compatible shim → ``build_generators()``.

    The old signature accepted a single dataset_dir + validation_split.
    This shim delegates to the pre-split directory layout instead.
    If ``dataset_dir`` is a processed directory (has train/ and val/ sub-dirs),
    it is used directly.  Otherwise ``settings.dataset_processed_dir`` is used.

    Parameters
    ----------
    dataset_dir : str | Path
        Either a processed root (containing train/ val/) or any directory.
        If train/ does not exist inside it, falls back to
        ``settings.dataset_processed_dir``.
    target_size : int | None
        Resize override (legacy kwarg).
    batch_size : int
    validation_split : float
        Ignored — kept for signature compatibility.
    aug_cfg : AugmentationConfig | None
    seed : int
    num_workers : int

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    dataset_dir = Path(dataset_dir)
    train_dir   = dataset_dir / "train"

    # When the caller supplied an explicit path, honour it strictly.
    # Only fall back to the canonical processed dir when the caller passes
    # the canonical processed dir itself (or a subdirectory of it), i.e.
    # when no explicit override was intended.
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_dir}'. "
            "Ensure the processed dataset exists or run POST /api/v1/dataset/prepare first."
        )

    if not train_dir.exists():
        # If the caller explicitly specified a non-default path that has no
        # train/ sub-directory, raise immediately rather than falling back
        # silently — this is almost certainly a configuration error.
        canonical = settings.dataset_processed_dir
        if dataset_dir.resolve() != canonical.resolve():
            raise FileNotFoundError(
                f"Training directory not found: '{train_dir}'. "
                "The dataset root must contain a 'train/' sub-directory. "
                "Run POST /api/v1/dataset/prepare to create the split layout."
            )
        # Caller passed the canonical dir but it has no train/ yet — fall back
        # (will raise later inside build_data_generators_from_split)
        logger.warning(
            f"build_data_generators: '{dataset_dir}/train' not found — "
            f"falling back to '{canonical}'"
        )
        dataset_dir = canonical

    cfg = PreprocessConfig(
        image_size=target_size or DEFAULT_CONFIG.image_size,
    ) if target_size else DEFAULT_CONFIG

    return build_generators(
        dataset_dir,
        batch_size=batch_size,
        cfg=cfg,
        aug_cfg=aug_cfg,
        seed=seed,
        num_workers=num_workers,
    )


# ── Expose legacy low-level shims used elsewhere ──────────────────────────────
normalize_image       = _normalize
apply_median_filter   = _median
apply_clahe_transform = _clahe   # renamed to avoid shadowing cv2 clahe
apply_clahe           = _clahe   # backward-compat alias
resize_image          = _resize
