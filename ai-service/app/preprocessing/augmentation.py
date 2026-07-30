"""
augmentation.py — PyTorch data augmentation for the training split only.

Augmentation is deliberately kept out of the validation, test, inference,
and Grad-CAM pipelines.  This module provides:

1. ``AugmentationConfig``         — typed dataclass controlling every parameter
                                    (defaults tuned for MRI brain-tumour data).
2. ``build_train_transform()``    — returns a torchvision Compose for training.
3. ``build_eval_transform()``     — returns a minimal Compose (resize + normalise)
                                    for validation/test/inference.
4. ``MRIDataset``                 — torch Dataset wrapping a class-folder directory.
5. ``build_data_generators_from_split()``
                                  — returns (train_loader, val_loader) from pre-split
                                    processed/ directories.
6. ``build_eval_datagen()``       — convenience alias returning an eval transform
                                    (backward-compat name used by preprocess.py).
7. ``apply_augmentation()``       — apply the augmentation stack to a single
                                    numpy uint8 RGB image (offline preview).

Why these augmentations for MRI?
---------------------------------
- Rotation (±15°)        Scanners may acquire at slightly different head angles.
- Width/height shift      Minor patient head positioning variation.
- Zoom (±10%)            Slight focal length and scan-plane differences.
- Horizontal flip         Left/right symmetry of the brain is medically valid.
- Color jitter brightness MRI signal intensity varies across acquisition protocols.
- NO vertical flip        Brain orientation is physically meaningful.
- NO aggressive colour    MRI channels carry the same signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from app.core.logging import logger
from app.preprocessing.config import IMAGENET_MEAN, IMAGENET_STD


# ── AugmentationConfig ────────────────────────────────────────────────────────

@dataclass
class AugmentationConfig:
    """
    Configures the torchvision augmentation pipeline.

    All parameters map to standard torchvision transform kwargs.
    Set a parameter to 0 / False to disable that augmentation.

    Parameters
    ----------
    rotation_range : float
        Degrees of random rotation (symmetric, e.g. 15 → ±15°).
    width_shift_range : float
        Fraction of total width for horizontal translation.
    height_shift_range : float
        Fraction of total height for vertical translation.
    shear_range : float
        Shear angle in degrees.
    zoom_range : float
        Zoom factor range.  [1-zoom, 1+zoom] applied randomly.
    horizontal_flip : bool
        Randomly flip images horizontally.
    vertical_flip : bool
        Randomly flip images vertically.  Must stay False for MRI anatomy.
    brightness_range : tuple[float, float] | None
        (min, max) brightness jitter factors.  None disables brightness aug.
    fill_mode : str
        Fill mode for affine transforms: 'nearest' | 'constant'.
        torchvision uses 0-fill for 'constant'.
    """

    rotation_range:        float = 15.0
    width_shift_range:     float = 0.08
    height_shift_range:    float = 0.08
    shear_range:           float = 5.0    # degrees (was radians in Keras; converted)
    zoom_range:            float = 0.10
    horizontal_flip:       bool  = True
    vertical_flip:         bool  = False  # must stay False for MRI anatomy
    brightness_range:      Optional[Tuple[float, float]] = (0.85, 1.15)
    fill_mode:             str   = "nearest"

    def to_dict(self) -> dict:
        return {
            "rotation_range":     self.rotation_range,
            "width_shift_range":  self.width_shift_range,
            "height_shift_range": self.height_shift_range,
            "shear_range":        self.shear_range,
            "zoom_range":         self.zoom_range,
            "horizontal_flip":    self.horizontal_flip,
            "vertical_flip":      self.vertical_flip,
            "brightness_range":   list(self.brightness_range) if self.brightness_range else None,
            "fill_mode":          self.fill_mode,
        }


# Module-level default
DEFAULT_AUG_CONFIG = AugmentationConfig()

# Fill value used for affine padding
_FILL_VALUE = 0


def _fill_for_config(cfg: AugmentationConfig) -> int:
    """torchvision uses an integer fill value; we only support constant=0."""
    return _FILL_VALUE


# ── Transform builders ────────────────────────────────────────────────────────

def build_train_transform(
    aug_cfg: Optional[AugmentationConfig] = None,
    image_size: int = 224,
) -> transforms.Compose:
    """
    Build a torchvision ``Compose`` with the full augmentation stack.

    The pipeline:
        1. Resize to (image_size, image_size)
        2. Random affine (rotation + shear + translation + zoom)
        3. Random horizontal flip (if enabled)
        4. Random vertical flip (if enabled — always False for MRI)
        5. Color jitter — brightness only
        6. ToTensor  (uint8 PIL → float32 [0, 1] NCHW)
        7. Normalize  (ImageNet mean / std)

    Parameters
    ----------
    aug_cfg : AugmentationConfig | None
        Augmentation parameters. Defaults to ``DEFAULT_AUG_CONFIG``.
    image_size : int
        Spatial dimension for the resize step.

    Returns
    -------
    transforms.Compose
    """
    cfg  = aug_cfg or DEFAULT_AUG_CONFIG
    fill = _fill_for_config(cfg)

    t: list = [transforms.Resize((image_size, image_size))]

    # Affine — rotation, shear, translation, scale
    affine_params: dict = {}
    if cfg.rotation_range > 0:
        affine_params["degrees"] = (-cfg.rotation_range, cfg.rotation_range)
    else:
        affine_params["degrees"] = 0

    if cfg.width_shift_range > 0 or cfg.height_shift_range > 0:
        affine_params["translate"] = (cfg.width_shift_range, cfg.height_shift_range)

    if cfg.shear_range > 0:
        affine_params["shear"] = (-cfg.shear_range, cfg.shear_range)

    if cfg.zoom_range > 0:
        scale_lo = max(1.0 - cfg.zoom_range, 0.1)
        scale_hi = 1.0 + cfg.zoom_range
        affine_params["scale"] = (scale_lo, scale_hi)

    if affine_params:
        affine_params.setdefault("degrees", 0)
        affine_params["fill"] = fill
        t.append(transforms.RandomAffine(**affine_params))

    if cfg.horizontal_flip:
        t.append(transforms.RandomHorizontalFlip(p=0.5))

    if cfg.vertical_flip:
        t.append(transforms.RandomVerticalFlip(p=0.5))

    if cfg.brightness_range is not None:
        lo, hi = cfg.brightness_range
        # torchvision ColorJitter brightness factor is symmetric around 1.0
        # so we compute the max deviation from 1.0
        brightness = max(abs(lo - 1.0), abs(hi - 1.0))
        if brightness > 0:
            t.append(transforms.ColorJitter(brightness=brightness))

    t.extend([
        transforms.ToTensor(),                                 # uint8 PIL → float32 [0,1]
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])

    logger.debug(f"Training transform built | aug_params={cfg.to_dict()}")
    return transforms.Compose(t)


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """
    Build a minimal torchvision ``Compose`` for validation / test / inference.

    No augmentation — only resize, ToTensor, and Normalize.

    Parameters
    ----------
    image_size : int
        Spatial dimension for the resize step.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


def build_eval_datagen(image_size: int = 224) -> transforms.Compose:
    """
    Convenience alias for ``build_eval_transform()``.

    Kept for backward compatibility with callers that used
    ``from app.preprocessing.augmentation import build_eval_datagen``.
    """
    return build_eval_transform(image_size=image_size)


# ── MRIDataset ────────────────────────────────────────────────────────────────

class MRIDataset(Dataset):
    """
    A torch ``Dataset`` that reads MRI images from a class-folder directory.

    Expected layout::

        root/
            glioma/        image1.jpg  image2.png  ...
            meningioma/    ...
            notumor/       ...
            pituitary/     ...

    The class-to-index mapping is built in alphabetical order (matching
    torchvision.datasets.ImageFolder behaviour).

    Parameters
    ----------
    root : str | Path
        Root directory containing one sub-folder per class.
    transform : callable | None
        A torchvision ``Compose`` (or any callable) applied to each PIL image.
    class_names : list[str] | None
        Canonical class ordering.  When provided, indices are assigned in
        list order; otherwise alphabetical order is used.
    extensions : tuple[str]
        File extensions considered as images.
    """

    EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    def __init__(
        self,
        root: str | Path,
        transform=None,
        class_names: Optional[List[str]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform
        self._extensions = extensions or self.EXTENSIONS

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Build class list
        subdirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if class_names:
            # Respect caller's canonical ordering; only include present dirs
            present = {p.name for p in subdirs}
            self.classes = [c for c in class_names if c in present]
        else:
            self.classes = [p.name for p in subdirs]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Build flat list of (path, label_index) pairs
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            idx = self.class_to_idx[cls]
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in self._extensions:
                    self.samples.append((p, idx))

        logger.debug(
            f"MRIDataset | root={self.root} classes={self.classes} "
            f"samples={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ── Keras DirectoryIterator compatibility surface ─────────────────────────

    @property
    def class_indices(self) -> dict:
        """Return class-to-index map (mirrors Keras DirectoryIterator.class_indices)."""
        return self.class_to_idx

    @property
    def num_classes(self) -> int:
        return len(self.classes)


# ── Generator / DataLoader builders ──────────────────────────────────────────

def build_data_generators_from_split(
    train_dir: str | Path,
    val_dir: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    aug_cfg: Optional[AugmentationConfig] = None,
    seed: int = 42,
    num_workers: int = 0,
    class_names: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build ``(train_loader, val_loader)`` from pre-split directories.

    This is the preferred approach when ``dataset/processed/train/`` and
    ``dataset/processed/val/`` have already been created by the splitter.
    Each directory must contain one sub-folder per class.

    Parameters
    ----------
    train_dir : str | Path
        Directory containing class sub-folders for training.
    val_dir : str | Path
        Directory containing class sub-folders for validation.
    image_size : int
        Target (H, W) in pixels.
    batch_size : int
        Mini-batch size.
    aug_cfg : AugmentationConfig | None
        Augmentation config for training. Defaults to ``DEFAULT_AUG_CONFIG``.
    seed : int
        Random seed for the training DataLoader sampler.
    num_workers : int
        Number of DataLoader worker processes.
    class_names : list[str] | None
        Canonical class ordering.  Passed to ``MRIDataset``.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    train_dir = Path(train_dir)
    val_dir   = Path(val_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_transform = build_train_transform(aug_cfg, image_size=image_size)
    val_transform   = build_eval_transform(image_size=image_size)

    train_ds = MRIDataset(train_dir, transform=train_transform, class_names=class_names)
    val_ds   = MRIDataset(val_dir,   transform=val_transform,   class_names=class_names)

    # Seeded generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(
        f"Split DataLoaders built | "
        f"train={len(train_ds)} val={len(val_ds)} "
        f"batch={batch_size} size={image_size}"
    )
    return train_loader, val_loader


# ── Offline augmentation preview ──────────────────────────────────────────────

def apply_augmentation(
    img_rgb: np.ndarray,
    aug_cfg: Optional[AugmentationConfig] = None,
    seed: int = 42,
    n_samples: int = 1,
    image_size: int = 224,
) -> list[np.ndarray]:
    """
    Apply augmentation to a single RGB uint8 image and return *n_samples* variants.

    Useful for offline preview / debugging and for generating augmented
    previews via the API.  The returned images are uint8 RGB (not normalised)
    so they can be base64-encoded and rendered in a browser.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB uint8 image (H, W, 3).
    aug_cfg : AugmentationConfig | None
        Augmentation parameters.
    seed : int
        Random seed.
    n_samples : int
        Number of augmented variants to return.
    image_size : int
        Target spatial dimension for the resize step.

    Returns
    -------
    list[np.ndarray]
        List of *n_samples* RGB uint8 augmented images.
    """
    cfg = aug_cfg or DEFAULT_AUG_CONFIG

    # Build a transform *without* the normalisation step so the result
    # stays in [0, 255] uint8 and can be displayed directly.
    fill = _fill_for_config(cfg)
    t: list = [transforms.Resize((image_size, image_size))]

    affine_params: dict = {}
    if cfg.rotation_range > 0:
        affine_params["degrees"] = (-cfg.rotation_range, cfg.rotation_range)
    else:
        affine_params["degrees"] = 0
    if cfg.width_shift_range > 0 or cfg.height_shift_range > 0:
        affine_params["translate"] = (cfg.width_shift_range, cfg.height_shift_range)
    if cfg.shear_range > 0:
        affine_params["shear"] = (-cfg.shear_range, cfg.shear_range)
    if cfg.zoom_range > 0:
        scale_lo = max(1.0 - cfg.zoom_range, 0.1)
        scale_hi = 1.0 + cfg.zoom_range
        affine_params["scale"] = (scale_lo, scale_hi)
    if affine_params:
        affine_params.setdefault("degrees", 0)
        affine_params["fill"] = fill
        t.append(transforms.RandomAffine(**affine_params))

    if cfg.horizontal_flip:
        t.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.vertical_flip:
        t.append(transforms.RandomVerticalFlip(p=0.5))
    if cfg.brightness_range is not None:
        lo, hi = cfg.brightness_range
        brightness = max(abs(lo - 1.0), abs(hi - 1.0))
        if brightness > 0:
            t.append(transforms.ColorJitter(brightness=brightness))

    preview_transform = transforms.Compose(t)
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))

    torch.manual_seed(seed)
    results: list[np.ndarray] = []
    for _ in range(n_samples):
        aug_pil = preview_transform(pil_img)
        results.append(np.array(aug_pil, dtype=np.uint8))

    return results
