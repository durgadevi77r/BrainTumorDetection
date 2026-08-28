"""
architectures.py — PyTorch model factory for all supported architectures.

Architectures
-------------
- mambavision : Official MambaVision-T (NVIDIA, ImageNet pretrained) with
                the classifier head replaced for 4 brain-tumor classes.
                Uses the official ``mambavision`` package + Hugging Face hub.
- cnn         : Lightweight custom CNN (fast baseline, CPU-friendly).
- vgg16       : torchvision VGG-16 with frozen backbone + custom head.
- resnet50    : torchvision ResNet-50 with frozen backbone + custom head.
- efficientnet: torchvision EfficientNet-B3 with frozen backbone + custom head.

All transfer-learning models use the same two-phase approach:
  Phase 1 — frozen backbone, train only the new classification head.
  Phase 2 — unfreeze the top N layers for fine-tuning (via
             ``unfreeze_top_layers()``).

Usage
-----
    from app.models.architectures import build_model
    model = build_model("mambavision")
    model = build_model("efficientnet")
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from transformers import AutoModelForImageClassification

from app.core.config import settings
from app.core.logging import logger
from app.models.mambavision.config import MambaVisionHFConfig
from app.models.mambavision.factory import build_mambavision_model


# ─── Shared classification head ───────────────────────────────────────────────

class _ClassificationHead(nn.Module):
    """
    Dense classification head appended to any backbone.

    Architecture:
        AdaptiveAvgPool2d → Flatten → BatchNorm1d → Linear(units, relu)
        → Dropout → Linear(num_classes)

    For MambaVision the backbone already ends with a pooled feature vector
    so the pool/flatten steps are skipped when ``needs_pool=False``.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        units: int = 256,
        dropout_rate: float = 0.5,
        needs_pool: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if needs_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten())
        layers.extend([
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(units, num_classes),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Custom CNN ───────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv→BN→ReLU × 2 → MaxPool → Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CustomCNN(nn.Module):
    """Lightweight 4-block CNN (32→64→128→256 filters)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBlock(in_channels, 32, dropout=0.25),
            _ConvBlock(32, 64, dropout=0.25),
            _ConvBlock(64, 128, dropout=0.30),
            _ConvBlock(128, 256, dropout=0.30),
        )
        self.head = _ClassificationHead(256, num_classes, units=512, dropout_rate=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


# ─── Transfer-learning wrappers ───────────────────────────────────────────────

class _TorchvisionModel(nn.Module):
    """
    Generic wrapper around a torchvision backbone that replaces the
    classifier head and exposes a ``freeze_backbone()`` / ``unfreeze_top()``
    interface consistent with the rest of the pipeline.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_features: int,
        num_classes: int,
        units: int = 256,
        dropout_rate: float = 0.5,
        name: str = "model",
    ) -> None:
        super().__init__()
        self.name_tag = name
        self.backbone = backbone
        self.head = _ClassificationHead(
            backbone_out_features, num_classes,
            units=units, dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        logger.debug(f"Backbone frozen for '{self.name_tag}'")

    def unfreeze_top(self, n_layers: int) -> None:
        """Unfreeze the last *n_layers* child modules of the backbone."""
        children = list(self.backbone.children())
        for child in children[-n_layers:]:
            for p in child.parameters():
                p.requires_grad_(True)
        logger.info(f"Unfrozen top {n_layers} modules of '{self.name_tag}'")


def _build_vgg16(num_classes: int) -> _TorchvisionModel:
    base = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    # Strip original classifier
    base.classifier = nn.Identity()
    # VGG-16 features output: (512, 7, 7) → AdaptiveAvgPool → 512
    model = _TorchvisionModel(base, 512, num_classes, units=256, name="vgg16")
    model.freeze_backbone()
    logger.debug(f"Built VGG-16 | classes={num_classes}")
    return model


def _build_resnet50(num_classes: int) -> _TorchvisionModel:
    base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = base.fc.in_features
    base.fc = nn.Identity()
    # ResNet already ends with AdaptiveAvgPool; head needs_pool=False
    base_out = in_features  # 2048
    model = _TorchvisionModel(
        base, base_out, num_classes,
        units=256, name="resnet50",
    )
    # Override head — ResNet returns a flat vector, no pool needed
    model.head = _ClassificationHead(
        base_out, num_classes, units=256, dropout_rate=0.5, needs_pool=False
    )
    model.freeze_backbone()
    logger.debug(f"Built ResNet-50 | classes={num_classes}")
    return model


def _build_efficientnet(num_classes: int) -> _TorchvisionModel:
    base = tv_models.efficientnet_b3(
        weights=tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )
    in_features = base.classifier[1].in_features
    base.classifier = nn.Identity()
    # EfficientNet ends with AdaptiveAvgPool → Flatten; head needs_pool=False
    model = _TorchvisionModel(
        base, in_features, num_classes,
        units=512, name="efficientnet",
    )
    model.head = _ClassificationHead(
        in_features, num_classes, units=512, dropout_rate=0.5, needs_pool=False
    )
    model.freeze_backbone()
    logger.debug(f"Built EfficientNet-B3 | classes={num_classes}")
    return model


# ─── MambaVision ──────────────────────────────────────────────────────────────

def _build_mambavision(num_classes: int) -> AutoModelForImageClassification:
    """
    Load official MambaVision-T pretrained on ImageNet-1K and resize the
    classifier head to ``num_classes`` using ignore_mismatched_sizes=True.

    Returns the raw Hugging Face model (not wrapped in _TorchvisionModel)
    because MambaVision already returns an object with a .logits attribute
    that TorchImageClassifier handles correctly.
    """
    cfg = MambaVisionHFConfig(num_classes=num_classes)
    model = build_mambavision_model(cfg=cfg, pretrained=True)
    logger.debug(f"Built MambaVision-T | classes={num_classes}")
    return model


# ─── Builders registry ────────────────────────────────────────────────────────

_BUILDERS = {
    "mambavision": _build_mambavision,
    "cnn":          lambda n: CustomCNN(num_classes=n),
    "vgg16":        _build_vgg16,
    "resnet50":     _build_resnet50,
    "efficientnet": _build_efficientnet,
}


# ─── Public factory ───────────────────────────────────────────────────────────

def build_model(
    model_name: Optional[str] = None,
    *,
    input_shape: Optional[Tuple[int, int, int]] = None,   # kept for compat; unused
    num_classes: Optional[int] = None,
    learning_rate: float = 1e-4,
) -> nn.Module:
    """
    Build a PyTorch model for brain-tumour classification.

    Parameters
    ----------
    model_name : str | None
        One of "mambavision" | "cnn" | "vgg16" | "resnet50" | "efficientnet".
        Defaults to ``settings.active_model``.
    input_shape : tuple | None
        Ignored (kept for backward-compatibility with old Keras callers).
        PyTorch models infer input shape from the first forward pass.
    num_classes : int | None
        Number of output classes — defaults to ``settings.num_classes``.
    learning_rate : float
        Not used here; kept for signature compatibility with old callers that
        passed ``learning_rate`` to ``build_model()``.  The actual optimiser
        is constructed by the training loop.

    Returns
    -------
    nn.Module
        Model in eval mode (training loop calls ``model.train()`` itself).

    Raises
    ------
    ValueError
        If ``model_name`` is not one of the supported architectures.
    """
    name  = (model_name or settings.active_model).lower()
    n_cls = num_classes or settings.num_classes

    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Choose one of: {list(_BUILDERS.keys())}"
        )

    model = _BUILDERS[name](n_cls)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model built | name={name} total_params={total_params:,} "
        f"trainable_params={trainable_params:,}"
    )
    return model


def unfreeze_top_layers(model: nn.Module, n_layers: int = 20) -> nn.Module:
    """
    Unfreeze the last ``n_layers`` backbone modules for Phase-2 fine-tuning.

    Works for ``_TorchvisionModel`` instances (vgg16/resnet50/efficientnet)
    and for HF MambaVision models.  For the custom CNN every layer is
    already trainable so this is a no-op.

    Parameters
    ----------
    model : nn.Module
        A model previously returned by ``build_model()``.
    n_layers : int
        Number of child modules from the end of the backbone to unfreeze.

    Returns
    -------
    nn.Module
        The same model with modified ``requires_grad`` flags (not copied).
    """
    if isinstance(model, _TorchvisionModel):
        model.unfreeze_top(n_layers)
        return model

    if isinstance(model, CustomCNN):
        # All parameters are already trainable
        logger.info("unfreeze_top_layers: CustomCNN is fully trainable — no-op")
        return model

    # HF / MambaVision — unfreeze the last n_layers backbone modules.
    #
    # MambaVisionModelForImageClassification has a single top-level child
    # ``model.model`` (the inner MambaVision object) which itself has these
    # children in order: [patch_embed, levels, norm, avgpool, head].
    #
    # ``levels`` is a ModuleList containing 4 level blocks, each of which
    # contains multiple transformer/SSM blocks.  To give n_layers meaningful
    # granularity we flatten the inner children of ``levels`` plus the other
    # top-level modules and unfreeze the last n.
    inner = getattr(model, "model", None)
    target = inner if (inner is not None and len(list(inner.children())) > 1) else model

    # Build a flat ordered list of unfreeze candidates
    flat_modules: list[nn.Module] = []
    for name, child in target.named_children():
        if name == "head":
            # head is already trainable — skip
            continue
        # Check if child is a ModuleList/Sequential worth expanding
        grandchildren = list(child.children())
        if grandchildren:
            flat_modules.extend(grandchildren)
        else:
            flat_modules.append(child)

    if not flat_modules:
        for p in model.parameters():
            p.requires_grad_(True)
        logger.info("unfreeze_top_layers: no backbone modules found — all unfrozen")
        return model

    n = min(n_layers, len(flat_modules))
    for mod in flat_modules[-n:]:
        for p in mod.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"unfreeze_top_layers: unfrozen top {n}/{len(flat_modules)} modules | "
        f"trainable_params={trainable:,}"
    )
    return model


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """
    Build an optimiser over a model's trainable parameters.

    Parameters
    ----------
    model : nn.Module
    learning_rate : float
    optimizer_name : str
        One of "adam" | "adamw" | "sgd" | "rmsprop".
    weight_decay : float
        L2 penalty (used by Adam/AdamW/SGD).
    momentum : float
        SGD momentum (ignored by Adam/AdamW/RMSProp).

    Returns
    -------
    optim.Optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]
    name   = optimizer_name.lower()

    if name == "adam":
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=learning_rate, momentum=momentum,
                         weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(
        f"Unknown optimizer '{optimizer_name}'. "
        "Choose one of: adam | adamw | sgd | rmsprop"
    )
