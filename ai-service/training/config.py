"""
training/config.py — TrainingConfig: single source of truth for every
hyperparameter and path used by the training pipeline.

The dataclass is fully serialisable to / from JSON so that every
experiment stores an exact copy of the configuration that produced it.

Usage
-----
    from training.config import TrainingConfig, DEFAULT_TRAINING_CONFIG

    # Defaults (reads image_size and classes from app.core.config.settings)
    cfg = TrainingConfig()

    # Custom experiment
    cfg = TrainingConfig(
        architecture="mambavision",
        epochs=30,
        learning_rate=5e-5,
    )

    # Round-trip through JSON
    d    = cfg.to_dict()
    cfg2 = TrainingConfig.from_dict(d)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ─── Supported architectures ──────────────────────────────────────────────────
SUPPORTED_ARCHITECTURES = ("mambavision", "cnn", "vgg16", "resnet50", "efficientnet")

# ─── Supported optimisers ─────────────────────────────────────────────────────
SUPPORTED_OPTIMISERS = ("adam", "sgd", "rmsprop", "adamw")


@dataclass
class TrainingConfig:
    """
    Complete configuration for one training run.

    Model
    -----
    architecture : str
        One of "mambavision" | "cnn" | "vgg16" | "resnet50" | "efficientnet".
    image_size : int
        Spatial resolution fed to the model (H = W).
    num_classes : int
        Number of output classes (derived from ``settings.num_classes``).
    class_names : list[str]
        Ordered class labels (derived from ``settings.classes``).

    Optimiser
    ---------
    optimiser : str
        One of "adam" | "sgd" | "rmsprop" | "adamw".
    learning_rate : float
        Phase-1 learning rate.
    momentum : float
        Momentum for SGD (ignored by Adam / RMSProp).
    weight_decay : float
        L2 weight-decay coefficient.

    Regularisation
    --------------
    dropout_rate : float
        Dropout rate applied in the classification head (0.0 – 1.0).
    l2_reg : float
        L2 regularisation (kept for compat; applied via weight_decay).

    Training loop
    -------------
    epochs : int
        Maximum Phase-1 epochs.
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed for reproducibility.
    class_weights : dict[str, float] | None
        Per-class weight map keyed by class name.
        ``None`` means uniform weighting.
    num_workers : int
        DataLoader worker processes.

    Transfer learning / fine-tuning
    --------------------------------
    freeze_backbone : bool
        Freeze the backbone during Phase 1.
    fine_tune : bool
        Run Phase-2 after Phase-1 converges.
    fine_tune_layers : int
        Number of backbone modules to unfreeze in Phase 2.
    fine_tune_epochs : int
        Max epochs for Phase 2.
    fine_tune_lr : float | None
        Phase-2 learning rate.  Defaults to ``learning_rate / 10``.

    Callbacks / scheduling
    ----------------------
    early_stopping_patience : int
        EarlyStopping patience (epochs without improvement).
    early_stopping_monitor : str
        Metric to monitor (currently only "val_loss" is used).
    reduce_lr_patience : int
        ReduceLROnPlateau patience.
    reduce_lr_factor : float
        LR reduction factor.
    reduce_lr_min : float
        Minimum LR floor.
    scheduler : str
        LR scheduler type: "reduce_on_plateau" | "cosine".
    cosine_t_max : int
        CosineAnnealingLR period in epochs (0 = auto-set to cfg.epochs).
    cosine_eta_min : float
        CosineAnnealingLR minimum LR.
    csv_log : bool
        Write per-epoch CSV logs to disk.
    use_amp : bool
        Enable mixed-precision training (CUDA only; auto-disabled on CPU).
    grad_clip_max_norm : float
        Maximum gradient norm for gradient clipping (0.0 = disabled).

    Paths
    -----
    dataset_dir : str | None
        Root of the processed (split) dataset.
        Defaults to ``settings.dataset_processed_dir``.
    output_dir : str | None
        Root for saved models and checkpoints.
        Defaults to ``settings.saved_models_dir``.
    """

    # ── Model ──────────────────────────────────────────────────────────────────
    architecture: str       = "mambavision"
    image_size:   int       = 224
    num_classes:  int       = 4
    class_names:  List[str] = field(
        default_factory=lambda: ["glioma", "meningioma", "notumor", "pituitary"]
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser:     str   = "adam"
    learning_rate: float = 1e-4
    momentum:      float = 0.9
    weight_decay:  float = 1e-4

    # ── Regularisation ────────────────────────────────────────────────────────
    dropout_rate: float = 0.5
    l2_reg:       float = 1e-4

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs:        int  = 30
    batch_size:    int  = 32
    seed:          int  = 42
    num_workers:   int  = 0
    class_weights: Optional[Dict[str, float]] = None

    # ── Transfer learning / fine-tuning ───────────────────────────────────────
    freeze_backbone:  bool            = True
    fine_tune:        bool            = True
    fine_tune_layers: int             = 20
    fine_tune_epochs: int             = 10
    fine_tune_lr:     Optional[float] = None   # None → learning_rate / 10

    # ── Callbacks / scheduling ────────────────────────────────────────────────
    early_stopping_patience: int   = 10
    early_stopping_monitor:  str   = "val_loss"
    reduce_lr_patience:      int   = 5
    reduce_lr_factor:        float = 0.5
    reduce_lr_min:           float = 1e-7
    # Scheduler type: "reduce_on_plateau" | "cosine"
    scheduler:               str   = "reduce_on_plateau"
    # Cosine annealing: minimum LR at the bottom of each cycle
    cosine_t_max:            int   = 0      # 0 → auto-set to cfg.epochs
    cosine_eta_min:          float = 1e-7
    csv_log:                 bool  = True

    # ── AMP / gradient clipping ───────────────────────────────────────────────
    # Mixed precision training (CUDA only; silently disabled on CPU).
    use_amp:              bool  = True
    # Max gradient norm for gradient clipping (0.0 = disabled).
    grad_clip_max_norm:   float = 1.0

    # ── Paths (None → settings defaults) ─────────────────────────────────────
    dataset_dir: Optional[str] = None
    output_dir:  Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        arch = self.architecture.lower()
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {SUPPORTED_ARCHITECTURES}, got '{arch}'"
            )
        self.architecture = arch

        opt = self.optimiser.lower()
        if opt not in SUPPORTED_OPTIMISERS:
            raise ValueError(
                f"optimiser must be one of {SUPPORTED_OPTIMISERS}, got '{opt}'"
            )
        self.optimiser = opt

        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        valid_schedulers = ("reduce_on_plateau", "cosine")
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"scheduler must be one of {valid_schedulers}, got '{self.scheduler}'"
            )

        if self.grad_clip_max_norm < 0.0:
            raise ValueError(
                f"grad_clip_max_norm must be >= 0.0 (0 = disabled), "
                f"got {self.grad_clip_max_norm}"
            )

    @property
    def effective_fine_tune_lr(self) -> float:
        """Phase-2 LR (defaults to Phase-1 LR ÷ 10)."""
        return self.fine_tune_lr if self.fine_tune_lr is not None else self.learning_rate / 10

    @property
    def effective_cosine_t_max(self) -> int:
        """CosineAnnealingLR T_max — defaults to cfg.epochs when set to 0."""
        return self.cosine_t_max if self.cosine_t_max > 0 else self.epochs

    @property
    def class_weight_map(self) -> Optional[Dict[int, float]]:
        """
        Convert {class_name: weight} → {class_index: weight}.

        Returns None when class_weights is not set.
        """
        if not self.class_weights:
            return None
        return {
            self.class_names.index(cls): w
            for cls, w in self.class_weights.items()
            if cls in self.class_names
        }

    @property
    def resolved_dataset_dir(self) -> Path:
        if self.dataset_dir:
            return Path(self.dataset_dir)
        from app.core.config import settings
        return settings.dataset_processed_dir

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        from app.core.config import settings
        return settings.saved_models_dir

    def to_dict(self) -> dict:
        """Serialise to a plain JSON-safe dict."""
        d = asdict(self)
        if d.get("dataset_dir") and isinstance(d["dataset_dir"], Path):
            d["dataset_dir"] = str(d["dataset_dir"])
        if d.get("output_dir") and isinstance(d["output_dir"], Path):
            d["output_dir"] = str(d["output_dir"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Deserialise from a plain dict (e.g. loaded from JSON)."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_settings(cls) -> "TrainingConfig":
        """Create a config seeded from the app ``settings`` singleton."""
        from app.core.config import settings
        return cls(
            image_size=settings.image_size,
            num_classes=settings.num_classes,
            class_names=settings.classes,
            architecture=settings.active_model,
        )


# ── Module-level default ──────────────────────────────────────────────────────
def _make_default() -> TrainingConfig:
    try:
        return TrainingConfig.from_settings()
    except Exception:
        return TrainingConfig()


DEFAULT_TRAINING_CONFIG: TrainingConfig = _make_default()
