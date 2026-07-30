"""
training/__init__.py — Public surface of the training package.

The canonical entry points are:

    from training import Trainer, TrainingConfig, train
    from training import Experiment, ExperimentRegistry
    from training import build_callbacks, CallbackBundle
"""

from training.callbacks import build_callbacks, CallbackBundle, get_best_checkpoint_path
from training.checkpoints import (
    checkpoint_dir,
    delete_checkpoint,
    list_checkpoints,
    load_best_weights,
    load_checkpoint_info,
    save_checkpoint_info,
)
from training.config import DEFAULT_TRAINING_CONFIG, TrainingConfig
from training.experiment import Experiment, ExperimentRegistry
from training.trainer import Trainer, train

__all__ = [
    "Trainer",
    "train",
    "TrainingConfig",
    "DEFAULT_TRAINING_CONFIG",
    "Experiment",
    "ExperimentRegistry",
    "build_callbacks",
    "CallbackBundle",
    "get_best_checkpoint_path",
    "checkpoint_dir",
    "save_checkpoint_info",
    "load_best_weights",
    "load_checkpoint_info",
    "list_checkpoints",
    "delete_checkpoint",
]
