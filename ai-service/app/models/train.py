"""
train.py — Full PyTorch model training pipeline.

Two-phase transfer learning strategy
--------------------------------------
Phase 1  Frozen backbone, train only the classification head for ``epochs``
         epochs with ``learning_rate``.  Early-stops on val_loss plateau.
Phase 2  Unfreeze the top ``fine_tune_layers`` modules of the backbone,
         continue for ``fine_tune_epochs`` at ``fine_tune_lr`` (default: lr/10).

Usage
-----
    from app.models.train import train_model
    result = train_model("mambavision", epochs=30, batch_size=16)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.logging import logger
from app.models.architectures import build_model, build_optimizer, unfreeze_top_layers
from app.models.save_model import save_best_checkpoint, load_best_checkpoint, save_keras_model
from app.preprocessing.preprocess import build_data_generators


# ─── Device helper ────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── History helpers ──────────────────────────────────────────────────────────

def _extract_final_metrics(history: Dict[str, List[float]]) -> Dict[str, float]:
    """Pull the last-epoch metrics out of a history dict."""
    return {
        "final_train_loss":     float(history.get("loss", [0.0])[-1]),
        "final_train_accuracy": float(history.get("accuracy", [0.0])[-1]),
        "final_val_loss":       float(history.get("val_loss", [0.0])[-1]),
        "final_val_accuracy":   float(history.get("val_accuracy", [0.0])[-1]),
        "epochs_run":           len(history.get("loss", [])),
    }


# ─── Per-epoch train / eval ───────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    training: bool,
) -> tuple[float, float]:
    """
    Run one epoch of training or evaluation.

    Returns
    -------
    tuple[float, float]
        (mean_loss, accuracy)
    """
    model.train(training)
    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            # HF models return an object with .logits; raw models return a tensor
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)

            if training and optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    mean_loss = total_loss / max(total, 1)
    accuracy  = correct / max(total, 1)
    return mean_loss, accuracy


# ─── Public training entry point ─────────────────────────────────────────────

def train_model(
    model_name: str = "mambavision",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    dataset_dir: Optional[str] = None,
    *,
    validation_split: float = 0.2,          # kept for signature compat; unused
    fine_tune: bool = True,
    fine_tune_layers: int = 20,
    fine_tune_epochs: int = 10,
    fine_tune_lr: Optional[float] = None,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    reduce_lr_min: float = 1e-7,
    class_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Train a deep learning model on MRI brain-tumour images.

    Parameters
    ----------
    model_name : str
        Architecture — "mambavision" | "cnn" | "vgg16" | "resnet50" | "efficientnet".
    epochs : int
        Max Phase-1 epochs (early-stopping may cut short).
    batch_size : int
        Mini-batch size for both train and validation loaders.
    learning_rate : float
        Phase-1 learning rate.
    dataset_dir : str | None
        Root of the processed (pre-split) dataset.
        Falls back to ``settings.dataset_processed_dir``.
    validation_split : float
        Legacy kwarg — ignored (use pre-split directories).
    fine_tune : bool
        Whether to run Phase-2 fine-tuning after Phase-1 converges.
        Skipped for the custom CNN (no frozen backbone to unfreeze).
    fine_tune_layers : int
        Number of backbone modules to unfreeze in Phase 2.
    fine_tune_epochs : int
        Additional epochs for Phase 2.
    fine_tune_lr : float | None
        Phase-2 learning rate. Defaults to ``learning_rate / 10``.
    optimizer_name : str
        Optimiser name: "adam" | "adamw" | "sgd" | "rmsprop".
    weight_decay : float
        L2 weight-decay coefficient.
    early_stopping_patience : int
        Epochs without val_loss improvement before stopping.
    reduce_lr_patience : int
        Epochs without improvement before halving the LR.
    reduce_lr_factor : float
        LR reduction factor for ReduceLROnPlateau.
    reduce_lr_min : float
        Minimum LR floor.
    class_weights : dict[str, float] | None
        Per-class loss weights keyed by class name (e.g. {"glioma": 1.5}).
    seed : int
        Random seed for reproducibility.
    num_workers : int
        DataLoader worker count.

    Returns
    -------
    dict
        {
          "model_name":           str,
          "epochs_phase1":        int,
          "epochs_phase2":        int,
          "final_train_accuracy": float,
          "final_val_accuracy":   float,
          "final_train_loss":     float,
          "final_val_loss":       float,
          "training_duration_s":  float,
          "saved_paths":          dict,
          "phase1_history":       dict,
          "phase2_history":       dict,
        }

    Raises
    ------
    FileNotFoundError
        When the dataset directory does not exist.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    name    = model_name.lower()
    ft_lr   = fine_tune_lr or (learning_rate / 10)
    device  = _resolve_device()
    classes = settings.classes

    data_dir = Path(dataset_dir) if dataset_dir else settings.dataset_processed_dir
    logger.info(
        f"Training started | model={name} epochs={epochs} "
        f"batch={batch_size} lr={learning_rate} dataset={data_dir} device={device}"
    )

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader, val_loader = build_data_generators(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Class-weighted loss ───────────────────────────────────────────────────
    weight_tensor: Optional[torch.Tensor] = None
    if class_weights:
        weights = [
            float(class_weights.get(cls, 1.0))
            for cls in classes
        ]
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        logger.info(f"Class weights applied: {dict(zip(classes, weights))}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(name)
    model.to(device)

    # ── Phase 1 — train classification head only ──────────────────────────────
    t0 = time.perf_counter()
    logger.info(f"Phase 1: training head | max_epochs={epochs}")

    optimizer_p1 = build_optimizer(
        model, learning_rate, optimizer_name=optimizer_name, weight_decay=weight_decay
    )
    scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1,
        mode="min",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=reduce_lr_min,
    )

    history_p1, phase1_epochs = _train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        max_epochs=epochs,
        patience=early_stopping_patience,
        device=device,
        model_name=name,
        phase=1,
    )

    metrics_p1 = _extract_final_metrics(history_p1)
    logger.info(
        f"Phase 1 complete | "
        f"val_acc={metrics_p1['final_val_accuracy']:.4f} "
        f"epochs={phase1_epochs}"
    )

    # ── Restore best weights from Phase 1 ────────────────────────────────────
    load_best_checkpoint(model, name, device=device)

    # ── Phase 2 — fine-tune top backbone layers ───────────────────────────────
    history_p2: Dict[str, Any] = {}
    phase2_epochs = 0

    if fine_tune and name != "cnn":
        logger.info(
            f"Phase 2: fine-tuning top {fine_tune_layers} modules | "
            f"max_epochs={fine_tune_epochs} lr={ft_lr}"
        )

        model = unfreeze_top_layers(model, n_layers=fine_tune_layers)

        optimizer_p2 = build_optimizer(
            model, ft_lr, optimizer_name=optimizer_name, weight_decay=weight_decay
        )
        scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_p2,
            mode="min",
            factor=reduce_lr_factor,
            patience=max(reduce_lr_patience // 2, 2),
            min_lr=reduce_lr_min,
        )

        history_p2, phase2_epochs = _train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_p2,
            scheduler=scheduler_p2,
            max_epochs=fine_tune_epochs,
            patience=max(early_stopping_patience // 2, 5),
            device=device,
            model_name=name,
            phase=2,
        )

        metrics_p2 = _extract_final_metrics(history_p2)
        logger.info(
            f"Phase 2 complete | "
            f"val_acc={metrics_p2['final_val_accuracy']:.4f} "
            f"epochs={phase2_epochs}"
        )
        # Restore best weights from Phase 2
        load_best_checkpoint(model, name, device=device)

    duration_s = time.perf_counter() - t0

    # ── Save final model ──────────────────────────────────────────────────────
    best_val_acc = float(
        max(
            history_p2.get("val_accuracy", [0.0])[-1] if history_p2 else 0.0,
            metrics_p1["final_val_accuracy"],
        )
    )

    saved_paths = save_keras_model(
        model,
        name,
        metadata={
            "epochs_phase1":        phase1_epochs,
            "epochs_phase2":        phase2_epochs,
            "final_val_accuracy":   best_val_acc,
            "final_val_loss":       float(
                history_p2.get("val_loss", [0.0])[-1] if history_p2
                else metrics_p1["final_val_loss"]
            ),
            "learning_rate":        learning_rate,
            "batch_size":           batch_size,
            "fine_tuned":           fine_tune and name != "cnn",
            "training_duration_s":  round(duration_s, 2),
        },
    )

    logger.info(
        f"Training complete | model={name} "
        f"val_acc={best_val_acc:.4f} "
        f"duration={duration_s:.1f}s"
    )

    return {
        "model_name":           name,
        "epochs_phase1":        phase1_epochs,
        "epochs_phase2":        phase2_epochs,
        "final_train_accuracy": metrics_p1["final_train_accuracy"],
        "final_val_accuracy":   best_val_acc,
        "final_train_loss":     metrics_p1["final_train_loss"],
        "final_val_loss":       metrics_p1["final_val_loss"],
        "training_duration_s":  round(duration_s, 2),
        "saved_paths":          saved_paths,
        "phase1_history":       history_p1,
        "phase2_history":       history_p2,
    }


# ─── Core training loop ───────────────────────────────────────────────────────

def _train_loop(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    max_epochs: int,
    patience: int,
    device: torch.device,
    model_name: str,
    phase: int,
) -> tuple[Dict[str, List[float]], int]:
    """
    Run the training loop for one phase.

    Returns
    -------
    tuple[dict, int]
        (history_dict, epochs_run)
        history_dict keys: "loss", "accuracy", "val_loss", "val_accuracy"
    """
    history: Dict[str, List[float]] = {
        "loss":         [],
        "accuracy":     [],
        "val_loss":     [],
        "val_accuracy": [],
    }

    best_val_loss  = float("inf")
    epochs_no_imp  = 0
    epochs_run     = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, training=False
        )

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        epochs_run = epoch

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[Phase {phase}] Epoch {epoch}/{max_epochs} | "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # Checkpoint when val_loss improves
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_imp = 0
            save_best_checkpoint(model, model_name)
        else:
            epochs_no_imp += 1

        # LR scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if epochs_no_imp >= patience:
            logger.info(
                f"[Phase {phase}] Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    return history, epochs_run
