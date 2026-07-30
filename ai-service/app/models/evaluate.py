"""
evaluate.py — Model evaluation on the held-out test split.

Computes the full set of classification metrics using scikit-learn,
then returns them in a structured dict that maps directly to the
EvaluateResponse Pydantic schema in routes.py.

Usage
-----
    from app.models.evaluate import evaluate_model
    metrics = evaluate_model("mambavision")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.config import settings
from app.core.logging import logger
from app.models.load_model import load_keras_model, get_model_info
from app.preprocessing.preprocess import build_test_generator


def evaluate_model(
    model_name: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Evaluate a trained model against a test split and return metrics.

    The function expects a directory whose structure mirrors the training
    dataset (one sub-folder per class).  The DataLoader iterates without
    augmentation and without shuffling.

    Parameters
    ----------
    model_name : str | None
        Architecture key. Falls back to ``settings.active_model``.
    dataset_dir : str | None
        Root of the test/evaluation dataset. Defaults to
        ``settings.dataset_raw_dir``.
    batch_size : int
        Batch size for the evaluation loop.

    Returns
    -------
    dict
        {
          "model_name":       str,
          "accuracy":         float,
          "precision":        float,   # macro-averaged
          "recall":           float,   # macro-averaged
          "f1":               float,   # macro-averaged
          "auc_roc":          float,   # macro OvR
          "confusion_matrix": [[int, ...], ...],
          "per_class":        {label: {"precision", "recall", "f1", "support"}, ...},
          "num_samples":      int,
          "class_names":      [str, ...],
          "model_info":       dict,    # from model_info.json
        }

    Raises
    ------
    FileNotFoundError
        When the dataset directory or model weights are not found.
    ValueError
        When no images are found in the dataset directory.
    """
    name     = (model_name or settings.active_model).lower()
    data_dir = Path(dataset_dir) if dataset_dir else settings.dataset_raw_dir
    classes  = settings.classes

    logger.info(f"Evaluation started | model={name} dataset={data_dir}")

    # ── Load model (cached TorchImageClassifier) ──────────────────────────────
    wrapped = load_keras_model(name)

    # ── Build test DataLoader ─────────────────────────────────────────────────
    test_loader = build_test_generator(
        data_dir,
        batch_size=batch_size,
        target_size=settings.image_size,
    )

    num_samples = len(test_loader.dataset)
    if num_samples == 0:
        raise ValueError(
            f"No images found in {data_dir}. "
            "Ensure the dataset directory contains class sub-folders."
        )

    # ── Run predictions ───────────────────────────────────────────────────────
    logger.info(f"Running predictions on {num_samples} test samples …")

    all_probs:  List[np.ndarray] = []
    all_labels: List[int]        = []

    for images, labels in test_loader:
        # images: torch.Tensor (N, C, H, W) float32 — already normalised
        # labels: torch.Tensor (N,) int64

        # TorchImageClassifier.predict() expects NHWC float32 in [0,1].
        # The DataLoader gives us NCHW float32 normalised.
        # We need to pass the raw tensor directly through the model instead.
        device  = wrapped.device
        images  = images.to(device, non_blocking=True)

        with torch.inference_mode():
            out    = wrapped.model(images)
            logits = out.logits if hasattr(out, "logits") else out
            probs  = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy().astype(np.float32))
        all_labels.extend(labels.numpy().tolist())

    raw_preds: np.ndarray = np.concatenate(all_probs, axis=0)   # (N, num_classes)
    y_true_raw: np.ndarray = np.array(all_labels, dtype=np.int64)

    # ── Map dataset class indices → canonical class order ─────────────────────
    dataset     = test_loader.dataset
    gen_classes: List[str]       = getattr(dataset, "classes", classes)
    gen_class_map: Dict[str, int] = getattr(dataset, "class_indices", {})

    # Build remapping: dataset_index → canonical_index
    canonical_map: Dict[int, int] = {}
    col_order: List[int] = []
    for cls in classes:
        if cls in gen_class_map:
            gen_idx = gen_class_map[cls]
            can_idx = classes.index(cls)
            canonical_map[gen_idx] = can_idx
            col_order.append(gen_idx)

    y_true = np.array(
        [canonical_map.get(int(i), int(i)) for i in y_true_raw],
        dtype=np.int64,
    )
    y_pred_indices = np.argmax(raw_preds, axis=1)
    y_pred = np.array(
        [canonical_map.get(int(i), int(i)) for i in y_pred_indices],
        dtype=np.int64,
    )

    # Reorder probability columns to match canonical class order
    if col_order:
        probs_canonical = raw_preds[:, col_order]
    else:
        probs_canonical = raw_preds

    # ── Scalar metrics ────────────────────────────────────────────────────────
    accuracy  = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        auc_roc = float(
            roc_auc_score(y_true, probs_canonical, multi_class="ovr", average="macro")
        )
    except ValueError as exc:
        logger.warning(f"AUC-ROC computation failed: {exc}")
        auc_roc = 0.0

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm: np.ndarray   = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_list: List[List[int]] = cm.tolist()

    # ── Per-class metrics ──────────────────────────────────────────────────────
    report: Dict[str, Any] = classification_report(
        y_true, y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class: Dict[str, Dict[str, float]] = {
        cls: {
            "precision": round(float(report[cls]["precision"]), 4),
            "recall":    round(float(report[cls]["recall"]),    4),
            "f1":        round(float(report[cls]["f1-score"]),  4),
            "support":   int(report[cls]["support"]),
        }
        for cls in classes
        if cls in report
    }

    logger.info(
        f"Evaluation complete | model={name} "
        f"accuracy={accuracy:.4f} f1={f1:.4f} auc_roc={auc_roc:.4f}"
    )

    return {
        "model_name":       name,
        "accuracy":         round(accuracy, 4),
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "auc_roc":          round(auc_roc, 4),
        "confusion_matrix": cm_list,
        "per_class":        per_class,
        "num_samples":      num_samples,
        "class_names":      classes,
        "model_info":       get_model_info(name),
    }
