"""
evaluate.py — Model evaluation on the held-out test split.

Computes the full set of classification metrics using scikit-learn,
then returns them in a structured dict that maps directly to the
EvaluateResponse Pydantic schema in routes.py.

When ``output_dir`` is provided the function delegates to
``evaluation.EvaluationPipeline``, which additionally generates every
evaluation artifact (confusion-matrix PNG, ROC-curve PNG, classification
reports, metrics summary, evaluation log) and writes them to that directory.

Usage
-----
    from app.models.evaluate import evaluate_model

    # Lightweight — metrics only (default, API-compatible)
    metrics = evaluate_model("mambavision")

    # Full pipeline — metrics + all artifacts saved to disk
    result = evaluate_model(
        "mambavision",
        output_dir="saved_models/mambavision/evaluation",
    )
    # result["artifact_paths"] → dict of {name: path}
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
    *,
    output_dir: Optional[str | Path] = None,
    save_artifacts: bool = True,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model against a test split and return metrics.

    The function expects a directory whose structure mirrors the training
    dataset (one sub-folder per class).  The DataLoader iterates without
    augmentation and without shuffling.

    When ``output_dir`` is provided the full ``evaluation.EvaluationPipeline``
    is used, which additionally generates every artifact and writes them to
    that directory.

    Parameters
    ----------
    model_name : str | None
        Architecture key. Falls back to ``settings.active_model``.
    dataset_dir : str | None
        Root of the test/evaluation dataset.  Defaults to
        ``settings.dataset_processed_dir / "test"``.
    batch_size : int
        Batch size for the evaluation loop.
    output_dir : str | Path | None
        When provided, switch to the full evaluation pipeline and save
        artifacts (confusion matrix, ROC curve, reports, summary) here.
        Defaults to ``None`` (metrics-only, API-compatible mode).
    save_artifacts : bool
        Only used when ``output_dir`` is set.  Set to ``False`` to run the
        full pipeline without writing artifacts to disk.
    use_amp : bool
        Enable AMP autocast during inference (CUDA only).
        Only used when ``output_dir`` is set.

    Returns
    -------
    dict
        Metrics-only mode (``output_dir=None``):
            {
              "model_name":       str,
              "accuracy":         float,
              "precision":        float,   # macro-averaged
              "recall":           float,   # macro-averaged
              "f1":               float,   # macro-averaged
              "auc_roc":          float,   # macro OvR
              "confusion_matrix": [[int, ...], ...],
              "per_class":        {label: {precision, recall, f1, support}, ...},
              "num_samples":      int,
              "class_names":      [str, ...],
              "model_info":       dict,
            }

        Full-pipeline mode (``output_dir`` provided):
            All of the above, plus:
            {
              "artifact_paths":  {artifact_name: path_str, ...},
              "checkpoint_meta": dict,
              "duration_s":      float,
              "precision_macro":    float,
              "precision_weighted": float,
              "recall_macro":       float,
              "recall_weighted":    float,
              "f1_macro":           float,
              "f1_weighted":        float,
              "per_class_accuracy": {class_name: float, ...},
            }

    Raises
    ------
    FileNotFoundError
        When the dataset directory or model weights are not found.
    ValueError
        When no images are found in the dataset directory.
    """
    # ── Full pipeline mode ─────────────────────────────────────────────────────
    if output_dir is not None:
        return _run_full_pipeline(
            model_name=model_name,
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            output_dir=Path(output_dir),
            save_artifacts=save_artifacts,
            use_amp=use_amp,
        )

    # ── Lightweight API-compatible mode (original implementation) ─────────────
    name     = (model_name or settings.active_model).lower()
    data_dir = (
        Path(dataset_dir) if dataset_dir
        else settings.dataset_processed_dir / "test"
    )
    classes  = settings.classes

    logger.info(f"Evaluation started | model={name} dataset={data_dir}")

    # Load model (cached TorchImageClassifier)
    wrapped = load_keras_model(name)

    # Build test DataLoader
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

    logger.info(f"Running predictions on {num_samples} test samples …")

    all_probs:  List[np.ndarray] = []
    all_labels: List[int]        = []

    for images, labels in test_loader:
        device = wrapped.device
        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            out    = wrapped.model(images)
            logits = out.logits if hasattr(out, "logits") else out
            probs  = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy().astype(np.float32))
        all_labels.extend(labels.numpy().tolist())

    raw_preds:  np.ndarray = np.concatenate(all_probs, axis=0)
    y_true_raw: np.ndarray = np.array(all_labels, dtype=np.int64)

    # Map dataset class indices → canonical class order
    dataset       = test_loader.dataset
    gen_class_map: Dict[str, int] = getattr(
        dataset, "class_to_idx",
        getattr(dataset, "class_indices", {}),
    )

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
    y_pred = np.array(
        [canonical_map.get(int(i), int(i)) for i in np.argmax(raw_preds, axis=1)],
        dtype=np.int64,
    )
    probs_canonical = raw_preds[:, col_order] if col_order else raw_preds

    # Scalar metrics
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

    cm: np.ndarray   = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_list: List[List[int]] = cm.tolist()

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


# ─── Full pipeline helper ─────────────────────────────────────────────────────

def _run_full_pipeline(
    *,
    model_name: Optional[str],
    dataset_dir: Optional[str],
    batch_size: int,
    output_dir: Path,
    save_artifacts: bool,
    use_amp: bool,
) -> Dict[str, Any]:
    """
    Delegate to ``evaluation.EvaluationPipeline`` and merge model_info into
    the result so callers always get the same top-level keys regardless of mode.
    """
    from evaluation.evaluator import EvaluationPipeline  # lazy — avoids circular imports

    arch = (model_name or settings.active_model).lower()

    pipeline = EvaluationPipeline(
        architecture=arch,
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        output_dir=output_dir,
        use_amp=use_amp,
    )
    result = pipeline.run(save_artifacts=save_artifacts)

    # Merge model_info for API compatibility
    result["model_name"] = arch
    result["model_info"] = get_model_info(arch)

    # Flatten top-level scalar keys that the API routes expect
    m = result["metrics"]
    result.setdefault("accuracy",         m.get("accuracy", 0.0))
    result.setdefault("precision",        m.get("precision_macro", 0.0))
    result.setdefault("recall",           m.get("recall_macro", 0.0))
    result.setdefault("f1",               m.get("f1_macro", 0.0))
    result.setdefault("auc_roc",          m.get("auc_roc", 0.0))
    result.setdefault("confusion_matrix", m.get("confusion_matrix", []))
    result.setdefault("per_class",        m.get("per_class", {}))
    result.setdefault("num_samples",      m.get("num_samples", 0))

    return result
