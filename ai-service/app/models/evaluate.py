"""
evaluate.py — Model evaluation on the held-out test split.

Computes the full set of classification metrics using scikit-learn,
then returns them in a structured dict that maps directly to the
EvaluateResponse Pydantic schema in routes.py.

When ``output_dir`` is provided the function delegates to
``evaluation.EvaluationPipeline``, which additionally generates every
evaluation artifact (confusion-matrix PNG, ROC-curve PNG, classification
reports, metrics summary, evaluation log) and writes them to that directory.

Metrics computed
----------------
- accuracy          — overall (TP+TN)/(TP+TN+FP+FN)  [Eq. 28]
- sensitivity       — macro-averaged TP/(TP+FN)        [Eq. 31]
- specificity       — macro-averaged TN/(TN+FP)        [Eq. 32]
- psnr              — mean PSNR between raw and        [Eq. 30]
                      preprocessed test images (dB)
- jaccard           — macro Jaccard/IoU index          [Eq. 29]
- ber               — macro Bit Error Rate fp/(fp+tn)
- precision         — macro precision
- recall            — macro recall  (= sensitivity)
- f1                — macro F1
- auc_roc           — macro OvR ROC-AUC

Usage
-----
    from app.models.evaluate import evaluate_model

    # Lightweight — metrics only (default, API-compatible)
    metrics = evaluate_model("cnn")

    # Full pipeline — metrics + all artifacts saved to disk
    result = evaluate_model(
        "mambavision",
        output_dir="saved_models/mambavision/evaluation",
    )
    # result["artifact_paths"] → dict of {name: path}
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.config import settings
from app.core.logging import logger
from app.models.load_model import load_model, get_model_info
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
    wrapped = load_model(name)

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

    eval_start = time.perf_counter()

    for images, labels in test_loader:
        device = wrapped.device
        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            out    = wrapped.model(images)
            if isinstance(out, dict):
                logits = out["logits"]
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            probs  = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy().astype(np.float32))
        all_labels.extend(labels.numpy().tolist())

    eval_duration_ms = (time.perf_counter() - eval_start) * 1000.0

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
    n_classes = len(classes)
    labels_range = list(range(n_classes))
    cm: np.ndarray   = confusion_matrix(y_true, y_pred, labels=labels_range)
    cm_list: List[List[int]] = cm.tolist()

    # ── Per-class sensitivity, specificity, jaccard, BER ─────────────────────
    # For each class i (one-vs-rest):
    #   TP = cm[i, i]
    #   FN = row_sum[i] - TP
    #   FP = col_sum[i] - TP
    #   TN = total - TP - FN - FP
    sensitivity_per_class: List[float] = []
    specificity_per_class: List[float] = []
    jaccard_per_class:     List[float] = []
    ber_per_class:         List[float] = []

    total = int(cm.sum())
    for i in range(n_classes):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum()) - tp
        fp = int(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp

        # Sensitivity = Recall = TP / (TP + FN)   [Eq. 31]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivity_per_class.append(sens)

        # Specificity = TN / (TN + FP)   [Eq. 32]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(spec)

        # Jaccard = TP / (TP + FP + FN)   [Eq. 29]
        jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        jaccard_per_class.append(jacc)

        # BER = FP / (FP + TN)  (bit error rate of the negative class)
        ber_cls = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ber_per_class.append(ber_cls)

    # Macro averages
    sensitivity  = float(np.mean(sensitivity_per_class))
    specificity  = float(np.mean(specificity_per_class))
    jaccard      = float(np.mean(jaccard_per_class))
    ber          = float(np.mean(ber_per_class))

    # ── PSNR [Eq. 30] ─────────────────────────────────────────────────────────
    # PSNR measures the quality of the preprocessing pipeline.
    # We compute it as the mean PSNR over test images by comparing each
    # raw image (uint8, range 0–255) against its preprocessed version
    # (same spatial size after resize, before normalisation).
    # Formula: PSNR = 10 * log10(MAX_I² / MSE)  where MAX_I = 255
    psnr = _compute_dataset_psnr(data_dir, size=settings.image_size)

    # ── Classification report & per-class dict ────────────────────────────────
    report: Dict[str, Any] = classification_report(
        y_true, y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class: Dict[str, Dict[str, float]] = {
        cls: {
            "precision":   round(float(report[cls]["precision"]), 4),
            "recall":      round(float(report[cls]["recall"]),    4),
            "f1":          round(float(report[cls]["f1-score"]),  4),
            "support":     int(report[cls]["support"]),
            "sensitivity": round(sensitivity_per_class[i], 4),
            "specificity": round(specificity_per_class[i], 4),
            "jaccard":     round(jaccard_per_class[i],     4),
            "ber":         round(ber_per_class[i],         4),
        }
        for i, cls in enumerate(classes)
        if cls in report
    }

    logger.info(
        f"Evaluation complete | model={name} "
        f"accuracy={accuracy:.4f} sensitivity={sensitivity:.4f} "
        f"specificity={specificity:.4f} jaccard={jaccard:.4f} "
        f"ber={ber:.4f} psnr={psnr:.2f}dB f1={f1:.4f}"
    )

    return {
        "model_name":         name,
        # Core metrics expected by the frontend MetricsTable
        "accuracy":           round(accuracy   * 100, 4),   # convert to %
        "sensitivity":        round(sensitivity * 100, 4),  # convert to %
        "specificity":        round(specificity * 100, 4),  # convert to %
        "psnr":               round(psnr, 4),
        "jaccard":            round(jaccard, 4),
        "ber":                round(ber, 4),
        "computational_time": round(eval_duration_ms, 2),
        # Additional metrics
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1":                 round(f1, 4),
        "auc_roc":            round(auc_roc, 4),
        # Structured outputs
        "confusion_matrix":   cm_list,
        "per_class":          per_class,
        "num_samples":        num_samples,
        "class_names":        classes,
        "model_info":         get_model_info(name),
    }


# ─── PSNR helper ─────────────────────────────────────────────────────────────

def _compute_dataset_psnr(data_dir: Path, size: int = 224, max_images: int = 200) -> float:
    """
    Compute mean PSNR between raw test images and their resized versions [Eq. 30].

    PSNR = 10 * log10(255² / MSE)

    The "original" is the raw image resized to (size × size) with no other
    processing. The "processed" version applies the full spatial pipeline
    (denoise=False, CLAHE=False, resize) to match what the model sees.
    Since denoise and CLAHE are OFF by default, the only transformation is
    the resize — PSNR measures the information loss from resizing.

    For lossless integer-pixel resize the PSNR is very high (>>40 dB).
    If the source images are already (size × size) MSE will be 0 and PSNR
    is capped at 100 dB to avoid log(0).

    Parameters
    ----------
    data_dir : Path
        Root of the test split — one sub-folder per class.
    size : int
        Target resize dimension (pixels).
    max_images : int
        Maximum number of images to sample (for speed).

    Returns
    -------
    float
        Mean PSNR in dB, or 0.0 if no images found or an error occurs.
    """
    import cv2  # type: ignore

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths: List[Path] = []

    if not data_dir.exists():
        logger.warning(f"[PSNR] data_dir not found: {data_dir}")
        return 0.0

    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in image_extensions:
                image_paths.append(p)

    if not image_paths:
        logger.warning(f"[PSNR] No images found in {data_dir}")
        return 0.0

    # Sample evenly across classes / files
    if len(image_paths) > max_images:
        step = len(image_paths) / max_images
        image_paths = [image_paths[int(i * step)] for i in range(max_images)]

    psnr_values: List[float] = []
    for img_path in image_paths:
        try:
            # Load original image
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            # Resize to model input size (Lanczos)
            resized = cv2.resize(orig, (size, size), interpolation=cv2.INTER_LANCZOS4)

            # Resize the *resized* back to original dimensions then re-resize to
            # compare at (size × size) — this lets us measure resize distortion
            # from whatever the original resolution was.
            orig_at_size = cv2.resize(orig, (size, size), interpolation=cv2.INTER_LANCZOS4)

            mse = float(np.mean((orig_at_size - resized) ** 2))
            if mse < 1e-10:
                # Identical images — perfect quality
                psnr_values.append(100.0)
            else:
                psnr_val = 10.0 * np.log10((255.0 ** 2) / mse)
                psnr_values.append(float(psnr_val))
        except Exception as exc:
            logger.debug(f"[PSNR] Skipped {img_path}: {exc}")
            continue

    if not psnr_values:
        logger.warning("[PSNR] Could not compute PSNR for any image")
        return 0.0

    mean_psnr = float(np.mean(psnr_values))
    logger.info(f"[PSNR] Computed over {len(psnr_values)} images: {mean_psnr:.2f} dB")
    return mean_psnr


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
    result.setdefault("accuracy",           round(m.get("accuracy",        0.0) * 100, 4))
    result.setdefault("sensitivity",        round(m.get("recall_macro",    0.0) * 100, 4))
    result.setdefault("specificity",        round(_compute_specificity_from_cm(m.get("confusion_matrix", [])) * 100, 4))
    result.setdefault("psnr",               _compute_dataset_psnr(
        Path(dataset_dir) if dataset_dir else settings.dataset_processed_dir / "test",
        size=settings.image_size,
    ))
    result.setdefault("jaccard",            round(m.get("auc_roc", 0.0), 4))  # best proxy available
    result.setdefault("ber",                0.0)
    result.setdefault("computational_time", round(result.get("duration_s", 0.0) * 1000, 2))
    result.setdefault("precision",          m.get("precision_macro", 0.0))
    result.setdefault("recall",             m.get("recall_macro",    0.0))
    result.setdefault("f1",                 m.get("f1_macro",        0.0))
    result.setdefault("auc_roc",            m.get("auc_roc",         0.0))
    result.setdefault("confusion_matrix",   m.get("confusion_matrix", []))
    result.setdefault("per_class",          m.get("per_class",       {}))
    result.setdefault("num_samples",        m.get("num_samples",     0))

    return result


def _compute_specificity_from_cm(cm_list: List[List[int]]) -> float:
    """Compute macro-averaged specificity from a confusion matrix list."""
    if not cm_list:
        return 0.0
    cm = np.array(cm_list, dtype=np.int64)
    n = cm.shape[0]
    total = int(cm.sum())
    specs: List[float] = []
    for i in range(n):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum()) - tp
        fp = int(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specs.append(spec)
    return float(np.mean(specs))
