"""
evaluation/evaluator.py — Core evaluation pipeline for trained MambaVision models.

``EvaluationPipeline`` is the single public class.  It:

  1. Loads the best available checkpoint via ``evaluation.loader``.
  2. Builds a non-shuffled test ``DataLoader`` from the processed dataset.
  3. Runs batch inference with optional AMP autocast (CUDA only).
  4. Computes the full sklearn metric suite.
  5. Optionally generates and saves all evaluation artifacts via
     ``evaluation.artifacts``.

Metrics computed
----------------
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-Score (macro and weighted)
- Confusion Matrix  (list[list[int]])
- Classification Report  (dict, sklearn format)
- ROC-AUC Score  (macro One-vs-Rest)
- Per-class Accuracy
- Per-class Precision / Recall / F1 / Support

Output directory (default)
--------------------------
    saved_models/<architecture>/evaluation/

Usage
-----
    from evaluation.evaluator import EvaluationPipeline

    pipeline = EvaluationPipeline(
        architecture="mambavision",
        dataset_dir="dataset/processed/test",
    )
    result = pipeline.run(save_artifacts=True)
    # result["metrics"]["accuracy"], result["artifact_paths"], ...

CLI
---
    python -m evaluation.evaluator --architecture mambavision --batch-size 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.logging import logger
from evaluation.artifacts import ArtifactWriter
from evaluation.loader import find_best_checkpoint, load_eval_model


# ─── Device helper ────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ─── Default output dir ───────────────────────────────────────────────────────

def _default_output_dir(architecture: str) -> Path:
    d = settings.saved_models_dir / architecture.lower() / "evaluation"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationPipeline
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationPipeline:
    """
    Complete evaluation pipeline for a trained brain-tumour classifier.

    Parameters
    ----------
    architecture : str
        Model architecture key (``"mambavision"`` | ``"cnn"`` | ``"vgg16"`` |
        ``"resnet50"`` | ``"efficientnet"``).
    dataset_dir : str | Path | None
        Root of the test split — must contain one sub-folder per class.
        Defaults to ``settings.dataset_processed_dir / "test"``.
    batch_size : int
        Batch size for the DataLoader.
    num_workers : int
        DataLoader worker processes.
    output_dir : str | Path | None
        Where to write evaluation artifacts.
        Defaults to ``saved_models/<architecture>/evaluation/``.
    device : torch.device | None
        Target compute device.  Auto-detected when *None*.
    use_amp : bool
        Enable AMP autocast during inference (CUDA only; silently ignored on CPU).
    model_output_dir : str | Path | None
        Override the root where saved model checkpoints are searched.
        Defaults to ``settings.saved_models_dir``.
    """

    def __init__(
        self,
        architecture: str = "mambavision",
        *,
        dataset_dir: Optional[str | Path] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        output_dir: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        model_output_dir: Optional[str | Path] = None,
    ) -> None:
        self.architecture      = architecture.lower()
        self.batch_size        = batch_size
        self.num_workers       = num_workers
        self.device            = device or _resolve_device()
        self.use_amp           = use_amp
        self.class_names: List[str] = settings.classes

        # Resolve directories
        self.dataset_dir = (
            Path(dataset_dir) if dataset_dir
            else settings.dataset_processed_dir / "test"
        )
        self.output_dir = (
            Path(output_dir) if output_dir
            else _default_output_dir(self.architecture)
        )
        self.model_output_dir = (
            Path(model_output_dir) if model_output_dir else None
        )

        logger.info(
            f"[EvaluationPipeline] init | arch={self.architecture} "
            f"dataset={self.dataset_dir} device={self.device} "
            f"amp={self.use_amp}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, *, save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline.

        Parameters
        ----------
        save_artifacts : bool
            When True, write all evaluation artifacts to ``self.output_dir``.

        Returns
        -------
        dict
            {
              "architecture":    str,
              "dataset_dir":     str,
              "num_samples":     int,
              "class_names":     list[str],
              "checkpoint_meta": dict,
              "metrics":         dict,         # all scalar + structured metrics
              "artifact_paths":  dict[str, str] | {},
              "duration_s":      float,
            }

        Raises
        ------
        FileNotFoundError
            When no checkpoint is found or the dataset directory is missing.
        ValueError
            When the test split contains no images.
        """
        t0 = time.perf_counter()

        # 1. Load model
        model, ckpt_meta = self._load_model()

        # 2. Build DataLoader
        loader = self._build_loader()

        # 3. Run inference
        y_true, y_pred, y_probs = self._run_inference(model, loader)

        # 4. Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, y_probs)

        duration_s = time.perf_counter() - t0
        metrics["duration_s"] = round(duration_s, 2)

        # 5. Generate artifacts
        artifact_paths: Dict[str, str] = {}
        if save_artifacts:
            writer = ArtifactWriter(
                output_dir=self.output_dir,
                class_names=self.class_names,
                architecture=self.architecture,
            )
            raw_paths = writer.write_all(
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred,
                y_probs=y_probs,
                dataset_dir=str(self.dataset_dir),
                checkpoint_meta=ckpt_meta,
            )
            artifact_paths = {k: str(v) for k, v in raw_paths.items()}

        logger.info(
            f"[EvaluationPipeline] complete | "
            f"accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} "
            f"auc_roc={metrics['auc_roc']:.4f} "
            f"duration={duration_s:.1f}s"
        )

        return {
            "architecture":    self.architecture,
            "dataset_dir":     str(self.dataset_dir),
            "num_samples":     int(metrics["num_samples"]),
            "class_names":     self.class_names,
            "checkpoint_meta": ckpt_meta,
            "metrics":         metrics,
            "artifact_paths":  artifact_paths,
            "duration_s":      round(duration_s, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal steps
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load best checkpoint and return (model, checkpoint_meta)."""
        model, ckpt_meta = load_eval_model(
            self.architecture,
            output_dir=self.model_output_dir,
            device=self.device,
        )
        return model, ckpt_meta

    def _build_loader(self) -> DataLoader:
        """Build the test DataLoader."""
        from app.preprocessing.preprocess import build_test_generator  # lazy

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Test dataset directory not found: {self.dataset_dir}. "
                "Run the dataset prepare step first."
            )

        loader = build_test_generator(
            self.dataset_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        n = len(loader.dataset)
        if n == 0:
            raise ValueError(
                f"No images found in {self.dataset_dir}. "
                "Ensure the directory contains class sub-folders with images."
            )

        logger.info(
            f"[EvaluationPipeline] DataLoader built | "
            f"samples={n} batch={self.batch_size}"
        )
        return loader

    def _run_inference(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run batch inference and return (y_true, y_pred, y_probs).

        Uses AMP autocast when ``self.use_amp=True`` and device is CUDA.
        Remaps dataset class indices to the canonical settings.classes order.
        """
        amp_enabled = self.use_amp and self.device.type == "cuda"
        if amp_enabled:
            logger.info("[EvaluationPipeline] AMP autocast enabled")

        model.eval()
        all_probs:  List[np.ndarray] = []
        all_labels: List[int]        = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type, enabled=amp_enabled
                ):
                    out    = model(images)
                    logits = out.logits if hasattr(out, "logits") else out
                    probs  = torch.softmax(logits, dim=-1)

                all_probs.append(probs.cpu().numpy().astype(np.float32))
                all_labels.extend(labels.numpy().tolist())

        raw_probs   = np.concatenate(all_probs, axis=0)   # (N, C)
        y_true_raw  = np.array(all_labels, dtype=np.int64)

        # ── Remap dataset class indices → canonical order ──────────────────
        dataset    = loader.dataset
        gen_classes: List[str]      = getattr(dataset, "classes", self.class_names)
        gen_cls_map: Dict[str, int] = getattr(dataset, "class_to_idx",
                                              getattr(dataset, "class_indices", {}))

        canonical_map: Dict[int, int] = {}
        col_order:     List[int]      = []
        for cls in self.class_names:
            if cls in gen_cls_map:
                gen_idx = gen_cls_map[cls]
                can_idx = self.class_names.index(cls)
                canonical_map[gen_idx] = can_idx
                col_order.append(gen_idx)

        if canonical_map:
            y_true  = np.array(
                [canonical_map.get(int(i), int(i)) for i in y_true_raw],
                dtype=np.int64,
            )
            y_probs = raw_probs[:, col_order] if col_order else raw_probs
        else:
            y_true  = y_true_raw
            y_probs = raw_probs

        y_pred = np.argmax(y_probs, axis=1).astype(np.int64)

        logger.info(
            f"[EvaluationPipeline] Inference complete | "
            f"samples={len(y_true)} classes={self.class_names}"
        )
        return y_true, y_pred, y_probs

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute the full metric suite and return a structured dict."""
        n_classes = len(self.class_names)
        labels    = list(range(n_classes))

        # ── Scalar metrics ─────────────────────────────────────────────────
        accuracy           = float(accuracy_score(y_true, y_pred))
        precision_macro    = float(precision_score(y_true, y_pred, average="macro",
                                                   labels=labels, zero_division=0))
        precision_weighted = float(precision_score(y_true, y_pred, average="weighted",
                                                   labels=labels, zero_division=0))
        recall_macro       = float(recall_score(y_true, y_pred, average="macro",
                                                labels=labels, zero_division=0))
        recall_weighted    = float(recall_score(y_true, y_pred, average="weighted",
                                                labels=labels, zero_division=0))
        f1_macro           = float(f1_score(y_true, y_pred, average="macro",
                                            labels=labels, zero_division=0))
        f1_weighted        = float(f1_score(y_true, y_pred, average="weighted",
                                            labels=labels, zero_division=0))

        # ── AUC-ROC (macro OvR) ────────────────────────────────────────────
        try:
            auc_roc = float(
                roc_auc_score(
                    y_true, y_probs,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError as exc:
            logger.warning(f"[EvaluationPipeline] AUC-ROC failed: {exc}")
            auc_roc = 0.0

        # ── Confusion matrix ───────────────────────────────────────────────
        cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=labels)

        # ── Per-class accuracy  (TP / (TP + FP + FN + TN) per class) ──────
        per_class_accuracy: Dict[str, float] = {}
        for i, cls in enumerate(self.class_names):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum()) - tp          # missed true positives
            fp = int(cm[:, i].sum()) - tp          # false alarms
            tn = int(cm.sum()) - tp - fn - fp
            denom = tp + tn + fp + fn
            per_class_accuracy[cls] = round((tp + tn) / denom, 4) if denom else 0.0

        # ── Per-class P / R / F1 / support ────────────────────────────────
        report_dict = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        per_class: Dict[str, Dict[str, float]] = {
            cls: {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall":    round(float(report_dict[cls]["recall"]),    4),
                "f1":        round(float(report_dict[cls]["f1-score"]),  4),
                "support":   int(report_dict[cls]["support"]),
            }
            for cls in self.class_names
            if cls in report_dict
        }

        return {
            # Scalar
            "accuracy":           round(accuracy,           4),
            "precision_macro":    round(precision_macro,    4),
            "precision_weighted": round(precision_weighted, 4),
            "recall_macro":       round(recall_macro,       4),
            "recall_weighted":    round(recall_weighted,    4),
            "f1_macro":           round(f1_macro,           4),
            "f1_weighted":        round(f1_weighted,        4),
            "auc_roc":            round(auc_roc,            4),
            # Legacy key kept for backward compat with app/models/evaluate.py
            "precision":          round(precision_macro,    4),
            "recall":             round(recall_macro,       4),
            "f1":                 round(f1_macro,           4),
            # Structured
            "confusion_matrix":   cm.tolist(),
            "classification_report": report_dict,
            "per_class":          per_class,
            "per_class_accuracy": per_class_accuracy,
            # Meta
            "num_samples":        len(y_true),
            "class_names":        self.class_names,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    architecture: str = "mambavision",
    *,
    dataset_dir: Optional[str | Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    output_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    model_output_dir: Optional[str | Path] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper: build an ``EvaluationPipeline`` and call ``run()``.
    """
    pipeline = EvaluationPipeline(
        architecture=architecture,
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        device=device,
        use_amp=use_amp,
        model_output_dir=model_output_dir,
    )
    return pipeline.run(save_artifacts=save_artifacts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI:  python -m evaluation.evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evaluation.evaluator",
        description="Evaluate a trained brain-tumour classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--architecture", "-a", default="mambavision",
        choices=["mambavision", "cnn", "vgg16", "resnet50", "efficientnet"],
        help="Model architecture to evaluate.",
    )
    p.add_argument("--dataset-dir",   default=None, dest="dataset_dir",
                   help="Path to test dataset directory.")
    p.add_argument("--batch-size",    type=int, default=32, dest="batch_size")
    p.add_argument("--num-workers",   type=int, default=0,  dest="num_workers")
    p.add_argument("--output-dir",    default=None, dest="output_dir",
                   help="Where to save evaluation artifacts.")
    p.add_argument("--no-artifacts",  action="store_false", dest="save_artifacts",
                   default=True,
                   help="Skip writing evaluation artifacts to disk.")
    p.add_argument("--no-amp",        action="store_false", dest="use_amp",
                   default=True,
                   help="Disable AMP autocast.")
    return p


def _main() -> None:
    import json as _json

    parser = _build_arg_parser()
    args   = parser.parse_args()

    result = evaluate(
        architecture=args.architecture,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        use_amp=args.use_amp,
        save_artifacts=args.save_artifacts,
    )

    summary = {
        k: v for k, v in result.items()
        if k not in ("metrics",)
    }
    summary["metrics"] = {
        k: v for k, v in result["metrics"].items()
        if k not in ("confusion_matrix", "classification_report",
                     "per_class", "per_class_accuracy")
    }

    print("\n" + "=" * 60)
    print("Evaluation complete")
    print("=" * 60)
    print(_json.dumps(summary, indent=2, default=str))

    if result.get("artifact_paths"):
        print("\nArtifacts written:")
        for name, path in result["artifact_paths"].items():
            print(f"  {name:<30} {path}")


if __name__ == "__main__":
    _main()
