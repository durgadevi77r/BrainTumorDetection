"""
evaluation/artifacts.py — Generate and save all evaluation artifacts.

Artifacts produced
------------------
confusion_matrix.png
    Annotated heat-map with per-cell counts and row-normalised percentages.
roc_curve.png
    One-vs-Rest ROC curves for every class + macro-average AUC.
classification_report.json
    Full sklearn classification_report dict (per-class precision / recall /
    F1 / support + weighted / macro / accuracy rows).
classification_report.txt
    Human-readable version of the same report.
metrics_summary.json
    All scalar metrics plus metadata: architecture, dataset_dir,
    num_samples, class_names, checkpoint_meta, evaluated_at.
evaluation.log
    Structured text log of the evaluation run.

All files are written to *output_dir* which defaults to::

    saved_models/<architecture>/evaluation/

Usage
-----
    from evaluation.artifacts import ArtifactWriter

    writer = ArtifactWriter(
        output_dir=Path("saved_models/mambavision/evaluation"),
        class_names=["glioma", "meningioma", "notumor", "pituitary"],
    )
    writer.write_all(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
    )
    # Returns dict of {"confusion_matrix_png": Path, "roc_curve_png": Path, ...}
"""

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from app.core.logging import logger


# ─── ArtifactWriter ──────────────────────────────────────────────────────────

class ArtifactWriter:
    """
    Writes every evaluation artifact for one evaluation run.

    Parameters
    ----------
    output_dir : Path
        Directory where all artifacts are written.  Created if absent.
    class_names : list[str]
        Ordered list of class labels (must match the column/row order used
        when computing metrics).
    architecture : str
        Architecture name — embedded in the metrics summary.
    """

    def __init__(
        self,
        output_dir: Path,
        class_names: List[str],
        architecture: str = "",
    ) -> None:
        self.output_dir   = Path(output_dir)
        self.class_names  = class_names
        self.architecture = architecture
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def write_all(
        self,
        *,
        metrics: Dict[str, Any],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        dataset_dir: str = "",
        checkpoint_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Generate and persist every artifact.

        Parameters
        ----------
        metrics : dict
            Scalar metrics dict as returned by ``EvaluationPipeline.run()``.
        y_true : np.ndarray  shape (N,)
            Ground-truth class indices.
        y_pred : np.ndarray  shape (N,)
            Predicted class indices.
        y_probs : np.ndarray  shape (N, num_classes)
            Softmax probabilities for every class.
        dataset_dir : str
            Dataset path — stored in the metrics summary for provenance.
        checkpoint_meta : dict | None
            Checkpoint metadata from ``evaluation.loader`` — stored for
            provenance.

        Returns
        -------
        dict[str, Path]
            ``{artifact_name: absolute_path, ...}``
        """
        paths: Dict[str, Path] = {}

        paths["confusion_matrix_png"] = self._write_confusion_matrix(y_true, y_pred)
        paths["roc_curve_png"]        = self._write_roc_curve(y_true, y_probs)
        paths["classification_report_json"], \
            paths["classification_report_txt"] = self._write_classification_report(
                y_true, y_pred
            )
        paths["metrics_summary_json"] = self._write_metrics_summary(
            metrics,
            dataset_dir=dataset_dir,
            checkpoint_meta=checkpoint_meta or {},
            num_samples=len(y_true),
        )
        paths["evaluation_log"] = self._write_evaluation_log(
            metrics,
            dataset_dir=dataset_dir,
            checkpoint_meta=checkpoint_meta or {},
        )

        logger.info(
            f"[ArtifactWriter] All artifacts written to {self.output_dir}"
        )
        return paths

    # ─────────────────────────────────────────────────────────────────────────
    # Individual artifact writers
    # ─────────────────────────────────────────────────────────────────────────

    def _write_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Path:
        """
        Save an annotated confusion-matrix heat-map.

        Each cell shows the raw count on the first line and the row-normalised
        percentage (recall within that class) on the second line.
        """
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend — safe in all envs
        import matplotlib.pyplot as plt

        n   = len(self.class_names)
        cm  = confusion_matrix(y_true, y_pred, labels=list(range(n)))

        # Row-normalised (recall per class) — handle all-zero rows gracefully
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

        fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

        tick_marks = np.arange(n)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(self.class_names, fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.set_title(
            f"Confusion Matrix — {self.architecture or 'model'}\n"
            f"(count / row-normalised recall)",
            fontsize=12,
        )

        # Annotate cells
        thresh = cm_norm.max() / 2.0
        for i in range(n):
            for j in range(n):
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center",
                    color=color, fontsize=9,
                )

        fig.tight_layout()
        out_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"[ArtifactWriter] confusion_matrix.png → {out_path}")
        return out_path

    def _write_roc_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> Path:
        """
        Save a One-vs-Rest ROC curve plot for all classes + macro-average.

        Falls back gracefully when a class has no positive samples.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n   = len(self.class_names)
        # Binarise labels for OvR
        y_bin = label_binarize(y_true, classes=list(range(n)))
        if n == 2:
            # label_binarize returns (N,1) for binary — expand to (N,2)
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig, ax = plt.subplots(figsize=(8, 6))

        # Per-class curves
        all_fprs: List[np.ndarray] = []
        all_tprs: List[np.ndarray] = []
        palette = plt.cm.tab10(np.linspace(0, 0.9, n))  # type: ignore[attr-defined]

        for i, cls_name in enumerate(self.class_names):
            if y_bin[:, i].sum() == 0:
                logger.warning(
                    f"[ArtifactWriter] Class '{cls_name}' has no positive "
                    "samples — ROC curve skipped for this class."
                )
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            ax.plot(
                fpr, tpr,
                color=palette[i],
                lw=1.5,
                label=f"{cls_name}  (AUC = {roc_auc:.3f})",
            )

        # Macro-average: interpolate all curves onto a common grid
        if all_fprs:
            mean_fpr = np.linspace(0, 1, 200)
            mean_tpr = np.zeros_like(mean_fpr)
            for fpr, tpr in zip(all_fprs, all_tprs):
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr /= len(all_fprs)
            macro_auc = auc(mean_fpr, mean_tpr)
            ax.plot(
                mean_fpr, mean_tpr,
                color="navy", lw=2.5, linestyle="--",
                label=f"Macro-average  (AUC = {macro_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"ROC Curves (One-vs-Rest) — {self.architecture or 'model'}",
            fontsize=12,
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out_path = self.output_dir / "roc_curve.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"[ArtifactWriter] roc_curve.png → {out_path}")
        return out_path

    def _write_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[Path, Path]:
        """
        Write ``classification_report.json`` and ``classification_report.txt``.

        Returns
        -------
        tuple[Path, Path]
            (json_path, txt_path)
        """
        report_dict = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            labels=list(range(len(self.class_names))),
            output_dict=True,
            zero_division=0,
        )
        report_txt = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            labels=list(range(len(self.class_names))),
            zero_division=0,
        )

        json_path = self.output_dir / "classification_report.json"
        txt_path  = self.output_dir / "classification_report.txt"

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2)

        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"Classification Report — {self.architecture or 'model'}\n")
            fh.write("=" * 70 + "\n")
            fh.write(report_txt)

        logger.debug(f"[ArtifactWriter] classification_report → {json_path}")
        return json_path, txt_path

    def _write_metrics_summary(
        self,
        metrics: Dict[str, Any],
        *,
        dataset_dir: str,
        checkpoint_meta: Dict[str, Any],
        num_samples: int,
    ) -> Path:
        """Write ``metrics_summary.json``."""
        summary: Dict[str, Any] = {
            "architecture":    self.architecture,
            "class_names":     self.class_names,
            "num_samples":     num_samples,
            "dataset_dir":     dataset_dir,
            "evaluated_at":    datetime.now(timezone.utc).isoformat(),
            "checkpoint_meta": {
                k: v for k, v in checkpoint_meta.items()
                if k not in ("model_state", "optimizer_state", "scheduler_state")
            },
            "metrics": metrics,
        }

        out_path = self.output_dir / "metrics_summary.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        logger.debug(f"[ArtifactWriter] metrics_summary.json → {out_path}")
        return out_path

    def _write_evaluation_log(
        self,
        metrics: Dict[str, Any],
        *,
        dataset_dir: str,
        checkpoint_meta: Dict[str, Any],
    ) -> Path:
        """Write a human-readable ``evaluation.log``."""
        out_path = self.output_dir / "evaluation.log"

        lines = [
            "=" * 70,
            f"  Evaluation Log — {self.architecture or 'model'}",
            f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 70,
            "",
            "Checkpoint",
            "-" * 40,
            f"  source       : {checkpoint_meta.get('source', 'unknown')}",
            f"  path         : {checkpoint_meta.get('path', 'unknown')}",
            f"  experiment_id: {checkpoint_meta.get('experiment_id', '')}",
            f"  epoch        : {checkpoint_meta.get('epoch', 'N/A')}",
            f"  val_loss     : {checkpoint_meta.get('val_loss', 'N/A')}",
            f"  val_accuracy : {checkpoint_meta.get('val_accuracy', 'N/A')}",
            f"  saved_at     : {checkpoint_meta.get('saved_at', '')}",
            "",
            "Dataset",
            "-" * 40,
            f"  directory    : {dataset_dir or 'default'}",
            f"  num_samples  : {metrics.get('num_samples', 'N/A')}",
            f"  class_names  : {', '.join(self.class_names)}",
            "",
            "Scalar Metrics",
            "-" * 40,
            f"  accuracy            : {metrics.get('accuracy', 0):.4f}",
            f"  precision (macro)   : {metrics.get('precision_macro', 0):.4f}",
            f"  precision (weighted): {metrics.get('precision_weighted', 0):.4f}",
            f"  recall (macro)      : {metrics.get('recall_macro', 0):.4f}",
            f"  recall (weighted)   : {metrics.get('recall_weighted', 0):.4f}",
            f"  F1 (macro)          : {metrics.get('f1_macro', 0):.4f}",
            f"  F1 (weighted)       : {metrics.get('f1_weighted', 0):.4f}",
            f"  AUC-ROC (macro OvR) : {metrics.get('auc_roc', 0):.4f}",
            "",
            "Per-class Accuracy",
            "-" * 40,
        ]

        per_class_acc: Dict[str, float] = metrics.get("per_class_accuracy", {})
        for cls in self.class_names:
            acc = per_class_acc.get(cls, 0.0)
            lines.append(f"  {cls:<15} : {acc:.4f}")

        lines += [
            "",
            "Per-class Metrics (precision / recall / F1 / support)",
            "-" * 40,
        ]
        per_class: Dict[str, Any] = metrics.get("per_class", {})
        for cls in self.class_names:
            d = per_class.get(cls, {})
            lines.append(
                f"  {cls:<15} "
                f"P={d.get('precision', 0):.4f}  "
                f"R={d.get('recall', 0):.4f}  "
                f"F1={d.get('f1', 0):.4f}  "
                f"sup={d.get('support', 0)}"
            )

        lines += ["", "=" * 70, ""]

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.debug(f"[ArtifactWriter] evaluation.log → {out_path}")
        return out_path
