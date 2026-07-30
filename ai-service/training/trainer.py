"""
training/trainer.py — High-level Trainer class for the full PyTorch training pipeline.

The ``Trainer`` orchestrates every step in one cohesive object:

    1. Build train / val / test DataLoaders from the processed dataset.
    2. Build a PyTorch model via ``app.models.architectures.build_model``.
    3. Run Phase-1 training (frozen backbone, head only).
    4. Optionally run Phase-2 fine-tuning (unfrozen top-N backbone modules).
    5. Save best checkpoint weights and the final model artefact.
    6. Evaluate the model against the test split.
    7. Persist experiment metadata to disk via ``ExperimentRegistry``.

Directory layout produced
--------------------------
    <output_dir>/
        <architecture>/
            model.safetensors | weights.pt    ← final model
            model_info.json
            checkpoints/
                <experiment_id>/
                    best_weights.pt
                    checkpoint_info.json
    <log_dir>/
        experiments/
            experiment_registry.json
            <experiment_id>/
                experiment.json
                training_config.json
        training/<experiment_id>_phase1.csv
        training/<experiment_id>_phase2.csv

CLI usage
---------
    python -m training.trainer --architecture mambavision --epochs 20

Python usage
------------
    from training.config import TrainingConfig
    from training.trainer import Trainer

    cfg     = TrainingConfig(architecture="mambavision", epochs=20)
    trainer = Trainer(cfg)
    result  = trainer.run()
    print(result["experiment_id"], result["best_val_accuracy"])
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.logging import logger
from app.models.architectures import build_model, build_optimizer, unfreeze_top_layers
from app.models.evaluate import evaluate_model
from app.models.save_model import save_keras_model
from app.preprocessing.augmentation import AugmentationConfig
from app.preprocessing.preprocess import build_generators, build_test_generator
from training.callbacks import build_callbacks, CallbackBundle
from training.checkpoints import save_checkpoint_info, load_best_weights
from training.config import TrainingConfig
from training.experiment import Experiment, ExperimentRegistry


# ─── Device helper ────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates the full two-phase transfer-learning training pipeline.

    Parameters
    ----------
    cfg : TrainingConfig
        Complete training configuration.
    aug_cfg : AugmentationConfig | None
        Augmentation configuration for the training DataLoader.
        Defaults to ``AugmentationConfig()`` (MRI-tuned defaults).
    experiments_dir : Path | None
        Override where experiment metadata is persisted.
        Defaults to ``settings.log_dir / "experiments"``.

    Attributes
    ----------
    experiment : Experiment
        Created on ``Trainer.__init__``; updated and saved throughout the run.
    model : nn.Module | None
        Populated after ``_build_model()`` is called.
    device : torch.device
        CUDA if available, otherwise CPU.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        *,
        aug_cfg: Optional[AugmentationConfig] = None,
        experiments_dir: Optional[Path] = None,
    ) -> None:
        self.cfg      = cfg
        self.aug_cfg  = aug_cfg or AugmentationConfig()
        self.device   = _resolve_device()
        self.model: Optional[nn.Module] = None

        # Create the experiment record immediately so callers can reference
        # the experiment_id before training begins (e.g. job polling).
        self.experiment = Experiment.create(cfg, experiments_dir=experiments_dir)
        self.experiment.save()
        logger.info(
            f"Trainer initialised | experiment_id={self.experiment.experiment_id} "
            f"architecture={cfg.architecture} epochs={cfg.epochs} device={self.device}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    def run(self) -> Dict[str, Any]:
        """
        Execute the full training pipeline and return a result summary.

        Returns
        -------
        dict
            {
              "experiment_id":       str,
              "architecture":        str,
              "epochs_phase1":       int,
              "epochs_phase2":       int,
              "best_val_accuracy":   float | None,
              "final_val_loss":      float | None,
              "training_duration_s": float,
              "eval_metrics":        dict,
              "model_paths":         dict,
              "status":              str,
            }
        """
        self.experiment.update_status("running")
        self.experiment.save()
        t0 = time.perf_counter()

        try:
            # 1. Data loaders
            train_loader, val_loader = self._build_generators()

            # 2. Build model
            self._build_model()

            # 3. Phase 1 — train classification head
            history_p1 = self._train_phase1(train_loader, val_loader)

            # 4. Phase 2 — fine-tune backbone (optional)
            history_p2 = self._train_phase2(train_loader, val_loader)

            # 5. Save final model
            model_paths = self._save_final_model()

            # 6. Evaluate on test split
            eval_metrics = self._evaluate(model_paths)

            # 7. Finalise experiment record
            duration_s = time.perf_counter() - t0
            self.experiment.set_duration(duration_s)
            self.experiment.update_status("completed")
            self.experiment.record_model_paths(model_paths)
            self.experiment.save()

            logger.info(
                f"Training complete | experiment={self.experiment_id} "
                f"duration={duration_s:.1f}s "
                f"best_val_accuracy={self.experiment.best_val_accuracy}"
            )

        except Exception as exc:
            duration_s = time.perf_counter() - t0
            self.experiment.set_duration(duration_s)
            self.experiment.record_error(exc)
            self.experiment.save()
            logger.exception(
                f"Training failed | experiment={self.experiment_id} error={exc}"
            )
            raise

        return self._build_result_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal steps
    # ─────────────────────────────────────────────────────────────────────────

    def _build_generators(self):
        """Build train and val DataLoaders from the processed dataset directory."""
        processed_dir = self.cfg.resolved_dataset_dir

        if not processed_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {processed_dir}. "
                "Run POST /api/v1/dataset/prepare first."
            )

        train_loader, val_loader = build_generators(
            processed_dir,
            batch_size=self.cfg.batch_size,
            aug_cfg=self.aug_cfg,
            seed=self.cfg.seed,
            num_workers=self.cfg.num_workers,
        )

        # Record dataset provenance in the experiment
        train_ds = train_loader.dataset
        val_ds   = val_loader.dataset
        class_indices = getattr(train_ds, "class_indices", {})

        self.experiment.record_dataset_info({
            "dataset_dir":    str(processed_dir),
            "train_samples":  len(train_ds),
            "val_samples":    len(val_ds),
            "class_names":    getattr(train_ds, "classes", []),
            "class_to_index": class_indices,
            "batch_size":     self.cfg.batch_size,
            "class_weights":  self.cfg.class_weights,
        })
        self.experiment.save()

        logger.info(
            f"DataLoaders built | train={len(train_ds)} "
            f"val={len(val_ds)} batch={self.cfg.batch_size}"
        )
        return train_loader, val_loader

    def _build_model(self) -> None:
        """Build and move the model to the target device."""
        self.model = build_model(
            self.cfg.architecture,
            num_classes=self.cfg.num_classes,
        )
        self.model.to(self.device)
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model built | architecture={self.cfg.architecture} "
            f"total_params={total:,} trainable_params={trainable:,} device={self.device}"
        )

    def _train_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Phase 1: train only the classification head (backbone frozen)."""
        assert self.model is not None, "Call _build_model() before _train_phase1()"

        logger.info(
            f"Phase 1 start | max_epochs={self.cfg.epochs} "
            f"lr={self.cfg.learning_rate}"
        )

        optimizer = build_optimizer(
            self.model,
            self.cfg.learning_rate,
            optimizer_name=self.cfg.optimiser,
            weight_decay=self.cfg.weight_decay,
        )
        bundle = build_callbacks(self.cfg, self.experiment_id, phase=1, optimizer=optimizer)

        hist, epochs_run = self._run_loop(
            train_loader, val_loader, optimizer, bundle,
            max_epochs=self.cfg.epochs, phase=1,
        )

        self.experiment.record_phase_history(1, hist)
        save_checkpoint_info(
            self.cfg, self.experiment_id,
            metrics={k: float(v[-1]) for k, v in hist.items() if v},
            epoch=epochs_run - 1,
            phase=1,
        )
        self.experiment.save()

        # Restore best weights before Phase 2
        load_best_weights(self.model, self.cfg, self.experiment_id, device=self.device)

        val_acc = float(hist.get("val_accuracy", [0.0])[-1])
        logger.info(f"Phase 1 complete | epochs={epochs_run} val_accuracy={val_acc:.4f}")
        return hist

    def _train_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Phase 2: fine-tune top backbone modules (skipped for CNN or if disabled)."""
        assert self.model is not None

        if not self.cfg.fine_tune or self.cfg.architecture == "cnn":
            logger.info(
                f"Phase 2 skipped "
                f"(fine_tune={self.cfg.fine_tune} arch={self.cfg.architecture})"
            )
            return {}

        logger.info(
            f"Phase 2 start | unfreeze={self.cfg.fine_tune_layers} modules "
            f"max_epochs={self.cfg.fine_tune_epochs} "
            f"lr={self.cfg.effective_fine_tune_lr}"
        )

        self.model = unfreeze_top_layers(self.model, n_layers=self.cfg.fine_tune_layers)

        optimizer = build_optimizer(
            self.model,
            self.cfg.effective_fine_tune_lr,
            optimizer_name=self.cfg.optimiser,
            weight_decay=self.cfg.weight_decay,
        )
        bundle = build_callbacks(self.cfg, self.experiment_id, phase=2, optimizer=optimizer)

        hist, epochs_run = self._run_loop(
            train_loader, val_loader, optimizer, bundle,
            max_epochs=self.cfg.fine_tune_epochs, phase=2,
        )

        self.experiment.record_phase_history(2, hist)
        save_checkpoint_info(
            self.cfg, self.experiment_id,
            metrics={k: float(v[-1]) for k, v in hist.items() if v},
            epoch=epochs_run - 1,
            phase=2,
        )
        self.experiment.save()

        # Restore best weights after Phase 2
        load_best_weights(self.model, self.cfg, self.experiment_id, device=self.device)

        val_acc = float(hist.get("val_accuracy", [0.0])[-1])
        logger.info(f"Phase 2 complete | epochs={epochs_run} val_accuracy={val_acc:.4f}")
        return hist

    def _run_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        bundle: CallbackBundle,
        *,
        max_epochs: int,
        phase: int,
    ) -> tuple[Dict[str, List[float]], int]:
        """Core epoch loop shared by both phases.

        Supports:
        - Mixed-precision training (AMP) via GradScaler when CUDA is available
          and ``cfg.use_amp=True``.
        - Gradient clipping via ``cfg.grad_clip_max_norm`` (0.0 = disabled).
        """
        criterion = nn.CrossEntropyLoss(weight=self._class_weight_tensor())

        # AMP: only meaningful on CUDA; silently disabled on CPU
        amp_enabled = self.cfg.use_amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        if amp_enabled:
            logger.info(f"[Phase {phase}] AMP enabled (GradScaler active)")

        clip_norm = self.cfg.grad_clip_max_norm  # 0.0 = disabled

        history: Dict[str, List[float]] = {
            "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []
        }
        epochs_run = 0

        for epoch in range(1, max_epochs + 1):
            # ── Training pass ────────────────────────────────────────────────
            self.model.train()
            train_loss, train_acc = self._run_epoch_amp(
                train_loader, criterion, optimizer,
                scaler=scaler, amp_enabled=amp_enabled,
                clip_norm=clip_norm, training=True,
            )

            # ── Validation pass ──────────────────────────────────────────────
            self.model.eval()
            val_loss, val_acc = self._run_epoch_amp(
                val_loader, criterion, optimizer=None,
                scaler=scaler, amp_enabled=amp_enabled,
                clip_norm=0.0, training=False,
            )

            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            epochs_run = epoch

            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"[Phase {phase}] Epoch {epoch}/{max_epochs} | "
                f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.2e}"
            )

            stop = bundle.on_epoch_end(
                epoch=epoch,
                val_loss=val_loss,
                metrics={
                    "loss": train_loss, "accuracy": train_acc,
                    "val_loss": val_loss, "val_accuracy": val_acc,
                },
                model=self.model,
                optimizer=optimizer,
            )
            if stop:
                break

        return history, epochs_run

    def _run_epoch_amp(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        *,
        scaler: "torch.cuda.amp.GradScaler",
        amp_enabled: bool,
        clip_norm: float,
        training: bool,
    ) -> tuple[float, float]:
        """Run one epoch with optional AMP and gradient clipping."""
        total_loss = 0.0
        correct    = 0
        total      = 0

        ctx = torch.enable_grad() if training else torch.inference_mode()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=amp_enabled,
                ):
                    outputs = self.model(images)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = criterion(logits, labels)

                if training and optimizer is not None:
                    scaler.scale(loss).backward()
                    # Unscale before clipping so the norm is meaningful
                    if clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * images.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        mean_loss = total_loss / max(total, 1)
        accuracy  = correct   / max(total, 1)
        return mean_loss, accuracy

    def _class_weight_tensor(self) -> Optional[torch.Tensor]:
        """Build a CrossEntropyLoss weight tensor from cfg.class_weights."""
        if not self.cfg.class_weights:
            return None
        weights = [
            float(self.cfg.class_weights.get(cls, 1.0))
            for cls in self.cfg.class_names
        ]
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _save_final_model(self) -> Dict[str, Any]:
        """Save the final model and return path dict."""
        assert self.model is not None

        p1_hist = self.experiment.phase1_history
        p2_hist = self.experiment.phase2_history

        saved_paths = save_keras_model(
            self.model,
            self.cfg.architecture,
            output_dir=self.cfg.resolved_output_dir,
            metadata={
                "experiment_id":     self.experiment_id,
                "architecture":      self.cfg.architecture,
                "epochs_phase1":     len(p1_hist.get("loss", [])),
                "epochs_phase2":     len(p2_hist.get("loss", [])),
                "best_val_accuracy": self.experiment.best_val_accuracy,
                "final_val_loss":    self.experiment.final_val_loss,
                "learning_rate":     self.cfg.learning_rate,
                "batch_size":        self.cfg.batch_size,
                "fine_tuned":        self.cfg.fine_tune and self.cfg.architecture != "cnn",
            },
        )

        logger.info(f"Final model saved → {saved_paths['model_dir']}")
        return saved_paths

    def _evaluate(self, model_paths: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the saved model against the test split (if it exists)."""
        test_dir = self.cfg.resolved_dataset_dir / "test"

        if not test_dir.exists() or not any(test_dir.iterdir()):
            logger.warning(f"No test split found at {test_dir} — skipping evaluation.")
            return {}

        try:
            metrics = evaluate_model(
                model_name=self.cfg.architecture,
                dataset_dir=str(test_dir),
                batch_size=self.cfg.batch_size,
            )
            self.experiment.record_eval_metrics(metrics)
            self.experiment.save()
            logger.info(
                f"Evaluation complete | "
                f"accuracy={metrics.get('accuracy', 0):.4f} "
                f"f1={metrics.get('f1', 0):.4f}"
            )
            return metrics
        except Exception as exc:
            logger.warning(f"Post-training evaluation failed (non-fatal): {exc}")
            return {}

    def _build_result_summary(self) -> Dict[str, Any]:
        return {
            "experiment_id":       self.experiment_id,
            "architecture":        self.cfg.architecture,
            "epochs_phase1":       len(self.experiment.phase1_history.get("loss", [])),
            "epochs_phase2":       len(self.experiment.phase2_history.get("loss", [])),
            "best_val_accuracy":   self.experiment.best_val_accuracy,
            "final_val_loss":      self.experiment.final_val_loss,
            "training_duration_s": self.experiment.duration_s,
            "eval_metrics":        self.experiment.eval_metrics,
            "model_paths":         self.experiment.model_paths,
            "phase1_history":      self.experiment.phase1_history,
            "phase2_history":      self.experiment.phase2_history,
            "status":              self.experiment.status,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def train(
    architecture: str = "mambavision",
    *,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    dataset_dir: Optional[str] = None,
    fine_tune: bool = True,
    fine_tune_layers: int = 20,
    fine_tune_epochs: int = 10,
    fine_tune_lr: Optional[float] = None,
    class_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
    num_workers: int = 0,
    aug_cfg: Optional[AugmentationConfig] = None,
    experiments_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: build a ``TrainingConfig`` + ``Trainer`` and run.
    """
    cfg = TrainingConfig(
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset_dir=dataset_dir,
        fine_tune=fine_tune,
        fine_tune_layers=fine_tune_layers,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_lr=fine_tune_lr,
        class_weights=class_weights,
        seed=seed,
        num_workers=num_workers,
        image_size=settings.image_size,
        num_classes=settings.num_classes,
        class_names=settings.classes,
    )
    trainer = Trainer(cfg, aug_cfg=aug_cfg, experiments_dir=experiments_dir)
    return trainer.run()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point:  python -m training.trainer
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.trainer",
        description="Train a brain tumour classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--architecture", "-a", default="mambavision",
                   choices=list({"mambavision", "cnn", "vgg16", "resnet50", "efficientnet"}),
                   help="Model architecture to train.")
    p.add_argument("--epochs", "-e", type=int, default=30,
                   help="Maximum Phase-1 epochs.")
    p.add_argument("--batch-size", "-b", type=int, default=32, dest="batch_size")
    p.add_argument("--learning-rate", "--lr", type=float, default=1e-4,
                   dest="learning_rate")
    p.add_argument("--dataset-dir", default=None, dest="dataset_dir")
    p.add_argument("--no-fine-tune", action="store_false", dest="fine_tune", default=True)
    p.add_argument("--fine-tune-layers", type=int, default=20, dest="fine_tune_layers")
    p.add_argument("--fine-tune-epochs", type=int, default=10, dest="fine_tune_epochs")
    p.add_argument("--fine-tune-lr", type=float, default=None, dest="fine_tune_lr")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    return p


def _main() -> None:
    import json as _json

    parser = _build_arg_parser()
    args   = parser.parse_args()

    logger.info(
        f"CLI training | arch={args.architecture} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.learning_rate}"
    )

    result = train(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset_dir=args.dataset_dir,
        fine_tune=args.fine_tune,
        fine_tune_layers=args.fine_tune_layers,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    summary = {k: v for k, v in result.items()
               if k not in ("phase1_history", "phase2_history", "eval_metrics")}
    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)
    print(_json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    _main()
