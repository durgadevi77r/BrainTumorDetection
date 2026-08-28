"""
train_mambavision.py — Train MambaVision-T on the brain tumor dataset.

Strategy (CPU-optimized two-stage script):
-------------------------------------------
Phase 1 — Feature Extraction + Linear Head Training  [THIS RUN]
    - Extract 640-dim features from ALL train/val images ONCE (frozen backbone).
    - Train the Linear(640, 4) head on cached features for up to 80 epochs.
    - Estimated time: ~9 min extraction + <1 min training = ~10 min total.
    - Saves model immediately on completion.

Phase 2 — Backbone Fine-Tuning  [SEPARATE RUN: train_mambavision_p2.py]
    - Loads the Phase-1 weights, unfreezes top 2 backbone levels.
    - Full forward/backward on raw images for up to 10 epochs.
    - ~9 min/epoch × 10 = ~90 min — run manually when convenient.

Outputs (Phase 1):
    saved_models/mambavision/
        config.json + model.safetensors + model_info.json
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))


def log(msg: str) -> None:
    """Print with immediate flush so log files stay live."""
    print(msg, flush=True)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import settings
from app.models.architectures import build_model
from app.models.save_model import save_model
from app.preprocessing.augmentation import (
    MRIDataset,
    build_eval_transform,
)

log("=" * 70)
log("  MambaVision Brain Tumor Classifier — Phase 1 Training")
log("=" * 70)
log("")

# ── Hyperparameters ───────────────────────────────────────────────────────────
DEVICE      = torch.device("cpu")
BATCH_SIZE  = 32        # feature extraction batch size
HEAD_EPOCHS = 80        # max epochs for head training on cached features
P1_LR       = 1e-3
SEED        = 42
ES_PATIENCE = 12        # stop if no val_loss improvement for this many epochs
CLASS_WEIGHTS = {
    "glioma":     0.853,
    "meningioma": 0.844,
    "notumor":    1.995,   # minority class — upweighted
    "pituitary":  0.877,
}

torch.manual_seed(SEED)

proc = settings.dataset_processed_dir
log(f"Dataset:      {proc}")
log(f"Classes:      {settings.classes}")
log(f"Device:       CPU")
log(f"Phase-1:      up to {HEAD_EPOCHS} epochs on cached 640-dim features")
log(f"ES patience:  {ES_PATIENCE} epochs")
log("")


# ── 1. Build model (pretrained backbone, frozen, 4-class head) ────────────────
log("Building MambaVision-T ...")
t0 = time.perf_counter()
model = build_model("mambavision")
model.to(DEVICE)
model.eval()
total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"  Built in {time.perf_counter()-t0:.1f}s")
log(f"  Total params:     {total_p:,}")
log(f"  Trainable params: {trainable_p:,}  (head only — backbone frozen)")
log("")


# ── 2. Feature extraction helpers ─────────────────────────────────────────────
def extract_features_batch(x: torch.Tensor) -> torch.Tensor:
    """Run frozen backbone, return (B, 640) feature vectors."""
    inner = model.model          # MambaVision instance
    with torch.inference_mode():
        feats, _ = inner.forward_features(x)
    return feats


eval_transform = build_eval_transform(settings.image_size)


# ── 3. Extract features (with disk cache) ────────────────────────────────────
CACHE_DIR  = proc.parent / "_feat_cache"
CACHE_DIR.mkdir(exist_ok=True)
TRAIN_CACHE = CACHE_DIR / "mambavision_train_feats.pt"
VAL_CACHE   = CACHE_DIR / "mambavision_val_feats.pt"


def extract_split(split: str, cache_path: "Path"):
    """Extract features for one split, reading from disk cache if available."""
    if cache_path.exists():
        log(f"  Loading {split} features from cache: {cache_path}")
        data = torch.load(str(cache_path), weights_only=True)
        log(f"    feats={tuple(data['feats'].shape)}  labels={tuple(data['labels'].shape)}")
        return data["feats"], data["labels"]

    ds = MRIDataset(
        proc / split,
        transform=eval_transform,
        class_names=settings.classes,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_feats, all_labels = [], []

    log(f"  Extracting {split:5s} ({len(ds):4d} images, {len(loader):2d} batches)...")
    t = time.perf_counter()
    for i, (imgs, lbls) in enumerate(loader):
        feats = extract_features_batch(imgs.to(DEVICE))
        all_feats.append(feats.cpu())
        all_labels.append(lbls)
        if (i + 1) % 20 == 0:
            log(f"    ... {i+1}/{len(loader)} batches  ({time.perf_counter()-t:.0f}s)")
    feats_t  = torch.cat(all_feats,  dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    log(f"    Done: shape={tuple(feats_t.shape)}  {time.perf_counter()-t:.1f}s")

    # Cache to disk
    torch.save({"feats": feats_t, "labels": labels_t}, str(cache_path))
    log(f"    Cached to {cache_path}")
    return feats_t, labels_t


log("Extracting backbone features (with disk cache) ...")
t_ext = time.perf_counter()
train_feats, train_labels = extract_split("train", TRAIN_CACHE)
val_feats,   val_labels   = extract_split("val",   VAL_CACHE)
log(f"  Total extraction time: {(time.perf_counter()-t_ext)/60:.1f} min")
log("")


# ── 4. Class-weighted loss ────────────────────────────────────────────────────
cw = torch.tensor(
    [float(CLASS_WEIGHTS.get(c, 1.0)) for c in settings.classes],
    dtype=torch.float32,
)
criterion = nn.CrossEntropyLoss(weight=cw)
log(f"Class weights: {dict(zip(settings.classes, [round(float(w),3) for w in cw]))}")
log("")


# ── 5. Phase 1: train head on cached features ─────────────────────────────────
log(f"Phase 1: training head on cached features (max {HEAD_EPOCHS} epochs) ...")
log("")

head = model.model.head       # nn.Linear(640, 4)
optimizer = optim.Adam(head.parameters(), lr=P1_LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=HEAD_EPOCHS, eta_min=1e-6
)

feat_train_loader = DataLoader(
    TensorDataset(train_feats, train_labels),
    batch_size=256, shuffle=True,
)
feat_val_loader = DataLoader(
    TensorDataset(val_feats, val_labels),
    batch_size=256, shuffle=False,
)

best_val_loss   = float("inf")
best_val_acc    = 0.0
best_head_state = None
no_improve      = 0
history         = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

t_p1 = time.perf_counter()
for epoch in range(1, HEAD_EPOCHS + 1):

    # ── train ──
    head.train()
    t_loss = t_correct = t_total = 0
    for feats, labels in feat_train_loader:
        optimizer.zero_grad()
        logits = head(feats)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * feats.size(0)
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total   += feats.size(0)
    tr_loss = t_loss / t_total
    tr_acc  = t_correct / t_total

    # ── val ──
    head.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for feats, labels in feat_val_loader:
            logits  = head(feats)
            loss    = criterion(logits, labels)
            v_loss    += loss.item() * feats.size(0)
            v_correct += (logits.argmax(1) == labels).sum().item()
            v_total   += feats.size(0)
    val_loss = v_loss / v_total
    val_acc  = v_correct / v_total

    history["loss"].append(tr_loss)
    history["acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    # Always log first 5, then every 5 epochs, plus whenever we improve
    improved = val_loss < best_val_loss - 1e-4
    if epoch <= 5 or epoch % 5 == 0 or improved:
        marker = " *best*" if improved else ""
        log(
            f"  [P1] Epoch {epoch:3d}/{HEAD_EPOCHS} | "
            f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={lr:.2e}{marker}"
        )

    # ── early stopping / checkpoint ──
    if improved:
        best_val_loss   = val_loss
        best_val_acc    = val_acc
        best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        no_improve      = 0
    else:
        no_improve += 1
        if no_improve >= ES_PATIENCE:
            log(f"  [P1] Early stopping at epoch {epoch} "
                f"(no improvement for {ES_PATIENCE} epochs)")
            break

# Restore best head weights
if best_head_state is not None:
    head.load_state_dict(best_head_state)
    log(f"  Best head weights restored.")

p1_elapsed = time.perf_counter() - t_p1
log("")
log(f"Phase 1 complete in {p1_elapsed:.0f}s")
log(f"  Best val_loss: {best_val_loss:.4f}")
log(f"  Best val_acc:  {best_val_acc:.4f}")
log("")


# ── 6. Save model ─────────────────────────────────────────────────────────────
log("Saving model ...")
model.eval()

paths = save_model(
    model,
    "mambavision",
    metadata={
        "architecture":      "mambavision",
        "best_val_accuracy": best_val_acc,
        "best_val_loss":     best_val_loss,
        "phase1_epochs":     len(history["loss"]),
        "phase2_epochs":     0,
        "fine_tuned":        False,
        "class_names":       settings.classes,
        "class_mapping":     {c: i for i, c in enumerate(settings.classes)},
        "learning_rate":     P1_LR,
        "batch_size":        BATCH_SIZE,
        "image_size":        settings.image_size,
        "training_device":   "cpu",
        "training_strategy": "feature_extraction",
    },
)
log(f"  Saved to: {paths['model_dir']}")
log("")


# ── 7. Test-set evaluation ────────────────────────────────────────────────────
log("Running test-set evaluation ...")
test_ds     = MRIDataset(proc / "test", transform=eval_transform, class_names=settings.classes)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        out    = model(imgs.to(DEVICE))
        logits = out["logits"] if isinstance(out, dict) else out.logits
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(lbls.tolist())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
test_acc  = (all_preds == all_true).mean()
log(f"  Test accuracy: {test_acc:.4f}  ({(all_preds==all_true).sum()}/{len(all_true)} correct)")

# Confusion matrix
n_cls = len(settings.classes)
cm = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(all_true, all_preds):
    cm[t][p] += 1

log("")
log("  Confusion matrix (rows=Actual, cols=Predicted):")
col_hdr = "             " + "  ".join(f"{c[:8]:>8}" for c in settings.classes)
log(f"  {col_hdr}")
for i, row in enumerate(cm):
    vals = "  ".join(f"{v:>8}" for v in row)
    log(f"  {settings.classes[i]:>12}: {vals}")

log("")
log("  Per-class metrics:")
for i, cls in enumerate(settings.classes):
    tp = int(cm[i, i])
    fp = int(cm[:, i].sum()) - tp
    fn = int(cm[i, :].sum()) - tp
    prec   = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)
    log(f"    {cls:>12}: TP={tp:4d}  Prec={prec:.3f}  Rec={recall:.3f}  F1={f1:.3f}")


# ── 8. Summary ────────────────────────────────────────────────────────────────
log("")
log("=" * 70)
log("  PHASE 1 TRAINING COMPLETE")
log("=" * 70)
log(f"  Val accuracy:   {best_val_acc:.4f}")
log(f"  Test accuracy:  {test_acc:.4f}")
log(f"  Model saved to: {paths['model_dir']}")
log("")
log("To run Phase 2 fine-tuning (optional, ~90 min on CPU):")
log("  python train_mambavision_p2.py")
log("")
log("To activate this model:")
log("  Set ACTIVE_MODEL=mambavision in ai-service/.env")
log("  Restart the AI service")
