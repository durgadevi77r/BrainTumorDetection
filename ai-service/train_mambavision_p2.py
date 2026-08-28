"""
train_mambavision_p2.py — Phase 2 fine-tuning for MambaVision-T.

Prerequisites:
    Run train_mambavision.py first.  This script loads the Phase-1 weights,
    unfreezes the top 2 backbone levels, and runs up to 10 epochs of full
    fine-tuning on raw images.

Estimated time:  ~9 min/epoch × up to 10 epochs = ~90 min on CPU.

Outputs:
    saved_models/mambavision/  (weights overwritten with Phase-2 best)
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))


def log(msg: str) -> None:
    print(msg, flush=True)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from app.core.config import settings
from app.models.load_model import load_model, clear_model_cache
from app.models.architectures import unfreeze_top_layers
from app.models.save_model import save_model
from app.preprocessing.augmentation import (
    MRIDataset,
    build_eval_transform,
    AugmentationConfig,
    build_data_generators_from_split,
)

log("=" * 70)
log("  MambaVision Brain Tumor Classifier — Phase 2 Fine-Tuning")
log("=" * 70)
log("")

# ── Hyperparameters ───────────────────────────────────────────────────────────
DEVICE      = torch.device("cpu")
BATCH_SIZE  = 32
P2_EPOCHS   = 10
P2_LR       = 5e-5
SEED        = 42
ES_PATIENCE = 5
CLASS_WEIGHTS = {
    "glioma":     0.853,
    "meningioma": 0.844,
    "notumor":    1.995,
    "pituitary":  0.877,
}

torch.manual_seed(SEED)

proc = settings.dataset_processed_dir
log(f"Dataset:      {proc}")
log(f"Device:       CPU")
log(f"Phase-2:      up to {P2_EPOCHS} epochs, LR={P2_LR}")
log("")


# ── 1. Load Phase-1 model ─────────────────────────────────────────────────────
log("Loading Phase-1 model ...")
clear_model_cache()
wrapped = load_model("mambavision")
model   = wrapped.model      # unwrap TorchImageClassifier → raw nn.Module
model.to(DEVICE)

# Verify head is 4-class
head = model.model.head
log(f"  Head: {head}")
assert head.out_features == 4, f"Expected 4-class head, got {head.out_features}"

p1_val_acc = 0.0
try:
    import json
    info = json.load(open(settings.saved_models_dir / "mambavision" / "model_info.json"))
    p1_val_acc = info.get("best_val_accuracy", 0.0)
    log(f"  Phase-1 best val_acc: {p1_val_acc:.4f}")
except Exception:
    pass
log("")


# ── 2. Unfreeze top 2 backbone modules ───────────────────────────────────────
log("Unfreezing top 2 backbone modules ...")
model = unfreeze_top_layers(model, n_layers=2)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"  Trainable params: {trainable:,}")
log("")


# ── 3. Build augmented data loaders ──────────────────────────────────────────
aug_cfg = AugmentationConfig(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
train_loader, val_loader = build_data_generators_from_split(
    proc / "train",
    proc / "val",
    image_size=settings.image_size,
    batch_size=BATCH_SIZE,
    aug_cfg=aug_cfg,
    seed=SEED,
    num_workers=0,
    class_names=settings.classes,
)
log(f"DataLoaders: train={len(train_loader.dataset)} val={len(val_loader.dataset)} batch={BATCH_SIZE}")
log("")

cw = torch.tensor(
    [float(CLASS_WEIGHTS.get(c, 1.0)) for c in settings.classes],
    dtype=torch.float32,
)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=P2_LR, weight_decay=1e-4,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=P2_EPOCHS, eta_min=1e-7
)


# ── 4. Fine-tuning loop ───────────────────────────────────────────────────────
log(f"Phase 2: fine-tuning for up to {P2_EPOCHS} epochs ...")
log(f"  Estimated ~9 min/epoch on CPU")
log("")

best_val_loss   = float("inf")
best_val_acc    = p1_val_acc
best_state      = None
no_improve      = 0

for epoch in range(1, P2_EPOCHS + 1):
    t_ep = time.perf_counter()

    # train
    model.train()
    t_loss = t_correct = t_total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out    = model(imgs)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        loss   = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        t_loss    += loss.item() * imgs.size(0)
        t_correct += (logits.argmax(1) == lbls).sum().item()
        t_total   += imgs.size(0)
    tr_loss = t_loss / t_total
    tr_acc  = t_correct / t_total

    # val
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out    = model(imgs)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            loss   = criterion(logits, lbls)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (logits.argmax(1) == lbls).sum().item()
            v_total   += imgs.size(0)
    val_loss = v_loss / v_total
    val_acc  = v_correct / v_total

    elapsed = time.perf_counter() - t_ep
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    improved = val_loss < best_val_loss - 1e-4

    log(
        f"  [P2] Epoch {epoch:2d}/{P2_EPOCHS} | "
        f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
        f"lr={lr:.2e} | {elapsed:.0f}s"
        + (" *best*" if improved else "")
    )

    if improved:
        best_val_loss = val_loss
        best_val_acc  = val_acc
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= ES_PATIENCE:
            log(f"  [P2] Early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    log("  Best Phase-2 weights restored.")

log("")
log(f"Phase 2 complete | best val_loss={best_val_loss:.4f} val_acc={best_val_acc:.4f}")
log("")


# ── 5. Save ───────────────────────────────────────────────────────────────────
log("Saving model ...")
model.eval()
paths = save_model(
    model,
    "mambavision",
    metadata={
        "architecture":      "mambavision",
        "best_val_accuracy": best_val_acc,
        "best_val_loss":     best_val_loss,
        "phase2_epochs":     epoch,
        "fine_tuned":        True,
        "class_names":       settings.classes,
        "class_mapping":     {c: i for i, c in enumerate(settings.classes)},
        "learning_rate_p2":  P2_LR,
        "batch_size":        BATCH_SIZE,
        "image_size":        settings.image_size,
        "training_device":   "cpu",
        "training_strategy": "fine_tuning",
    },
)
log(f"  Saved to: {paths['model_dir']}")
log("")


# ── 6. Test-set evaluation ────────────────────────────────────────────────────
log("Test-set evaluation ...")
eval_transform = build_eval_transform(settings.image_size)
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
log(f"  Test accuracy: {test_acc:.4f}")

n_cls = len(settings.classes)
cm = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(all_true, all_preds):
    cm[t][p] += 1

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

log("")
log("=" * 70)
log("  PHASE 2 COMPLETE")
log("=" * 70)
log(f"  P1 val acc:    {p1_val_acc:.4f}")
log(f"  P2 val acc:    {best_val_acc:.4f}")
log(f"  Test accuracy: {test_acc:.4f}")
log(f"  Model saved:   {paths['model_dir']}")
