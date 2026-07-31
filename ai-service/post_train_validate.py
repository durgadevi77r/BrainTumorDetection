"""
post_train_validate.py -- Auto-runs after training completes.

DEPRECATED: This script uses TensorFlow/Keras and was part of the legacy
training pipeline. The project has migrated to PyTorch. Use
evaluation/evaluator.py or POST /api/v1/evaluate instead.

1. Loads the Phase-2 best checkpoint into the model
2. Saves efficientnet.keras + model_info.json
3. Runs full test-set validation
4. Prints confusion matrix + classification report
5. Prints sample predictions (3 per class)

Run from ai-service/:
    python -X utf8 post_train_validate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json, numpy as np
from pathlib import Path
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB3
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE   = 224
SPLIT_DIR  = Path("dataset/processed")
MODEL_DIR  = Path("saved_models/efficientnet")
CKPT_P2    = MODEL_DIR / "checkpoints" / "best_phase2.weights.h5"
CKPT_P1    = MODEL_DIR / "checkpoints" / "best_phase1.weights.h5"
MODEL_PATH = MODEL_DIR / "efficientnet.keras"
INFO_PATH  = MODEL_DIR / "model_info.json"
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]

print("=" * 72)
print("  POST-TRAINING VALIDATION")
print("=" * 72)

# Pick best available checkpoint
ckpt = CKPT_P2 if CKPT_P2.exists() else CKPT_P1
print(f"Loading checkpoint: {ckpt}")

# Rebuild architecture
backbone = EfficientNetB3(include_top=False, weights="imagenet",
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
backbone.trainable = True
inp  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
x    = backbone(inp, training=False)
x    = layers.GlobalAveragePooling2D(name="gap")(x)
x    = layers.BatchNormalization(name="bn_head")(x)
x    = layers.Dense(256, activation="relu", name="fc1")(x)
x    = layers.Dropout(0.4, name="dropout_head")(x)
out  = layers.Dense(4, activation="softmax", name="predictions")(x)
model = models.Model(inp, out, name="efficientnet")
model.compile(optimizer=optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights(str(ckpt))
print(f"Weights loaded -- {model.count_params():,} params")

# Save full model
model.save(str(MODEL_PATH))
print(f"Saved -> {MODEL_PATH}")

# Run test-set inference using the SAME preprocessing as live server
from app.preprocessing.preprocess import preprocess_for_inference
from app.preprocessing.config import DEFAULT_CONFIG
print(f"\nPreprocess: normalise={DEFAULT_CONFIG.normalise}, clahe={DEFAULT_CONFIG.apply_clahe}")

y_true, y_pred, y_conf = [], [], []
print("\n" + "=" * 72)
print("  SAMPLE PREDICTIONS  (first 3 per class)")
print("=" * 72)
print(f"  {'Image':<38} {'True':<12} {'Pred':<12} {'Conf':>6}  Status")
print(f"  {'-'*38} {'-'*12} {'-'*12} {'-'*6}  ------")

shown = {c: 0 for c in CLASSES}
for ci, cls in enumerate(CLASSES):
    cdir = SPLIT_DIR / "test" / cls
    if not cdir.exists():
        continue
    imgs = sorted(cdir.glob("*.jpg")) + sorted(cdir.glob("*.jpeg")) + sorted(cdir.glob("*.png"))
    for img in imgs:
        t   = preprocess_for_inference(str(img), expand_dims=True)
        p   = model.predict(t, verbose=0)[0]
        pi  = int(np.argmax(p))
        cf  = float(p[pi])
        y_true.append(ci); y_pred.append(pi); y_conf.append(cf)
        if shown[cls] < 3:
            ok = "OK   " if pi == ci else "WRONG"
            print(f"  {img.name:<38} {cls:<12} {CLASSES[pi]:<12} {cf:>5.1%}  {ok}")
            shown[cls] += 1

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_conf = np.array(y_conf)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\n" + "=" * 72)
print("  CONFUSION MATRIX  (rows=true, cols=predicted)")
print("=" * 72)
cw = 13
print(" " * 14 + "".join(f"{c:>{cw}}" for c in CLASSES))
print(" " * 14 + "-" * (cw * 4))
for i, row in enumerate(cm):
    print(f"  {CLASSES[i]:>11} |" + "".join(f"{v:>{cw}}" for v in row))

# Per-class metrics
print("\n" + "=" * 72)
print("  PER-CLASS ACCURACY")
print("=" * 72)
print(f"  {'Class':<14} {'Correct':>8} {'Total':>7} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print(f"  {'-'*14} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for i, cls in enumerate(CLASSES):
    mask    = y_true == i
    correct = int(np.sum(y_pred[mask] == i))
    total   = int(mask.sum())
    tp, fp, fn = correct, int(np.sum((y_pred==i)&(y_true!=i))), total-correct
    acc  = correct/total if total else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    print(f"  {cls:<14} {correct:>8} {total:>7} {acc:>7.1%}  {prec:>7.1%}  {rec:>7.1%}  {f1:>7.1%}")

overall = float(np.mean(y_true == y_pred))
print(f"\n  Overall accuracy : {overall:.4f}  ({overall:.1%})")
print(f"  Avg confidence   : {np.mean(y_conf):.4f}")
print(f"  Test samples     : {len(y_true)}")

# sklearn report
report = classification_report(
    [CLASSES[i] for i in y_true], [CLASSES[i] for i in y_pred],
    target_names=CLASSES, digits=4)
print("\n" + "=" * 72)
print("  CLASSIFICATION REPORT")
print("=" * 72)
print(report)

# Update model_info.json
# Read existing to get checkpoint val_accuracy
existing = json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}
val_acc  = existing.get("val_accuracy", None)
info = {
    "model_name":    "efficientnet",
    "save_format":   "keras",
    "input_shape":   [IMG_SIZE, IMG_SIZE, 3],
    "num_classes":   4,
    "class_names":   CLASSES,
    "class_indices": {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3},
    "total_params":  model.count_params(),
    "saved_at":      datetime.now(timezone.utc).isoformat(),
    "model_path":    str(MODEL_PATH),
    "h5_path":       str(MODEL_PATH),
    "bootstrap":     False,
    "preprocessing": "rescale_only_div255",
    "training_phase": "phase2_finetuned",
    "val_accuracy":  val_acc,
    "test_accuracy": float(overall),
    "note": (
        "Phase-2 fine-tuned (top-30 backbone layers). "
        "Preprocessing: /255 rescale only -- EfficientNetB3 has built-in [-1,1] mapping."
    ),
}
INFO_PATH.write_text(json.dumps(info, indent=2))
print(f"\nmodel_info.json updated -> {INFO_PATH}")
print("=" * 72)
print(f"  DONE  test_accuracy={overall:.4f}  checkpoint={ckpt.name}")
print("=" * 72)
