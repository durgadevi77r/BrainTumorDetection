"""
download_and_train.py — Download dataset, prepare splits, train EfficientNetB3.

DEPRECATED: This script uses TensorFlow/Keras and was part of the legacy
training pipeline. The project has migrated to PyTorch. Use the Trainer
class in training/trainer.py or POST /api/v1/train/start instead.

Dataset: Brain Tumor MRI Dataset (Msoud Nickparvar, Kaggle)
  Classes: glioma / meningioma / notumor / pituitary
  ~7,023 images

Run from ai-service/:
    python download_and_train.py

Stages:
  1. Download & unzip dataset from GitHub mirror
  2. Build train/val/test split (70/15/15)
  3. Train EfficientNetB3 — Phase 1 (frozen backbone, 15 epochs)
  4. Fine-tune — Phase 2 (unfreeze top 30 layers, 10 epochs)
  5. Save model + print confusion matrix
"""
import sys, os, shutil, zipfile, urllib.request, random, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
RAW_DIR   = BASE / "dataset" / "raw"
SPLIT_DIR = BASE / "dataset" / "processed"
MODEL_DIR = BASE / "saved_models" / "efficientnet"

# ── Dataset source ─────────────────────────────────────────────────────────────
# Mirror hosted on GitHub Releases (Sartaj Bhuvaji et al., augmented split)
DATASET_URL = (
    "https://github.com/sartajbhuvaji/brain-tumor-classification-dataset"
    "/archive/refs/heads/master.zip"
)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ─────────────────────────────────────────────────────────────────────────────
def download_dataset():
    zip_path = BASE / "dataset" / "brain_tumor_dataset.zip"
    if zip_path.exists():
        print(f"[DOWNLOAD] zip already exists: {zip_path}")
        return zip_path
    print(f"[DOWNLOAD] Downloading dataset from GitHub…")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print(f"[DOWNLOAD] Saved to {zip_path}")
    return zip_path


def extract_and_organise(zip_path: Path):
    """
    Extract the Sartaj dataset and flatten into:
      raw/<class>/img.jpg
    """
    if RAW_DIR.exists() and any(RAW_DIR.glob("*/*.jpg")):
        count = sum(1 for _ in RAW_DIR.glob("*/*.jpg"))
        print(f"[EXTRACT] raw dataset already present ({count} images)")
        return

    print(f"[EXTRACT] Extracting {zip_path} …")
    extract_tmp = BASE / "dataset" / "_tmp_extract"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_tmp)

    # Sartaj repo layout:
    #   brain-tumor-classification-dataset-master/
    #     Training/   glioma_tumor / meningioma_tumor / no_tumor / pituitary_tumor
    #     Testing/    …
    # Map folder names → our class names
    folder_map = {
        "glioma_tumor":      "glioma",
        "meningioma_tumor":  "meningioma",
        "no_tumor":          "notumor",
        "pituitary_tumor":   "pituitary",
        # flat names (some mirrors)
        "glioma":     "glioma",
        "meningioma": "meningioma",
        "notumor":    "notumor",
        "pituitary":  "pituitary",
    }

    for split in ("Training", "Testing", "train", "test", "val"):
        split_dir = next(extract_tmp.glob(f"**/{split}"), None)
        if split_dir is None:
            continue
        for folder in split_dir.iterdir():
            cls = folder_map.get(folder.name.lower())
            if cls is None:
                continue
            dest = RAW_DIR / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img in folder.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    shutil.copy2(img, dest / img.name)

    shutil.rmtree(extract_tmp, ignore_errors=True)
    total = sum(1 for _ in RAW_DIR.glob("*/*.jpg"))
    total += sum(1 for _ in RAW_DIR.glob("*/*.jpeg"))
    print(f"[EXTRACT] Done. Total images in raw/: {total}")
    for cls in CLASSES:
        n = len(list((RAW_DIR / cls).glob("*.*")))
        print(f"         {cls}: {n}")


def make_split(train_r=0.70, val_r=0.15, seed=42):
    """Split raw/<class>/ into processed/train|val|test/<class>/"""
    if SPLIT_DIR.exists() and any(SPLIT_DIR.glob("train/*/*.jpg")):
        n_train = sum(1 for _ in SPLIT_DIR.glob("train/*/*.jpg"))
        n_train += sum(1 for _ in SPLIT_DIR.glob("train/*/*.jpeg"))
        print(f"[SPLIT] processed split already exists (train={n_train} images)")
        return
    print(f"[SPLIT] Building train/val/test splits …")
    random.seed(seed)
    for cls in CLASSES:
        imgs = sorted((RAW_DIR / cls).glob("*.*"))
        imgs = [p for p in imgs if p.suffix.lower() in (".jpg",".jpeg",".png")]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        splits  = {"train": imgs[:n_train],
                   "val":   imgs[n_train:n_train+n_val],
                   "test":  imgs[n_train+n_val:]}
        for sname, files in splits.items():
            dest = SPLIT_DIR / sname / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
        print(f"  {cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print("[SPLIT] Done.")


# ─────────────────────────────────────────────────────────────────────────────
def train():
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix, classification_report
    import json

    IMG_SIZE   = 224
    BATCH      = 32
    EPOCHS_1   = 20    # Phase 1: frozen backbone
    EPOCHS_2   = 15    # Phase 2: fine-tune top layers
    LR_1       = 1e-3
    LR_2       = 1e-5

    print(f"\n{'='*60}")
    print(f"  TRAINING EfficientNetB3 — {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Phase 1: {EPOCHS_1} epochs  LR={LR_1}")
    print(f"  Phase 2: {EPOCHS_2} epochs  LR={LR_2} (fine-tune top 30)")
    print(f"{'='*60}")

    # ── Data generators ────────────────────────────────────────────────────────
    # EfficientNetB3 has built-in preprocessing — feed raw [0,255] pixels.
    # Use simple rescale=1./255 so input is [0,1].
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        str(SPLIT_DIR / "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_gen = eval_datagen.flow_from_directory(
        str(SPLIT_DIR / "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        str(SPLIT_DIR / "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"\n[DATA] class_indices: {train_gen.class_indices}")
    print(f"[DATA] train={train_gen.samples} val={val_gen.samples} test={test_gen.samples}")

    # Verify class order matches our expected order
    expected = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
    actual   = train_gen.class_indices
    if actual != expected:
        print(f"\n[WARNING] class_indices mismatch!")
        print(f"  Expected: {expected}")
        print(f"  Actual  : {actual}")
        print("  The model output indices will follow the ACTUAL order.")
        # Update .env CLASS_NAMES to match
        actual_order = sorted(actual.keys(), key=lambda k: actual[k])
        print(f"  Correct CLASS_NAMES order: {','.join(actual_order)}")
        # Write corrected order to .env
        env_path = BASE / ".env"
        env_content = env_path.read_text(encoding="utf-8")
        old_line = next((l for l in env_content.splitlines() if "CLASS_NAMES=" in l), None)
        if old_line:
            new_line = f"CLASS_NAMES={','.join(actual_order)}"
            env_content = env_content.replace(old_line, new_line)
            env_path.write_text(env_content, encoding="utf-8")
            print(f"  Updated .env CLASS_NAMES → {new_line}")
    else:
        print(f"[DATA] class order confirmed: {list(actual.keys())}")

    # ── Class weights (handle imbalance) ───────────────────────────────────────
    counts = {cls: len(list((SPLIT_DIR / "train" / cls).glob("*.*"))) for cls in CLASSES}
    total  = sum(counts.values())
    class_weight = {
        train_gen.class_indices[cls]: total / (4 * counts[cls])
        for cls in counts if cls in train_gen.class_indices
    }
    print(f"[DATA] class_weights: { {k: round(v,3) for k,v in class_weight.items()} }")

    # ── Build model ────────────────────────────────────────────────────────────
    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x       = backbone(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.BatchNormalization(name="bn_head")(x)
    x       = layers.Dense(256, activation="relu", name="fc1")(x)
    x       = layers.Dropout(0.4, name="dropout_head")(x)
    outputs = layers.Dense(4, activation="softmax", name="predictions")(x)
    model   = models.Model(inputs, outputs, name="efficientnet")

    # ── Phase 1: train head only ───────────────────────────────────────────────
    model.compile(
        optimizer=optimizers.Adam(LR_1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n[PHASE 1] Training classification head ({EPOCHS_1} epochs)…")

    ckpt_dir = MODEL_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cbs = [
        callbacks.ModelCheckpoint(
            str(ckpt_dir / "best_phase1.weights.h5"),
            monitor="val_accuracy", save_best_only=True,
            save_weights_only=True, mode="max", verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1,
        ),
    ]

    h1 = model.fit(
        train_gen, epochs=EPOCHS_1,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=cbs,
    )

    best_val_acc_p1 = max(h1.history["val_accuracy"])
    print(f"\n[PHASE 1] Best val_accuracy: {best_val_acc_p1:.4f}")

    # ── Phase 2: fine-tune top 30 backbone layers ──────────────────────────────
    print(f"\n[PHASE 2] Unfreezing top 30 backbone layers ({EPOCHS_2} epochs, LR={LR_2})…")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    # Keep BatchNorm frozen to preserve learned statistics
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(LR_2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs2 = [
        callbacks.ModelCheckpoint(
            str(ckpt_dir / "best_phase2.weights.h5"),
            monitor="val_accuracy", save_best_only=True,
            save_weights_only=True, mode="max", verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
    ]

    h2 = model.fit(
        train_gen, epochs=EPOCHS_2,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=cbs2,
        initial_epoch=len(h1.history["accuracy"]),
    )

    best_val_acc_p2 = max(h2.history["val_accuracy"])
    print(f"\n[PHASE 2] Best val_accuracy: {best_val_acc_p2:.4f}")

    # ── Evaluate on test set ───────────────────────────────────────────────────
    print(f"\n[EVAL] Running on test set …")
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    idx_to_cls = {v: k for k, v in train_gen.class_indices.items()}
    y_true_names = [idx_to_cls[i] for i in y_true]
    y_pred_names = [idx_to_cls[i] for i in y_pred]

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[CONFUSION MATRIX]")
    print(f"  Classes: {[idx_to_cls[i] for i in range(4)]}")
    print(cm)

    report = classification_report(
        y_true_names, y_pred_names,
        target_names=CLASSES, digits=4,
    )
    print(f"\n[CLASSIFICATION REPORT]")
    print(report)

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"[TEST] loss={test_loss:.4f}  accuracy={test_acc:.4f}")

    # ── Save model ─────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "efficientnet.keras"
    model.save(str(model_path))
    print(f"\n[SAVE] Model saved → {model_path}")

    # ── Write model_info.json ──────────────────────────────────────────────────
    import json
    from datetime import datetime, timezone

    class_order = [idx_to_cls[i] for i in range(4)]
    model_info = {
        "model_name":     "efficientnet",
        "save_format":    "keras",
        "input_shape":    [IMG_SIZE, IMG_SIZE, 3],
        "num_classes":    4,
        "class_names":    class_order,
        "class_indices":  train_gen.class_indices,
        "total_params":   model.count_params(),
        "saved_at":       datetime.now(timezone.utc).isoformat(),
        "model_path":     str(model_path),
        "h5_path":        str(model_path),
        "bootstrap":      False,
        "preprocessing":  "rescale_only_div255",
        "val_accuracy":   float(best_val_acc_p2),
        "test_accuracy":  float(test_acc),
        "note": (
            "Trained on Brain Tumor MRI Dataset (Sartaj Bhuvaji). "
            "Preprocessing: simple /255 rescale (EfficientNetB3 has built-in [-1,1] mapping). "
            "Phase 1: frozen backbone. Phase 2: top-30 fine-tuned."
        ),
    }
    info_path = MODEL_DIR / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"[SAVE] model_info.json → {info_path}")
    print(f"\n  class_names order: {class_order}")
    print(f"  val_accuracy:      {best_val_acc_p2:.4f}")
    print(f"  test_accuracy:     {test_acc:.4f}")

    return class_order, test_acc


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    zip_path = download_dataset()
    extract_and_organise(zip_path)
    make_split()
    class_order, test_acc = train()
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"  COMPLETE in {elapsed:.1f} min")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Class order:   {class_order}")
    print(f"{'='*60}")
