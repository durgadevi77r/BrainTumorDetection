"""
train_full.py -- Download the Msoud Nickparvar dataset (7023 images, balanced)
and train EfficientNetB3 from scratch with proper settings.

DEPRECATED: This script uses TensorFlow/Keras and was part of the legacy
training pipeline. The project has migrated to PyTorch. Use the Trainer
class in training/trainer.py or POST /api/v1/train/start instead.

Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
GitHub mirror used here for direct download.

Run:  python -X utf8 train_full.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
os.environ["OMP_NUM_THREADS"]         = "4"
os.environ["TF_NUM_INTRAOP_THREADS"]  = "4"
os.environ["TF_NUM_INTEROP_THREADS"]  = "2"

import json, numpy as np, shutil, zipfile, urllib.request, time
from pathlib import Path
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE  = 224
BATCH     = 16          # conservative for CPU
EPOCHS_P1 = 25          # frozen backbone
EPOCHS_P2 = 15          # unfreeze top-50 layers
LR_P1     = 3e-4
LR_P2     = 1e-5

BASE      = Path(__file__).parent
DATA_DIR  = BASE / "dataset" / "nickparvar"
SPLIT_DIR = BASE / "dataset" / "processed_nickparvar"
MODEL_DIR = BASE / "saved_models" / "efficientnet"
CLASSES   = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Download ──────────────────────────────────────────────────────────────────
# Primary: kaggle archive via direct GitHub releases mirror
URLS = [
    "https://github.com/sartajbhuvaji/brain-tumor-classification-dataset/archive/refs/heads/master.zip",
]

def download_or_skip():
    zip_path = BASE / "dataset" / "nickparvar_raw.zip"
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.jpg")):
        cnt = sum(1 for _ in DATA_DIR.rglob("*.jpg"))
        print(f"[SKIP] Dataset already present ({cnt} images)")
        return
    if zip_path.exists():
        print(f"[SKIP] Zip already downloaded")
    else:
        for url in URLS:
            try:
                print(f"[DOWNLOAD] {url}")
                urllib.request.urlretrieve(url, zip_path)
                print(f"[DOWNLOAD] OK {zip_path.stat().st_size//1024} KB")
                break
            except Exception as e:
                print(f"[DOWNLOAD] failed: {e}")

    if not zip_path.exists():
        raise RuntimeError("Could not download dataset")

    extract_tmp = BASE / "dataset" / "_tmp_nick"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_tmp)

    folder_map = {
        "glioma_tumor":      "glioma",
        "meningioma_tumor":  "meningioma",
        "no_tumor":          "notumor",
        "pituitary_tumor":   "pituitary",
        "glioma":    "glioma",
        "meningioma":"meningioma",
        "notumor":   "notumor",
        "pituitary": "pituitary",
    }
    for split in ("Training", "Testing", "train", "test", "val"):
        sd = next(extract_tmp.glob(f"**/{split}"), None)
        if sd is None: continue
        for folder in sd.iterdir():
            cls = folder_map.get(folder.name.lower())
            if cls is None: continue
            dest = DATA_DIR / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img in folder.iterdir():
                if img.suffix.lower() in (".jpg",".jpeg",".png"):
                    shutil.copy2(img, dest / img.name)
    shutil.rmtree(extract_tmp, ignore_errors=True)
    total = sum(1 for _ in DATA_DIR.rglob("*.jpg"))
    total += sum(1 for _ in DATA_DIR.rglob("*.jpeg"))
    print(f"[EXTRACT] {total} images ready")
    for c in CLASSES:
        n = len(list((DATA_DIR/c).glob("*.*"))) if (DATA_DIR/c).exists() else 0
        print(f"  {c}: {n}")

def make_split():
    if SPLIT_DIR.exists() and any(SPLIT_DIR.glob("train/*/*.jpg")):
        n = sum(1 for _ in SPLIT_DIR.glob("train/*/*.jpg"))
        print(f"[SPLIT] Already exists (train={n})")
        return
    import random; random.seed(42)
    print("[SPLIT] Building 70/15/15 split ...")
    for cls in CLASSES:
        d = DATA_DIR / cls
        if not d.exists(): continue
        imgs = sorted(d.glob("*.*"))
        imgs = [p for p in imgs if p.suffix.lower() in (".jpg",".jpeg",".png")]
        random.shuffle(imgs)
        n = len(imgs)
        n1 = int(n*0.70); n2 = int(n*0.85)
        for sname, files in [("train",imgs[:n1]),("val",imgs[n1:n2]),("test",imgs[n2:])]:
            dest = SPLIT_DIR/sname/cls; dest.mkdir(parents=True,exist_ok=True)
            for f in files: shutil.copy2(f, dest/f.name)
        print(f"  {cls}: train={n1} val={n2-n1} test={n-n2}")

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU: {[g.name for g in gpus]}")
    else:
        print("CPU training")

    def gen(split, augment=False):
        if augment:
            dg = ImageDataGenerator(
                rescale=1./255, rotation_range=20,
                width_shift_range=0.15, height_shift_range=0.15,
                shear_range=0.1, zoom_range=0.15,
                horizontal_flip=True, fill_mode="nearest")
        else:
            dg = ImageDataGenerator(rescale=1./255)
        return dg.flow_from_directory(
            str(SPLIT_DIR/split), target_size=(IMG_SIZE,IMG_SIZE),
            batch_size=BATCH, class_mode="categorical",
            shuffle=(split=="train"), seed=42)

    train_g = gen("train", augment=True)
    val_g   = gen("val")
    test_g  = gen("test")
    print(f"class_indices: {train_g.class_indices}")
    print(f"train={train_g.samples} val={val_g.samples} test={test_g.samples}")

    # Class weights
    counts = {c: len(list((SPLIT_DIR/"train"/c).glob("*.*"))) for c in CLASSES}
    total  = sum(counts.values())
    cw = {train_g.class_indices[c]: total/(4*counts[c]) for c in CLASSES}
    print(f"class_weights: { {k:round(v,3) for k,v in cw.items()} }")

    # Build model
    bb  = EfficientNetB3(include_top=False,weights="imagenet",input_shape=(IMG_SIZE,IMG_SIZE,3))
    bb.trainable = False
    inp = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3), name="input_image")
    x   = bb(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.BatchNormalization(name="bn_head")(x)
    x   = layers.Dense(512, activation="relu", name="fc1")(x)
    x   = layers.Dropout(0.5, name="drop1")(x)
    x   = layers.Dense(256, activation="relu", name="fc2")(x)
    x   = layers.Dropout(0.3, name="drop2")(x)
    out = layers.Dense(4, activation="softmax", name="predictions")(x)
    model = models.Model(inp, out, name="efficientnet")
    model.compile(optimizer=optimizers.Adam(LR_P1),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    ckdir = MODEL_DIR/"checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)

    def run_phase(name, epochs, ck_name):
        print(f"\n[{name}] epochs={epochs}")
        h = model.fit(train_g, epochs=epochs, validation_data=val_g,
            class_weight=cw, callbacks=[
                callbacks.ModelCheckpoint(str(ckdir/ck_name), monitor="val_accuracy",
                    save_best_only=True, save_weights_only=True, mode="max", verbose=1),
                callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                    restore_best_weights=True, verbose=1),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                    patience=3, min_lr=1e-7, verbose=1),
            ])
        return max(h.history["val_accuracy"])

    best_p1 = run_phase("PHASE-1 (frozen)", EPOCHS_P1, "best_p1.weights.h5")
    print(f"Phase-1 best val_acc={best_p1:.4f}")

    # Phase-2: unfreeze top-50
    bb.trainable = True
    for lyr in bb.layers[:-50]: lyr.trainable = False
    for lyr in bb.layers:
        if isinstance(lyr, layers.BatchNormalization): lyr.trainable = False
    model.compile(optimizer=optimizers.Adam(LR_P2),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    best_p2 = run_phase("PHASE-2 (fine-tune)", EPOCHS_P2, "best_p2.weights.h5")
    print(f"Phase-2 best val_acc={best_p2:.4f}")

    # Final eval
    test_g.reset()
    yp = model.predict(test_g, verbose=1)
    yp_idx = np.argmax(yp, axis=1)
    yt_idx = test_g.classes
    i2c = {v:k for k,v in train_g.class_indices.items()}

    cm = confusion_matrix(yt_idx, yp_idx)
    print("\n[CONFUSION MATRIX] rows=true cols=pred")
    cw2=13; print(" "*14+"".join(f"{c:>{cw2}}" for c in CLASSES))
    for i,row in enumerate(cm):
        print(f"  {CLASSES[i]:>11} |"+"".join(f"{v:>{cw2}}" for v in row))

    rpt = classification_report([i2c[i] for i in yt_idx],[i2c[i] for i in yp_idx],
                                  target_names=CLASSES, digits=4)
    print("\n[CLASSIFICATION REPORT]"); print(rpt)

    tl, ta = model.evaluate(test_g, verbose=0)
    print(f"test_accuracy={ta:.4f}  test_loss={tl:.4f}")

    # Save
    mp = MODEL_DIR/"efficientnet.keras"
    model.save(str(mp))
    info = {"model_name":"efficientnet","save_format":"keras",
            "input_shape":[IMG_SIZE,IMG_SIZE,3],"num_classes":4,
            "class_names":CLASSES,
            "class_indices":{"glioma":0,"meningioma":1,"notumor":2,"pituitary":3},
            "total_params":model.count_params(),
            "saved_at":datetime.now(timezone.utc).isoformat(),
            "model_path":str(mp),"h5_path":str(mp),
            "bootstrap":False,"preprocessing":"rescale_only_div255",
            "training_phase":"full_train_phase2",
            "val_accuracy":float(best_p2),"test_accuracy":float(ta),
            "note":"Trained on Sartaj+Nickparvar dataset. /255 rescale only."}
    (MODEL_DIR/"model_info.json").write_text(json.dumps(info,indent=2))
    print(f"Saved -> {mp}")
    return ta

if __name__ == "__main__":
    t0 = time.time()
    download_or_skip()
    make_split()
    acc = train()
    print(f"TOTAL TIME: {(time.time()-t0)/60:.1f} min  test_acc={acc:.4f}")
