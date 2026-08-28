"""Quick environment and config check before training."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.core.config import settings

print("=== SETTINGS ===")
print(f"active_model:          {settings.active_model}")
print(f"classes:               {settings.classes}")
print(f"num_classes:           {settings.num_classes}")
print(f"image_size:            {settings.image_size}")
print(f"dataset_processed_dir: {settings.dataset_processed_dir}")
print(f"saved_models_dir:      {settings.saved_models_dir}")

import torch
print()
print("=== TORCH ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"Device:          {'cuda' if torch.cuda.is_available() else 'cpu'}")

import os
from pathlib import Path

print()
print("=== MAMBAVISION CHECKPOINT ===")
mv_dir = settings.saved_models_dir / "mambavision"
print(f"mambavision dir exists: {mv_dir.exists()}")
if mv_dir.exists():
    items = list(mv_dir.iterdir())
    print(f"contents: {[str(i.name) for i in items]}")

print()
print("=== DATASET ===")
proc = settings.dataset_processed_dir
for split in ("train", "val", "test"):
    split_dir = proc / split
    if split_dir.exists():
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                cnt = len(list(cls_dir.glob("*.*")))
                print(f"  {split}/{cls_dir.name}: {cnt}")
    else:
        print(f"  {split}: MISSING")

print()
print("=== HF PACKAGE ===")
try:
    from transformers import AutoModelForImageClassification
    import transformers
    print(f"transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"transformers NOT found: {e}")

try:
    import mambavision
    print(f"mambavision package: {mambavision.__version__}")
except ImportError:
    print("mambavision standalone package not installed (expected - uses HF)")

print()
print("OK - environment looks good")
