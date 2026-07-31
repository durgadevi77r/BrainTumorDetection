"""
bootstrap_model.py — Save an EfficientNetB3 model with ImageNet backbone weights.

This initialises the model architecture and saves it to saved_models/efficientnet/
so the AI service can load and run inference immediately without requiring a
full training run.

The EfficientNetB3 backbone carries pretrained ImageNet weights, which provide
meaningful spatial feature maps. The 4-class head is freshly initialised and
will produce predictions (not necessarily accurate ones until the model is
trained on the brain tumour dataset).

Run once from the ai-service directory:
    python bootstrap_model.py
"""

import sys
import os

# Ensure the ai-service package is on the path
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from app.core.config import settings
from app.core.logging import logger
from app.models.architectures import build_model
from app.models.save_model import save_model

MODEL_NAME = "efficientnet"

def main():
    print(f"\n{'='*60}")
    print(f"  Bootstrapping model: {MODEL_NAME}")
    print(f"  Input shape : {settings.input_shape}")
    print(f"  Classes     : {settings.classes}")
    print(f"  Output dir  : {settings.saved_models_dir / MODEL_NAME}")
    print(f"{'='*60}\n")

    print("Building model architecture (downloading ImageNet weights if needed)...")
    model = build_model(MODEL_NAME, learning_rate=1e-4)
    model.summary(line_length=80)

    print(f"\nSaving model to {settings.saved_models_dir / MODEL_NAME} ...")
    paths = save_model(
        model,
        MODEL_NAME,
        metadata={
            "bootstrap": True,
            "note": "Pretrained ImageNet backbone, head not yet fine-tuned on brain tumour data.",
            "val_accuracy": None,
        },
    )

    print(f"\nModel saved:")
    for k, v in paths.items():
        print(f"  {k:12}: {v}")

    print("\nBootstrap complete. The AI service can now load and run inference.")
    print("For accurate predictions, train the model with:")
    print("  POST /api/v1/train/start  {\"architecture\": \"efficientnet\", \"epochs\": 30}")

if __name__ == "__main__":
    main()
