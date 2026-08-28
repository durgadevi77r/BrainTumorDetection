import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from app.models.architectures import build_model

model = build_model("mambavision")
inner = model.model  # MambaVision

print("Inner MambaVision children:")
for i, (name, child) in enumerate(inner.named_children()):
    n_params = sum(p.numel() for p in child.parameters())
    n_train  = sum(p.numel() for p in child.parameters() if p.requires_grad)
    print(f"  [{i}] {name:20s}  total={n_params:>10,}  trainable={n_train:>10,}")
