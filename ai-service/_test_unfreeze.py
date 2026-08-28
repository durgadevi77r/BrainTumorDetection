"""Test that unfreeze_top_layers works correctly for MambaVision Phase-2."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import torch
from app.models.architectures import build_model, unfreeze_top_layers

print("Building model...")
model = build_model("mambavision")
trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable before Phase-2 unfreeze: {trainable_before:,}")

# Phase 2: unfreeze top 2 modules (test with small number)
model = unfreeze_top_layers(model, n_layers=2)
trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable after unfreeze_top_layers(2): {trainable_after:,}")

if trainable_after > trainable_before:
    print("PASS: More parameters trainable after unfreeze")
else:
    print("FAIL: unfreeze_top_layers did not unfreeze more parameters")

# Show top-level children
print()
print("Top-level children of MambaVision HF model:")
for i, (name, child) in enumerate(model.named_children()):
    n_params = sum(p.numel() for p in child.parameters())
    n_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
    print(f"  [{i}] {name:20s}  total={n_params:>10,}  trainable={n_trainable:>10,}")

# Check forward still works after unfreeze
dummy = torch.zeros(1, 3, 224, 224)
with torch.inference_mode():
    out = model(dummy)
logits = out.logits if hasattr(out, 'logits') else out.get('logits', list(out.values())[0])
print()
print(f"Forward after unfreeze: shape={logits.shape}  argmax={logits.argmax().item()}")
assert logits.shape == (1, 4), f"Expected (1,4) got {logits.shape}"
print("PASS: Forward still produces 4-class output after unfreeze")
