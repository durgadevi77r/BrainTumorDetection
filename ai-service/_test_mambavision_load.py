"""
Verify MambaVision loads, head has 4 outputs, and forward pass works.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForImageClassification

NUM_CLASSES = 4

print("=" * 60)
print("Loading nvidia/MambaVision-T-1K with num_labels=4 ...")
model = AutoModelForImageClassification.from_pretrained(
    "nvidia/MambaVision-T-1K",
    trust_remote_code=True,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model.eval()

print()
print("=== Architecture walk ===")
# Walk all named modules to find head / classifier
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"  Linear: {name:50s}  in={module.in_features:6d}  out={module.out_features:6d}")

print()
print("=== Config ===")
print(f"  config.num_labels:  {model.config.num_labels}")
print(f"  config.num_classes: {getattr(model.config, 'num_classes', 'N/A')}")

print()
print("=== Forward pass ===")
dummy = torch.zeros(1, 3, 224, 224)
with torch.inference_mode():
    out = model(dummy)

print(f"  Output type:  {type(out)}")
if hasattr(out, 'logits'):
    print(f"  logits shape: {out.logits.shape}")
    logits = out.logits
elif isinstance(out, dict):
    print(f"  dict keys: {list(out.keys())}")
    logits = out.get('logits', list(out.values())[0])
    print(f"  logits shape: {logits.shape}")
else:
    print(f"  output shape: {out.shape}")
    logits = out

print(f"  logits: {logits.tolist()}")
probs = torch.softmax(logits, dim=-1)
print(f"  probs:  {probs.tolist()}")
print(f"  argmax: {logits.argmax(dim=-1).item()}")

print()
print("=== Parameters ===")
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  total:     {total:,}")
print(f"  trainable: {trainable:,}")

# Check if head has 4 or 1000 outputs
final_layer = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        final_layer = (name, module)
fname, flayer = final_layer
print()
print(f"Last linear layer: {fname}  out={flayer.out_features}")
if flayer.out_features == NUM_CLASSES:
    print("PASS: Head has correct 4-class output")
else:
    print(f"PROBLEM: Head has {flayer.out_features} outputs, expected {NUM_CLASSES}")
    print("The head needs to be replaced manually.")
