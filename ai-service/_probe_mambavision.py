"""
Probe MambaVision model structure to identify the correct layer for Grad-CAM.
"""
import sys
import os

sys.path.insert(0, ".")
os.environ["AI_SERVICE_ENV"] = "test"

import torch
from transformers import AutoModelForImageClassification

print("Loading MambaVision from HF cache or hub ...")
model = AutoModelForImageClassification.from_pretrained(
    "nvidia/MambaVision-T-1K",
    trust_remote_code=True,
    num_labels=4,
    ignore_mismatched_sizes=True,
)
model.eval()

print("\n=== ALL NAMED MODULES ===")
layer_types = {}
for name, m in model.named_modules():
    t = type(m).__name__
    layer_types[t] = layer_types.get(t, 0) + 1

print("\n=== LAYER TYPE COUNTS ===")
for t, c in sorted(layer_types.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

print("\n=== TOP-LEVEL CHILDREN ===")
for name, m in model.named_children():
    print(f"  {name!r}: {type(m).__name__}")

print("\n=== MODEL.MODEL CHILDREN (if exists) ===")
if hasattr(model, "model"):
    for name, m in model.model.named_children():
        print(f"  model.{name!r}: {type(m).__name__}")
    print("\n  model.model deeper structure:")
    if hasattr(model.model, "model"):
        for name, m in model.model.model.named_children():
            print(f"    model.model.{name!r}: {type(m).__name__}")

print("\n=== LOOKING FOR CONV2D LAYERS ===")
conv_layers = []
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        conv_layers.append((name, m))
        print(f"  Conv2d: {name!r}  in={m.in_channels} out={m.out_channels} k={m.kernel_size}")

print(f"\nTotal Conv2d layers: {len(conv_layers)}")

print("\n=== LAST 5 CONV2D LAYERS ===")
for name, m in conv_layers[-5:]:
    print(f"  {name!r}  in={m.in_channels} out={m.out_channels} k={m.kernel_size}")

print("\n=== LOOKING FOR BATCHNORM / LAYERNORM ===")
norm_names = []
for name, m in model.named_modules():
    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
        norm_names.append((name, type(m).__name__))
print(f"  BN/LN count: {len(norm_names)}")
for n, t in norm_names[-5:]:
    print(f"  last 5: {n!r} ({t})")

print("\n=== FORWARD PASS to observe output shapes via hooks ===")
activation_shapes = {}

def make_hook(layer_name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_shapes[layer_name] = tuple(output.shape)
        elif hasattr(output, "last_hidden_state"):
            activation_shapes[layer_name] = tuple(output.last_hidden_state.shape)
    return hook

# Register hooks on the last few conv layers
hooks = []
if conv_layers:
    for name, m in conv_layers[-3:]:
        hooks.append(m.register_forward_hook(make_hook(name)))

# Also hook top-level model children
if hasattr(model, "model"):
    if hasattr(model.model, "layers"):
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(make_hook(f"model.layers[{i}]")))

dummy = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)

for h in hooks:
    h.remove()

print("\n  Output type:", type(out).__name__)
if hasattr(out, "logits"):
    print("  Output logits shape:", out.logits.shape)

print("\n  Captured activation shapes:")
for name, shape in activation_shapes.items():
    print(f"    {name!r}: {shape}")

print("\n=== CHECKING MODEL STRUCTURE FOR FEATURE STAGES ===")
# Specifically look for the MambaVision stage structure
if hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
    print(f"  model.model.layers has {len(layers)} stage(s)")
    for i, stage in enumerate(layers):
        print(f"  Stage {i}: {type(stage).__name__}")
        for cname, child in stage.named_children():
            print(f"    .{cname}: {type(child).__name__}")

# Check for norm layers, patch_embed, etc.
print("\n=== FULL NAMED MODULES (selected) ===")
for name, m in model.named_modules():
    if any(x in name for x in ["norm", "head", "patch_embed", "downsample", "proj", "classifier"]):
        print(f"  {name!r}: {type(m).__name__}")
