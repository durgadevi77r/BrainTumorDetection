"""
Probe MambaVision model structure using the mambavision package directly.
"""
import sys
import os

sys.path.insert(0, ".")
os.environ["AI_SERVICE_ENV"] = "test"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch

# Try loading via the mambavision package
print("=== Trying mambavision package ===")
try:
    import mambavision
    print("mambavision package found:", mambavision.__file__)
    print("mambavision dir:", [x for x in dir(mambavision) if not x.startswith('_')])
except Exception as e:
    print("mambavision import error:", e)

# Try the build_model from the project
print("\n=== Trying project factory ===")
try:
    from app.models.mambavision.factory import build_mambavision_model
    from app.models.mambavision.config import MambaVisionHFConfig
    print("Building model with pretrained=False to inspect structure without downloading...")
    cfg = MambaVisionHFConfig(num_classes=4)
    model = build_mambavision_model(cfg=cfg, pretrained=False)
    print("Model type:", type(model).__name__)
    print("Model MRO:", [c.__name__ for c in type(model).__mro__[:5]])
    model.eval()
    
    print("\n=== TOP-LEVEL NAMED CHILDREN ===")
    for name, child in model.named_children():
        print(f"  {name!r}: {type(child).__name__}")

    print("\n=== ALL MODULE TYPES ===")
    layer_types = {}
    for name, m in model.named_modules():
        t = type(m).__name__
        layer_types[t] = layer_types.get(t, 0) + 1
    for t, c in sorted(layer_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    
    print("\n=== CONV2D LAYERS ===")
    conv_layers = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append((name, m))
    print(f"Total Conv2d: {len(conv_layers)}")
    for name, m in conv_layers[-10:]:
        print(f"  {name!r} in={m.in_channels} out={m.out_channels} k={m.kernel_size}")
    
    print("\n=== ALL MODULE NAMES (first 80) ===")
    count = 0
    for name, m in model.named_modules():
        if count < 80:
            print(f"  [{count:03d}] {name!r}: {type(m).__name__}")
        count += 1
    print(f"  ... total modules: {count}")
    
    print("\n=== FORWARD PASS WITH HOOKS ===")
    captured = {}
    
    def make_hook(n):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                captured[n] = out.shape
            elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                captured[n] = out[0].shape
        return hook
    
    handles = []
    # Hook all Conv2d layers and last few modules
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            handles.append(m.register_forward_hook(make_hook(name)))
    
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    
    for h in handles:
        h.remove()
    
    print("Output type:", type(out).__name__)
    if hasattr(out, "logits"):
        print("logits shape:", out.logits.shape)
    
    print("\nCaptured shapes (spatial tensors only):")
    spatial = [(n, s) for n, s in captured.items() if len(s) == 4 and s[2] > 1]
    for n, s in spatial[-15:]:
        print(f"  {n!r}: {s}")
    
    # Find the LAST spatial feature map
    print("\n=== LAST SPATIAL LAYER ===")
    if spatial:
        last_spatial_name, last_spatial_shape = spatial[-1]
        print(f"  Name: {last_spatial_name!r}")
        print(f"  Shape: {last_spatial_shape}")

except Exception as e:
    import traceback
    print("Project factory error:", e)
    traceback.print_exc()

# Try loading from mambavision package directly
print("\n=== Trying mambavision.models ===")
try:
    from mambavision import models as mv_models
    print("mambavision.models:", dir(mv_models))
except Exception as e:
    print("Error:", e)

try:
    from mambavision.models.mamba_vision import MambaVision
    print("MambaVision class found")
    # Check what MambaVision looks like
    import inspect
    sig = inspect.signature(MambaVision.__init__)
    print("MambaVision.__init__ params:", list(sig.parameters.keys()))
except Exception as e:
    print("MambaVision class error:", e)
