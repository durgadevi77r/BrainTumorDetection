"""
Test the fixed factory.py - verify 4-class head and forward pass.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import torch

print("=" * 60)
print("Testing build_mambavision_model() via fixed factory.py ...")

from app.models.mambavision.factory import build_mambavision_model
from app.models.mambavision.config import MambaVisionHFConfig

cfg = MambaVisionHFConfig(num_classes=4)
model = build_mambavision_model(cfg=cfg, pretrained=True)
model.eval()

print()
print("=== Head check ===")
final_layer = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        final_layer = (name, module)
fname, flayer = final_layer
print(f"Last linear: {fname}  in={flayer.in_features}  out={flayer.out_features}")

if flayer.out_features == 4:
    print("PASS: Head correctly has 4 outputs")
else:
    print(f"FAIL: Head has {flayer.out_features} outputs, expected 4")
    sys.exit(1)

print()
print("=== Trainable parameters ===")
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = total - trainable
print(f"Total:     {total:>12,}")
print(f"Trainable: {trainable:>12,}  (head only)")
print(f"Frozen:    {frozen:>12,}  (backbone)")

print()
print("=== Forward pass ===")
dummy = torch.zeros(1, 3, 224, 224)
with torch.inference_mode():
    out = model(dummy)

logits = out.logits if hasattr(out, 'logits') else out.get('logits', list(out.values())[0])
print(f"Input shape:  {dummy.shape}")
print(f"Output shape: {logits.shape}")
print(f"Logits: {[round(x, 4) for x in logits[0].tolist()]}")
probs = torch.softmax(logits, dim=-1)
print(f"Probs:  {[round(x, 4) for x in probs[0].tolist()]}")
print(f"Sum:    {probs.sum().item():.6f}")
print(f"Argmax: {logits.argmax(dim=-1).item()}")

print()
# Now test through architectures.build_model
print("=== Test via architectures.build_model('mambavision') ===")
from app.models.architectures import build_model
model2 = build_model("mambavision")
model2.eval()

final_layer2 = None
for name, module in model2.named_modules():
    if isinstance(module, torch.nn.Linear):
        final_layer2 = (name, module)
fname2, flayer2 = final_layer2
print(f"Last linear: {fname2}  in={flayer2.in_features}  out={flayer2.out_features}")
if flayer2.out_features == 4:
    print("PASS: build_model() also returns 4-class head")
else:
    print(f"FAIL: build_model() returned {flayer2.out_features}-class head")
    sys.exit(1)

dummy2 = torch.zeros(2, 3, 224, 224)  # batch=2
with torch.inference_mode():
    out2 = model2(dummy2)
logits2 = out2.logits if hasattr(out2, 'logits') else out2.get('logits', list(out2.values())[0])
print(f"Batch forward: input={dummy2.shape}  output={logits2.shape}")
assert logits2.shape == (2, 4), f"Expected (2,4) got {logits2.shape}"
print("PASS: batch forward shape correct")

print()
print("=" * 60)
print("ALL TESTS PASSED - MambaVision factory is working correctly")
print("=" * 60)
