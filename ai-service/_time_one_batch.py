"""Time one forward+backward pass to estimate training duration."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import torch
from app.models.architectures import build_model

print("Building model (head only, backbone frozen)...")
model = build_model("mambavision")
model.train()

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3
)
criterion = torch.nn.CrossEntropyLoss()

batch_sizes = [4, 8, 16]
for bs in batch_sizes:
    x = torch.randn(bs, 3, 224, 224)
    y = torch.randint(0, 4, (bs,))

    # Warm up
    with torch.no_grad():
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out.logits

    # Time
    t0 = time.perf_counter()
    optimizer.zero_grad()
    out = model(x)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - t0

    n_batches_per_epoch = 2210 // bs
    est_minutes = (elapsed * n_batches_per_epoch + elapsed * 472 // bs) / 60

    print(f"  batch_size={bs:2d}: {elapsed:.2f}s/batch → "
          f"~{n_batches_per_epoch} train batches → "
          f"est. {est_minutes:.1f} min/epoch")

print()
print("With backbone FROZEN only 2,564 params are updated per step.")
print("Most of the cost is the frozen forward pass through 31M params.")
