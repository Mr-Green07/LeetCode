"""Smoke-test the real chest X-ray loader and measure training throughput."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from app.data.xray_datasets import load_nih_parquet  # noqa: E402

ROOT = Path(r"D:\minimaxc\meddiag\data\datasets\nih_cxr")

start = time.time()
dataset = load_nih_parquet(ROOT, size=224)
print(f"loaded {len(dataset)} images in {time.time() - start:.1f}s")
print("per class:", dataset.metadata["per_class"])
print("image shape:", dataset.images.shape, dataset.images.dtype)
print("groups:", len(set(dataset.groups)), "unique for", len(dataset), "images")
print("sample group ids:", dataset.groups[:3])

# Class balance and the first prediction-order sanity check
counts = np.bincount(dataset.labels, minlength=6)
print("class counts (NORM,PNEU,CONS,EFF,TB,COVID):", counts.tolist())

print("\n--- throughput probe: one training batch ---")
import torch  # noqa: E402

from app.ml.xray.features import FEATURE_ORDER  # noqa: E402
from app.ml.xray.model import XrayCNN  # noqa: E402

network = XrayCNN(num_classes=6)
optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
images = torch.from_numpy(dataset.images[:32])
features = torch.from_numpy(np.zeros((32, len(FEATURE_ORDER)), dtype=np.float32))
labels = torch.from_numpy(dataset.labels[:32])
start = time.time()
for _ in range(3):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(network(images, features), labels)
    loss.backward()
    optimizer.step()
per_batch = (time.time() - start) / 3
print(f"one batch of 32: {per_batch:.2f}s -> {per_batch / 32 * 1000:.0f} ms/image")
print(f"1000 images/epoch ~ {per_batch * 1000 / 32:.0f}s")
