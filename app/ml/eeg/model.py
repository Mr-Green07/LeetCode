"""Compact EEGNet-style 2D CNN for EEG classification.

Temporal convolutions learn the spike/shape morphology while the depthwise
separable blocks compress it into spectral-band sensitive filters, which is the
standard EEGNet recipe.  A clinical-feature fusion head mirrors the ECG
network so both modalities share one architecture style.
"""

from __future__ import annotations

from typing import List

import numpy as np

from app.ml.base import TORCH_AVAILABLE, nn

if TORCH_AVAILABLE:  # pragma: no cover
    import torch
    import torch.nn.functional as F
else:  # pragma: no cover
    torch = None
    F = None

from app.ml.eeg.features import FEATURE_ORDER

EEG_CLASSES: List[str] = ["NORMAL", "EPILEPSY", "PARKINSON", "DEPRESSION"]

CLASS_DESCRIPTIONS = {
    "NORMAL": "Normal EEG - background rhythm is within expected limits.",
    "EPILEPSY": "Epileptiform activity - spike-and-wave discharges consistent with seizures.",
    "PARKINSON": "Parkinsonian pattern - reduced alpha with dominant beta activity.",
    "DEPRESSION": "Depressive pattern - elevated theta/beta ratio and reduced alpha asymmetry.",
}

CLASS_SEVERITY = {
    "NORMAL": "normal",
    "EPILEPSY": "critical",
    "PARKINSON": "moderate",
    "DEPRESSION": "moderate",
}


if TORCH_AVAILABLE:  # pragma: no cover

    class EEGNet(nn.Module):
        """EEGNet convolutional core with a clinical-feature fusion head."""

        def __init__(self, num_classes: int = len(EEG_CLASSES), n_features: int = len(FEATURE_ORDER), dropout: float = 0.25):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(1, 125), stride=(1, 1), padding=(0, 64), bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 8, kernel_size=(1, 33), stride=(1, 1), padding=(0, 16), groups=8, bias=False),
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8), bias=False),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8), groups=16, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
                nn.Dropout(dropout),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                nn.Dropout(dropout),
            )
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            # Global temporal pooling keeps the fusion head independent of the
            # recording length, so the same network accepts any window size.
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Linear(32 + 64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes),
            )

        def forward(self, signal, features):
            x = self.block1(signal)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x)
            x = x.flatten(start_dim=1)
            feat = self.feature_proj(features)
            merged = torch.cat([x, feat], dim=1)
            return self.head(merged)

else:  # pragma: no cover

    class EEGNet:  # type: ignore[no-redef]
        """Placeholder raised at construction time when torch is missing."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate EEGNet")
