"""1D residual CNN for ECG classification.

The network consumes two streams:

* the conditioned 5-second ECG window (``1 x 2500`` samples at 500 Hz);
* the 17 clinical features (rate, intervals, HRV, repolarisation) computed by
  :mod:`app.ml.ecg.features`.

Fusing the feature stream lets the CNN focus on waveform morphology while the
head still reasons over the numbers a cardiologist uses.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.ml.base import TORCH_AVAILABLE, nn

if TORCH_AVAILABLE:  # pragma: no cover - exercised when torch is installed
    import torch
    import torch.nn.functional as F
else:  # pragma: no cover
    torch = None
    F = None

from app.ml.ecg.features import FEATURE_ORDER
from app.ml.ecg.preprocess import WINDOW_SAMPLES

ECG_CLASSES: List[str] = ["NORM", "AFIB", "MI", "STDE", "PVC"]

CLASS_DESCRIPTIONS = {
    "NORM": "Normal sinus rhythm - no significant abnormality detected.",
    "AFIB": "Atrial fibrillation - irregularly irregular rhythm with absent P waves.",
    "MI": "Myocardial infarction - ST-segment depression with ischaemic morphology.",
    "STDE": "ST-segment elevation - acute myocardial injury pattern.",
    "PVC": "Premature ventricular complexes - frequent ectopic ventricular beats.",
}

CLASS_SEVERITY = {
    "NORM": "normal",
    "AFIB": "moderate",
    "MI": "severe",
    "STDE": "critical",
    "PVC": "mild",
}


if TORCH_AVAILABLE:  # pragma: no cover

    class ResNet1DBlock(nn.Module):
        """Pre-activation residual block for 1-D signals."""

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15, stride: int = 1, dropout: float = 0.2):
            super().__init__()
            # An odd kernel keeps the residual identity path length-exact: with
            # padding = (k - 1) // 2 a stride-1 conv returns L samples and a
            # stride-2 conv returns floor((L - 1) / 2) + 1, which is exactly what
            # the 1x1 downsample branch produces.  An even kernel would grow or
            # shrink the conv path by one sample and break the addition below.
            padding = (kernel_size - 1) // 2
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.downsample = (
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                if stride != 1 or in_channels != out_channels
                else None
            )

        def forward(self, x):
            identity = x if self.downsample is None else self.downsample(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out = out + identity
            return F.relu(out)

    class ECGNet(nn.Module):
        """Residual 1-D CNN with a clinical-feature fusion head."""

        def __init__(self, num_classes: int = len(ECG_CLASSES), n_features: int = len(FEATURE_ORDER), dropout: float = 0.3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            self.stage1 = nn.Sequential(
                ResNet1DBlock(64, 64, kernel_size=15, stride=1, dropout=dropout),
                ResNet1DBlock(64, 128, kernel_size=15, stride=2, dropout=dropout),
            )
            self.stage2 = nn.Sequential(
                ResNet1DBlock(128, 128, kernel_size=15, stride=1, dropout=dropout),
                ResNet1DBlock(128, 256, kernel_size=15, stride=2, dropout=dropout),
            )
            self.stage3 = nn.Sequential(
                ResNet1DBlock(256, 256, kernel_size=15, stride=1, dropout=dropout),
                ResNet1DBlock(256, 256, kernel_size=15, stride=2, dropout=dropout),
            )
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Linear(512 + 64, 256),  # 256 avg-pool + 256 max-pool + 64 features
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, signal, features):
            x = self.stem(signal)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            avg = torch.mean(x, dim=2)
            mx = torch.max(x, dim=2).values
            feat = self.feature_proj(features)
            merged = torch.cat([avg, mx, feat], dim=1)
            return self.head(merged)

else:  # pragma: no cover - torch-less installation

    class ECGNet:  # type: ignore[no-redef]
        """Placeholder raised at construction time when torch is missing."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate ECGNet")
