"""Compact 2D CNN for chest X-ray classification.

Four convolutional stages with batch normalisation and max pooling, followed by
global pooling and a clinical-feature fusion head.  The architecture is small
enough to train on modest hardware while still learning the focal-opacity
patterns that separate the lung-disease classes.
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

from app.ml.xray.features import FEATURE_ORDER

XRAY_CLASSES: List[str] = [
    "NORMAL",
    "PNEUMONIA",
    "CONSOLIDATION",
    "EFFUSION",
    "TUBERCULOSIS",
    "COVID19",
]

CLASS_DESCRIPTIONS = {
    "NORMAL": "No acute cardiopulmonary abnormality identified.",
    "PNEUMONIA": "Findings consistent with pneumonia - patchy parenchymal opacity.",
    "CONSOLIDATION": "Dense air-space consolidation suggesting infective or inflammatory change.",
    "EFFUSION": "Pleural effusion - blunting of the costophrenic angles with basal opacity.",
    "TUBERCULOSIS": "Upper-zone/apical abnormality suggestive of pulmonary tuberculosis.",
    "COVID19": "Peripheral, bilateral ground-glass opacity pattern typical of COVID-19 pneumonia.",
}

CLASS_SEVERITY = {
    "NORMAL": "normal",
    "PNEUMONIA": "moderate",
    "CONSOLIDATION": "moderate",
    "EFFUSION": "moderate",
    "TUBERCULOSIS": "severe",
    "COVID19": "moderate",
}


if TORCH_AVAILABLE:  # pragma: no cover

    class ConvBlock(nn.Module):
        """Conv -> BN -> ReLU -> MaxPool block."""

        def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

        def forward(self, x):
            return self.pool(F.relu(self.bn(self.conv(x))))

    class XrayCNN(nn.Module):
        """Chest radiograph classifier with clinical-feature fusion."""

        def __init__(self, num_classes: int = len(XRAY_CLASSES), n_features: int = len(FEATURE_ORDER), dropout: float = 0.3):
            super().__init__()
            self.stem = ConvBlock(1, 32)
            self.block1 = ConvBlock(32, 64)
            self.block2 = ConvBlock(64, 128)
            self.block3 = ConvBlock(128, 256, pool=False)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Linear(256 + 64, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, image, features):
            x = self.stem(image)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x).flatten(start_dim=1)
            feat = self.feature_proj(features)
            merged = torch.cat([x, feat], dim=1)
            return self.head(merged)

else:  # pragma: no cover

    class XrayCNN:  # type: ignore[no-redef]
        """Placeholder raised at construction time when torch is missing."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate XrayCNN")
