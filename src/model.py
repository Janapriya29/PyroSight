"""
src/model.py
------------
MobileNetV2 baseline for PyroSight binary smoke classification.

Architecture
------------
Input: 224 x 224 x 3  (MobileNetV2 normalised, range [-1, 1])
  -> MobileNetV2 backbone (ImageNet pretrained, frozen by default)
  -> Global Average Pooling  (built into MobileNetV2 classifier replacement)
  -> Dropout(p)
  -> Linear(1280, num_classes=1)
  -> Sigmoid  (applied externally via BCEWithLogitsLoss during training)

The backbone's final classifier layer is replaced so that only the new
head is trained in the frozen phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_mobilenetv2_baseline(
    dropout_p: float = 0.3,
    freeze_backbone: bool = True,
    num_classes: int = 1,
) -> nn.Module:
    """Create a MobileNetV2-based binary classifier.

    Parameters
    ----------
    dropout_p:
        Dropout probability applied before the final linear layer. Default 0.3.
    freeze_backbone:
        If True, all MobileNetV2 backbone parameters are frozen so only the
        classification head is trained. Default True.
    num_classes:
        Output dimension. 1 for binary classification with BCEWithLogitsLoss.

    Returns
    -------
    nn.Module
        The assembled model, ready to be moved to a device.
    """
    # Load pretrained backbone
    backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all backbone parameters if requested
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    # Replace the default classifier head
    # backbone.classifier = [Dropout(0.2), Linear(1280, 1000)]
    # We swap it for our custom head
    in_features = backbone.classifier[1].in_features  # 1280
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )

    return backbone


# ---------------------------------------------------------------------------
# Parameter / size utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }


def model_size_mb(model: nn.Module) -> float:
    """Return approximate model size in MB (float32 parameters only)."""
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params * 4 / (1024 ** 2), 2)  # 4 bytes per float32
