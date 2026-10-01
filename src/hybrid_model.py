"""
src/hybrid_model.py
-------------------
CNN + Transformer hybrid for PyroSight binary wildfire smoke classification.

Architecture overview
---------------------
Input: 224 × 224 × 3  (MobileNetV2 normalised, range [-1, 1])
    │
    ▼
MobileNetV2 feature extractor  (ImageNet pretrained, backbone freezing configurable)
    │  Output: [B, 320, 7, 7]   (spatial feature map, before global pooling)
    │
    ├─────────────────────────────┐
    │                             │
    ▼                             ▼
Branch 1 — CNN               Branch 2 — Transformer
────────────────             ──────────────────────
DepthwiseSeparableConv       Flatten 7×7 → 49 tokens
(320→320, 3×3)               Project 320 → 128 (token_dim)
BN + ReLU                    Add learnable positional embedding
AdaptiveAvgPool → [B, 320]   Add [CLS] token → 50 tokens × 128
                             TransformerEncoder
                               - 2 layers
                               - 4 attention heads
                               - FFN dim 256
                               - Dropout 0.1
                             Take CLS token → [B, 128]
    │                             │
    └──────────────┬──────────────┘
                   ▼
           Fusion head
           ─────────────
           Concat [B, 320 + 128] → [B, 448]
           Linear(448 → 128) + ReLU + Dropout(0.3)
           Linear(128 → 1)
                   │
                   ▼
           Output: [B, 1]  (logit — apply Sigmoid externally)

Design choices
--------------
- Backbone frozen by default; can be unfrozen for fine-tuning.
- DepthwiseSeparable conv keeps the CNN branch parameter-efficient.
- Token dim = 128 keeps the Transformer memory footprint small (7×7 = 49 tokens).
- 2 Transformer layers × 4 heads is sufficient to model spatial dependencies
  across the 49 spatial positions without overfitting on this dataset size.
- CLS token pooling is standard and avoids an extra mean-pool op.
- No Sigmoid inside the model; BCEWithLogitsLoss handles it during training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


# ---------------------------------------------------------------------------
# Hyperparameters (kept at module level for easy reference / override)
# ---------------------------------------------------------------------------

# Backbone output channels at 224×224 input (MobileNetV2 final feature map)
BACKBONE_CHANNELS: int = 320      # channels in the 7×7 spatial map
SPATIAL_SIZE:      int = 7        # height = width of the spatial map
N_TOKENS:          int = SPATIAL_SIZE * SPATIAL_SIZE   # 49 spatial tokens

# Transformer branch
TOKEN_DIM:         int = 128      # projected dimension for each token
N_TRANSFORMER_LAYERS: int = 2     # number of TransformerEncoderLayer blocks
N_HEADS:           int = 4        # multi-head self-attention heads
FFN_DIM:           int = 256      # feed-forward inner dimension
TRANSFORMER_DROPOUT: float = 0.1  # dropout inside Transformer

# Fusion head
FUSION_DIM:        int = 128      # intermediate dim in fusion MLP
FUSION_DROPOUT:    float = 0.3


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Lightweight depthwise-separable 3×3 conv block.

    depthwise  : Conv2d(C, C, 3×3, groups=C, padding=1)
    pointwise  : Conv2d(C, C_out, 1×1)
    normalisation + activation applied after each step.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False,
        )
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class CNNBranch(nn.Module):
    """CNN branch operating on the MobileNetV2 spatial feature map.

    Input : [B, BACKBONE_CHANNELS, 7, 7]
    Output: [B, BACKBONE_CHANNELS]  (after global average pooling)
    """

    def __init__(self, in_channels: int = BACKBONE_CHANNELS) -> None:
        super().__init__()
        self.dws_conv = DepthwiseSeparableConv(in_channels, in_channels)
        self.gap      = nn.AdaptiveAvgPool2d(1)   # → [B, C, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dws_conv(x)          # [B, 320, 7, 7]
        x = self.gap(x)               # [B, 320, 1, 1]
        x = x.flatten(1)              # [B, 320]
        return x


class TransformerBranch(nn.Module):
    """Transformer branch operating on the MobileNetV2 spatial feature map.

    Steps:
      1. Flatten 7×7 spatial positions → 49 tokens of dim BACKBONE_CHANNELS.
      2. Project each token to TOKEN_DIM via a linear layer.
      3. Prepend a learnable [CLS] token.
      4. Add learnable positional embeddings.
      5. Pass through N_TRANSFORMER_LAYERS TransformerEncoderLayer blocks.
      6. Return the CLS token representation.

    Input : [B, BACKBONE_CHANNELS, 7, 7]
    Output: [B, TOKEN_DIM]
    """

    def __init__(
        self,
        in_channels:       int   = BACKBONE_CHANNELS,
        token_dim:         int   = TOKEN_DIM,
        n_tokens:          int   = N_TOKENS,
        n_layers:          int   = N_TRANSFORMER_LAYERS,
        n_heads:           int   = N_HEADS,
        ffn_dim:           int   = FFN_DIM,
        dropout:           float = TRANSFORMER_DROPOUT,
    ) -> None:
        super().__init__()
        self.n_tokens  = n_tokens
        self.token_dim = token_dim

        # 1. Linear projection: backbone_channels → token_dim
        self.token_proj = nn.Linear(in_channels, token_dim)

        # 2. Learnable [CLS] token (1 × token_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3. Learnable positional embedding: (1 + n_tokens) positions
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, 1 + n_tokens, token_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,   # expects [B, seq_len, d_model]
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Flatten spatial dims: [B, C, H, W] → [B, H*W, C]
        x = x.flatten(2)              # [B, 320, 49]
        x = x.permute(0, 2, 1)        # [B, 49, 320]

        # Project to token_dim
        x = self.token_proj(x)        # [B, 49, 128]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, 128]
        x   = torch.cat([cls, x], dim=1)          # [B, 50, 128]

        # Add positional embedding
        x = x + self.pos_embed        # [B, 50, 128]

        # Transformer encoding
        x = self.transformer(x)       # [B, 50, 128]

        # Return CLS token
        cls_out = x[:, 0, :]          # [B, 128]
        return cls_out


class FusionHead(nn.Module):
    """Fusion MLP that combines CNN and Transformer feature vectors.

    Input : cnn_feat  [B, cnn_dim]
            xfmr_feat [B, xfmr_dim]
    Output: [B, 1]  (logit)
    """

    def __init__(
        self,
        cnn_dim:    int   = BACKBONE_CHANNELS,   # 320
        xfmr_dim:   int   = TOKEN_DIM,           # 128
        fusion_dim: int   = FUSION_DIM,          # 128
        dropout_p:  float = FUSION_DROPOUT,
    ) -> None:
        super().__init__()
        concat_dim = cnn_dim + xfmr_dim          # 448
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fusion_dim, 1),
        )

    def forward(
        self,
        cnn_feat:  torch.Tensor,
        xfmr_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([cnn_feat, xfmr_feat], dim=1)   # [B, 448]
        return self.mlp(x)                              # [B, 1]


# ---------------------------------------------------------------------------
# Full hybrid model
# ---------------------------------------------------------------------------

class SmokeHybridModel(nn.Module):
    """CNN + Transformer hybrid for binary wildfire smoke classification.

    Parameters
    ----------
    freeze_backbone:
        If True, all MobileNetV2 backbone parameters are frozen.
        Default True (only head branches are trained initially).
    dropout_p:
        Dropout probability in the fusion head. Default 0.3.

    Forward
    -------
    Input : torch.Tensor  [B, 3, 224, 224]
    Output: torch.Tensor  [B, 1]   (raw logit — no Sigmoid)
    """

    def __init__(
        self,
        freeze_backbone: bool  = True,
        dropout_p:       float = FUSION_DROPOUT,
    ) -> None:
        super().__init__()

        # ---- Shared backbone ----
        _mobilenet = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1
        )
        # Extract only the feature layers (drop classifier and adaptive pool)
        # MobileNetV2.features outputs [B, 1280, 7, 7] at 224×224 input.
        # We tap at the last inverted residual block output:
        #   features[:-1] ends at InvertedResidual with output channels=320.
        # features[-1]  is a Conv-BN-ReLU6 block that expands 320 → 1280.
        # We keep features[:-1] to get the richer 320-channel spatial map.
        self.backbone = _mobilenet.features[:-1]  # output: [B, 320, 7, 7]

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---- Dual branches ----
        self.cnn_branch         = CNNBranch(in_channels=BACKBONE_CHANNELS)
        self.transformer_branch = TransformerBranch(
            in_channels=BACKBONE_CHANNELS,
            token_dim=TOKEN_DIM,
            n_tokens=N_TOKENS,
            n_layers=N_TRANSFORMER_LAYERS,
            n_heads=N_HEADS,
            ffn_dim=FFN_DIM,
            dropout=TRANSFORMER_DROPOUT,
        )

        # ---- Fusion head ----
        self.fusion_head = FusionHead(
            cnn_dim=BACKBONE_CHANNELS,
            xfmr_dim=TOKEN_DIM,
            fusion_dim=FUSION_DIM,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        torch.Tensor  [B, 1]  — raw logit
        """
        # Shared feature extraction
        feat = self.backbone(x)              # [B, 320, 7, 7]

        # Dual branches (parallel)
        cnn_feat  = self.cnn_branch(feat)         # [B, 320]
        xfmr_feat = self.transformer_branch(feat) # [B, 128]

        # Fusion and classification
        logit = self.fusion_head(cnn_feat, xfmr_feat)  # [B, 1]
        return logit


# ---------------------------------------------------------------------------
# Parameter / size utilities  (mirrors src/model.py interface)
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
    return round(total_params * 4 / (1024 ** 2), 2)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_hybrid_model(
    freeze_backbone: bool  = True,
    dropout_p:       float = FUSION_DROPOUT,
) -> SmokeHybridModel:
    """Instantiate and return a SmokeHybridModel.

    Parameters
    ----------
    freeze_backbone:
        Freeze MobileNetV2 backbone weights. Default True.
    dropout_p:
        Dropout in the fusion head. Default 0.3.

    Returns
    -------
    SmokeHybridModel
    """
    return SmokeHybridModel(
        freeze_backbone=freeze_backbone,
        dropout_p=dropout_p,
    )
