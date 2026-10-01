"""
src/check_hybrid_model.py
--------------------------
Architecture validation script for the SmokeHybridModel.

Runs a single forward pass with dummy data, prints output shape,
parameter counts, model size, and architecture summary.

NO training is performed.

Usage:
    python -m src.check_hybrid_model
"""

from __future__ import annotations

import torch
from src.hybrid_model import (
    build_hybrid_model,
    count_parameters,
    model_size_mb,
    N_TRANSFORMER_LAYERS,
    N_HEADS,
    TOKEN_DIM,
    FFN_DIM,
    BACKBONE_CHANNELS,
    FUSION_DIM,
    N_TOKENS,
    SPATIAL_SIZE,
)

SEP = "=" * 60


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{SEP}")
    print("  SmokeHybridModel — Architecture Validation")
    print(SEP)
    print(f"  Device: {device}")

    # --- 1. Instantiate model ------------------------------------------
    print("\n[1] Instantiating model (freeze_backbone=True)...")
    model = build_hybrid_model(freeze_backbone=True)
    model = model.to(device)
    model.eval()
    print("    Model instantiated successfully.")

    # --- 2. Dummy input -----------------------------------------------
    print("\n[2] Creating dummy input [2, 3, 224, 224]...")
    dummy = torch.randn(2, 3, 224, 224, device=device)
    print(f"    Input shape : {list(dummy.shape)}")

    # --- 3. Forward pass ----------------------------------------------
    print("\n[3] Running forward pass...")
    with torch.no_grad():
        output = model(dummy)
    print(f"    Output shape: {list(output.shape)}")
    assert output.shape == (2, 1), (
        f"Expected output shape (2, 1), got {output.shape}"
    )
    print("    Output shape assertion PASSED.")

    # Check output is raw logits (not probabilities — Sigmoid not applied)
    print(f"    Output values (logits): {output.squeeze().tolist()}")

    # --- 4. Parameter counts ------------------------------------------
    params = count_parameters(model)
    print(f"\n[4] Parameter counts:")
    print(f"    Total     : {params['total']:>12,}")
    print(f"    Trainable : {params['trainable']:>12,}")
    print(f"    Frozen    : {params['frozen']:>12,}")

    # Per-module breakdown
    print(f"\n    Per-module trainable params:")
    for name, module in model.named_children():
        mod_trainable = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        mod_total = sum(p.numel() for p in module.parameters())
        print(f"      {name:<25} trainable={mod_trainable:>9,} / total={mod_total:>9,}")

    # --- 5. Model size ------------------------------------------------
    size = model_size_mb(model)
    print(f"\n[5] Approximate model size (fp32): {size} MB")

    # --- 6. Architecture summary (key dimensions) ---------------------
    print(f"\n[6] Architecture key dimensions:")
    print(f"    Input                      : 224 × 224 × 3")
    print(f"    Backbone                   : MobileNetV2 features[:-1]")
    print(f"    Backbone output            : {BACKBONE_CHANNELS} × {SPATIAL_SIZE} × {SPATIAL_SIZE}")
    print(f"    Spatial tokens             : {N_TOKENS}  ({SPATIAL_SIZE}×{SPATIAL_SIZE})")
    print(f"    CNN branch output          : {BACKBONE_CHANNELS}-dim vector")
    print(f"    Transformer token dim      : {TOKEN_DIM}")
    print(f"    Transformer layers         : {N_TRANSFORMER_LAYERS}")
    print(f"    Transformer attention heads: {N_HEADS}")
    print(f"    Transformer FFN dim        : {FFN_DIM}")
    print(f"    Transformer output (CLS)   : {TOKEN_DIM}-dim vector")
    print(f"    Fusion concat dim          : {BACKBONE_CHANNELS + TOKEN_DIM}  ({BACKBONE_CHANNELS} + {TOKEN_DIM})")
    print(f"    Fusion hidden dim          : {FUSION_DIM}")
    print(f"    Output                     : [B, 1]  (raw logit)")

    # --- 7. Full architecture printout --------------------------------
    print(f"\n[7] Full model architecture:")
    print(SEP)
    print(model)
    print(SEP)

    print(f"\nValidation: ALL CHECKS PASSED\n")


if __name__ == "__main__":
    main()
