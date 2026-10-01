"""
src/evaluate_figlib.py
-----------------------
External evaluation of PyroSight hybrid models on the FIgLib dataset.

Evaluates TWO models on the same FIgLib images:
  1. models/hybrid_best.pt          (standard hybrid, Phase 5.1)
  2. models/hybrid_robust_best.pt   (augmentation-trained, Phase 7.2)

Both models are evaluated without any modification.
FIgLib is used ONLY as an external cross-domain test — it was never seen
during training or validation.

Labels derived from filename (Phase 8.2 verified rule):
  <timestamp>_-<offset>.jpg  → NO_SMOKE (0)
  <timestamp>_+<offset>.jpg  |
  <timestamp>_%2B<offset>.jpg → SMOKE   (1)

Metrics computed per model (overall + per sequence):
  Accuracy, Precision (SMOKE), Recall (SMOKE), F1, False Alarm Rate,
  TP, TN, FP, FN, avg inference time (ms/image)

Outputs  (new files — existing results untouched)
-------
  results/figlib_hybrid_metrics.json
  results/figlib_robust_metrics.json
  results/figlib_comparison.png

Usage
-----
    python -m src.evaluate_figlib
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.hybrid_model import build_hybrid_model
from src.preprocessing import load_and_preprocess, TARGET_SIZE
from src.train_hybrid import HYBRID_CONFIG, RESULTS_DIR
from src.train_robust import ROBUST_CONFIG, ROBUST_CHECKPOINT_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
FIGLIB_DIR   = PROJECT_ROOT / "data" / "figlib_external" / "images"

HYBRID_CHECKPOINT = MODELS_DIR / "hybrid_best.pt"
ROBUST_CHECKPOINT = ROBUST_CHECKPOINT_PATH          # models/hybrid_robust_best.pt

FIGLIB_HYBRID_JSON = RESULTS_DIR / "figlib_hybrid_metrics.json"
FIGLIB_ROBUST_JSON = RESULTS_DIR / "figlib_robust_metrics.json"
FIGLIB_COMPARE_PNG = RESULTS_DIR / "figlib_comparison.png"

# Guard: must not overwrite existing per-split results
_PROTECTED = [
    "hybrid_test_metrics.json",
    "hybrid_confusion_matrix.png",
    "robust_model_robustness_results.json",
    "robustness_results.json",
]

BATCH_SIZE  = 32
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Sequence registry — same order as Phase 8.3
# ---------------------------------------------------------------------------
SEQUENCES = [
    "20221116-Willowfire-om-n-mobo-c",
    "20200806_SpringsFire_om-n-mobo-c",
    "20191006_FIRE_lp-n-mobo-c",
    "20190924_FIRE_ml-w-mobo-c",
    "20180806_Holy_sp-s-mobo-c",
]

# Regex: matches both '-', '+', and '%2B' as offset sign
_OFFSET_RE = re.compile(r"_(-|\+|%2B)(\d+)\.jpg$", re.IGNORECASE)


def _assign_label(filename: str) -> int | None:
    """0 = NO_SMOKE (negative offset), 1 = SMOKE (zero/positive offset)."""
    m = _OFFSET_RE.search(filename)
    if not m:
        return None
    return 0 if m.group(1) == "-" else 1


def _seq_of(filename: str) -> str:
    """Identify which sequence a file belongs to by timestamp range."""
    # Each sequence has a known Unix-epoch range derived from its filenames.
    # We identify sequence membership using timestamp prefix comparison.
    ts_match = re.match(r"^(\d+)_", filename)
    if not ts_match:
        return "unknown"
    ts = int(ts_match.group(1))
    # Timestamp boundaries (from first/last file of each sequence)
    ranges = {
        "20221116-Willowfire-om-n-mobo-c":   (1668652555, 1668657353),
        "20200806_SpringsFire_om-n-mobo-c":  (1596761735, 1596766475),
        "20191006_FIRE_lp-n-mobo-c":         (1570381681, 1570386481),
        "20190924_FIRE_ml-w-mobo-c":         (1569359957, 1569364757),
        "20180806_Holy_sp-s-mobo-c":         (1533583350, 1533588150),
    }
    for seq, (lo, hi) in ranges.items():
        if lo <= ts <= hi:
            return seq
    return "unknown"


# ---------------------------------------------------------------------------
# FIgLib PyTorch Dataset
# ---------------------------------------------------------------------------

class FIgLibDataset(Dataset):
    """Loads all FIgLib JPEGs, applies MobileNetV2 preprocessing.

    Labels are derived from filenames; no separate annotation file needed.
    """

    def __init__(self, image_dir: Path) -> None:
        self.image_dir = image_dir
        self.samples: list[tuple[Path, int, str]] = []  # (path, label, seq)
        skipped = 0

        for p in sorted(image_dir.glob("*.jpg")):
            label = _assign_label(p.name)
            if label is None:
                skipped += 1
                continue
            seq = _seq_of(p.name)
            self.samples.append((p, label, seq))

        if skipped:
            print(f"  [WARNING] Skipped {skipped} files with unparseable labels.")

        self.labels   = [s[1] for s in self.samples]
        self.seqs     = [s[2] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, _ = self.samples[idx]
        arr = load_and_preprocess(str(path), TARGET_SIZE)          # (224,224,3) float32
        tensor = torch.from_numpy(arr).permute(2, 0, 1)            # CHW
        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())
    acc  = (TP + TN) / len(labels)         if len(labels) > 0 else 0.0
    prec = TP / (TP + FP)                  if (TP + FP) > 0   else 0.0
    rec  = TP / (TP + FN)                  if (TP + FN) > 0   else 0.0
    f1   = 2*prec*rec / (prec+rec)         if (prec+rec) > 0  else 0.0
    far  = FP / (FP + TN)                  if (FP + TN) > 0   else 0.0
    return dict(accuracy=round(acc,6), precision=round(prec,6),
                recall=round(rec,6), f1=round(f1,6),
                false_alarm_rate=round(far,6),
                TP=TP, TN=TN, FP=FP, FN=FN)


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    checkpoint_path: Path,
    config:          dict,
    dataset:         FIgLibDataset,
    model_label:     str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ----------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"\n  Checkpoint : epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.4f}")

    # --- Rebuild model ------------------------------------------------
    model = build_hybrid_model(
        freeze_backbone=config["freeze_backbone"],
        dropout_p=config["fusion_dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- DataLoader ---------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- Inference ----------------------------------------------------
    all_preds   = []
    all_labels  = []
    times       = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            t0      = time.perf_counter()
            logits  = model(batch_x).squeeze(1)
            t1      = time.perf_counter()
            times.append((t1 - t0) / batch_x.size(0))
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.long().numpy().tolist())

    all_preds  = np.array(all_preds,  dtype=np.int64)
    all_labels = np.array(all_labels, dtype=np.int64)
    avg_ms     = round(float(np.mean(times)) * 1000, 4)

    # --- Overall metrics ----------------------------------------------
    overall = _metrics(all_preds, all_labels)

    print(f"\n  {'='*54}")
    print(f"  {model_label.upper()} — FIgLib EXTERNAL EVALUATION")
    print(f"  {'='*54}")
    print(f"  Total images      : {len(all_labels)}")
    print(f"  SMOKE             : {int((all_labels==1).sum())}  "
          f"NO_SMOKE: {int((all_labels==0).sum())}")
    print(f"  Accuracy          : {overall['accuracy']:.4f}  "
          f"({overall['accuracy']*100:.2f}%)")
    print(f"  Precision (SMOKE) : {overall['precision']:.4f}")
    print(f"  Recall    (SMOKE) : {overall['recall']:.4f}")
    print(f"  F1-score  (SMOKE) : {overall['f1']:.4f}")
    print(f"  False Alarm Rate  : {overall['false_alarm_rate']:.4f}")
    print(f"  TP={overall['TP']}  TN={overall['TN']}  "
          f"FP={overall['FP']}  FN={overall['FN']}")
    print(f"  Avg inference/img : {avg_ms:.3f} ms")

    # --- Per-sequence metrics -----------------------------------------
    seq_results = []
    print(f"\n  Per-sequence breakdown:")
    for seq in SEQUENCES:
        mask = np.array([dataset.seqs[i] == seq
                         for i in range(len(dataset.samples))], dtype=bool)
        if mask.sum() == 0:
            continue
        sm = _metrics(all_preds[mask], all_labels[mask])
        n_smoke    = int((all_labels[mask] == 1).sum())
        n_no_smoke = int((all_labels[mask] == 0).sum())
        seq_results.append({
            "sequence":       seq,
            "n_total":        int(mask.sum()),
            "n_smoke":        n_smoke,
            "n_no_smoke":     n_no_smoke,
            **sm,
        })
        print(f"    {seq[:45]:<45}  "
              f"Acc={sm['accuracy']:.3f}  F1={sm['f1']:.3f}  "
              f"FAR={sm['false_alarm_rate']:.3f}  "
              f"(n={int(mask.sum())}, S={n_smoke}, NS={n_no_smoke})")

    result = {
        "model":                      model_label,
        "checkpoint_path":            str(checkpoint_path),
        "checkpoint_epoch":           ckpt["epoch"],
        "dataset":                    "FIgLib external subset",
        "total_images":               len(all_labels),
        "smoke_count":                int((all_labels == 1).sum()),
        "no_smoke_count":             int((all_labels == 0).sum()),
        "accuracy":                   overall["accuracy"],
        "precision":                  overall["precision"],
        "recall":                     overall["recall"],
        "f1":                         overall["f1"],
        "false_alarm_rate":           overall["false_alarm_rate"],
        "TP":                         overall["TP"],
        "TN":                         overall["TN"],
        "FP":                         overall["FP"],
        "FN":                         overall["FN"],
        "avg_inference_ms_per_image": avg_ms,
        "per_sequence":               seq_results,
    }
    return result


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _save_comparison_plot(hybrid: dict, robust: dict) -> None:
    seqs       = [r["sequence"].replace("_", "\n").replace("-", "\n")
                  for r in hybrid["per_sequence"]]
    seq_labels = [s.split("\n")[0][:22] for s in seqs]   # abbreviated for axis

    metrics_cfg = [
        ("accuracy",         "Accuracy",          False),
        ("f1",               "F1",                False),
        ("false_alarm_rate", "False Alarm Rate",  True),   # lower is better
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "FIgLib External Evaluation — Hybrid vs Robust Hybrid\n"
        "(Cross-domain: Southern California wildfire cameras)",
        fontsize=12,
    )

    for ax, (key, label, lower_better) in zip(axes, metrics_cfg):
        h_overall = hybrid[key]
        r_overall = robust[key]
        h_seq     = [s[key] for s in hybrid["per_sequence"]]
        r_seq     = [s[key] for s in robust["per_sequence"]]

        x     = np.arange(len(seq_labels) + 1)   # +1 for Overall bar
        width = 0.35

        display_labels = ["Overall"] + seq_labels
        h_vals = [h_overall] + h_seq
        r_vals = [r_overall] + r_seq

        b1 = ax.bar(x - width/2, h_vals, width,
                    label="Hybrid (standard)", color="#1565C0", alpha=0.85)
        b2 = ax.bar(x + width/2, r_vals, width,
                    label="Hybrid (robust)",   color="#E65100", alpha=0.85)

        # Value labels
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)

        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.18)
        ax.axvline(x=0.5, color="grey", linestyle=":", linewidth=0.8)   # overall separator
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if lower_better:
            ax.set_ylabel("Score (lower is better)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGLIB_COMPARE_PNG, dpi=150)
    plt.close(fig)
    print(f"\nComparison plot saved : {FIGLIB_COMPARE_PNG}")


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_summary(hybrid: dict, robust: dict) -> None:
    col = 14
    sep = "-" * (38 + col * 5)
    print(f"\n{'='*(38 + col*5)}")
    print("  FIGLIB EXTERNAL EVALUATION — FINAL SUMMARY")
    print(f"{'='*(38 + col*5)}")
    header = (f"{'Metric':<24}"
              f"{'Hybrid':>{col}}"
              f"{'Robust':>{col}}"
              f"{'Delta':>{col}}")
    print(header)
    print(sep)
    rows = [
        ("Accuracy",          "accuracy",         False),
        ("Precision (SMOKE)", "precision",        False),
        ("Recall (SMOKE)",    "recall",           False),
        ("F1-score",          "f1",               False),
        ("False Alarm Rate",  "false_alarm_rate", True),
    ]
    for name, key, lower_better in rows:
        h = hybrid[key]
        r = robust[key]
        d = r - h
        improved = (d < 0) if lower_better else (d > 0)
        arrow = " +" if improved else (" -" if d != 0 else "  ")
        print(f"{name:<24}{h:>{col}.4f}{r:>{col}.4f}  {d:+.4f}{arrow}")
    print(sep)
    print(f"{'TP/TN/FP/FN':<24}"
          f"{str(hybrid['TP'])+'/'+str(hybrid['TN'])+'/'+str(hybrid['FP'])+'/'+str(hybrid['FN']):>{col}}"
          f"{str(robust['TP'])+'/'+str(robust['TN'])+'/'+str(robust['FP'])+'/'+str(robust['FN']):>{col}}")
    print(f"{'Inference (ms/img)':<24}"
          f"{hybrid['avg_inference_ms_per_image']:>{col}.3f}"
          f"{robust['avg_inference_ms_per_image']:>{col}.3f}")
    print(f"{'Total images':<24}{hybrid['total_images']:>{col}}")
    print(f"{'SMOKE / NO_SMOKE':<24}"
          f"{str(hybrid['smoke_count'])+' / '+str(hybrid['no_smoke_count']):>{col}}")
    print(f"{'='*(38 + col*5)}")
    print("  + = improvement over standard hybrid   - = regression")
    print("  FAR: lower is better")
    print(f"{'='*(38 + col*5)}")

    # Note on cross-domain interpretation
    print("\n  NOTE: These results reflect cross-domain performance.")
    print("  FIgLib (Southern California wildfire cameras) was NEVER")
    print("  used in training or validation. Differences from IIITDMJ_Smoke")
    print("  results are expected due to domain gap.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_figlib_evaluation() -> tuple[dict, dict]:
    # Guard: existing protected files must not be overwritten
    for name in _PROTECTED:
        out_path = RESULTS_DIR / name
        assert str(FIGLIB_HYBRID_JSON) != str(out_path), f"Would overwrite {name}!"
        assert str(FIGLIB_ROBUST_JSON) != str(out_path), f"Would overwrite {name}!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"FIgLib image dir: {FIGLIB_DIR}")

    # --- Build dataset once (shared between both models) --------------
    print("\nBuilding FIgLib dataset...")
    dataset = FIgLibDataset(FIGLIB_DIR)
    print(f"  Total images : {len(dataset)}")
    print(f"  SMOKE        : {sum(l==1 for l in dataset.labels)}")
    print(f"  NO_SMOKE     : {sum(l==0 for l in dataset.labels)}")

    # --- Evaluate model 1: standard hybrid ----------------------------
    print(f"\n{'='*60}")
    print("MODEL 1: Standard Hybrid  (hybrid_best.pt)")
    print(f"{'='*60}")
    hybrid_results = evaluate_model(
        HYBRID_CHECKPOINT, HYBRID_CONFIG, dataset, "Hybrid (standard)"
    )

    # --- Evaluate model 2: robust hybrid ------------------------------
    print(f"\n{'='*60}")
    print("MODEL 2: Robust Hybrid  (hybrid_robust_best.pt)")
    print(f"{'='*60}")
    robust_results = evaluate_model(
        ROBUST_CHECKPOINT, ROBUST_CONFIG, dataset, "Hybrid (robust)"
    )

    # --- Save JSON results --------------------------------------------
    with open(FIGLIB_HYBRID_JSON, "w") as f:
        json.dump(hybrid_results, f, indent=2)
    print(f"\nHybrid metrics saved  : {FIGLIB_HYBRID_JSON}")

    with open(FIGLIB_ROBUST_JSON, "w") as f:
        json.dump(robust_results, f, indent=2)
    print(f"Robust metrics saved  : {FIGLIB_ROBUST_JSON}")

    # --- Comparison plot ----------------------------------------------
    _save_comparison_plot(hybrid_results, robust_results)

    # --- Summary table ------------------------------------------------
    _print_summary(hybrid_results, robust_results)

    return hybrid_results, robust_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_figlib_evaluation()
