"""
src/data_loader.py
------------------
Dataset loader for PyroSight binary smoke classification.

Provides:
  - SmokeDataset  : iterable dataset that walks a split directory,
                    maps four classes to binary labels, and optionally
                    applies preprocessing on __getitem__.
  - get_class_weights() : compute balanced class weights from training labels.
  - load_split()         : convenience function that returns a SmokeDataset
                           for a named split (train / val / test).
  - report_split_stats() : print binary class counts and class weights.

Usage example
-------------
    from src.data_loader import load_split, report_split_stats, get_class_weights

    train_ds = load_split("train")
    val_ds   = load_split("val")
    test_ds  = load_split("test")

    report_split_stats(train_ds, "train")
    weights = get_class_weights(train_ds)
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from src.preprocessing import (
    FOUR_TO_BINARY,
    BINARY_CLASS_NAMES,
    folder_to_binary_label,
    load_and_preprocess,
    TARGET_SIZE,
)

# ---------------------------------------------------------------------------
# Dataset path configuration
# ---------------------------------------------------------------------------

# Resolve the project root as the directory two levels above this file
# (src/ → project_root/)
_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent

# Actual path on disk after extraction
DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "IIITDMJ_Smoke"
    / "IIITDMJ_Smoke"
    / "IIITDMJ_Smoke"
    / "IIITDMJ_Smoke"
)

VALID_SPLITS = ("train", "val", "test")
VALID_CATEGORIES = tuple(FOUR_TO_BINARY.keys())
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# SmokeDataset
# ---------------------------------------------------------------------------

class SmokeDataset:
    """Lightweight dataset that enumerates images and binary labels for one
    split of the IIITDMJ_Smoke dataset.

    Parameters
    ----------
    split_dir:
        Path to a split directory (e.g. …/IIITDMJ_Smoke/train).
    preprocess:
        If True, __getitem__ returns a preprocessed float32 numpy array.
        If False, __getitem__ returns the raw PIL Image (useful for EDA).
    target_size:
        Resize target passed to load_and_preprocess. Default: (224, 224).

    Attributes
    ----------
    samples : list[tuple[Path, int]]
        All (image_path, binary_label) pairs in the split.
    labels : list[int]
        Binary labels only (parallel to samples).
    counts : Counter
        {binary_label: count} over the split.
    """

    def __init__(
        self,
        split_dir: str | Path,
        preprocess: bool = True,
        target_size: tuple[int, int] = TARGET_SIZE,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.preprocess = preprocess
        self.target_size = target_size

        if not self.split_dir.is_dir():
            raise NotADirectoryError(
                f"Split directory not found: {self.split_dir}"
            )

        self.samples: list[tuple[Path, int]] = []
        self._build_sample_list()

        self.labels: list[int] = [label for _, label in self.samples]
        self.counts: Counter = Counter(self.labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sample_list(self) -> None:
        """Walk each category subfolder and collect (path, binary_label) pairs."""
        for category_dir in sorted(self.split_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            folder_name = category_dir.name.lower()
            if folder_name not in FOUR_TO_BINARY:
                # skip unexpected folders silently
                continue
            binary_label = folder_to_binary_label(folder_name)
            for img_file in sorted(category_dir.iterdir()):
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_file, binary_label))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return (array_or_image, binary_label) for the sample at *idx*.

        Returns
        -------
        tuple[np.ndarray | PIL.Image.Image, int]
            If preprocess=True : (float32 array [H, W, 3], label)
            If preprocess=False: (PIL.Image.Image, label)
        """
        img_path, label = self.samples[idx]
        if self.preprocess:
            return load_and_preprocess(str(img_path), self.target_size), label
        else:
            from PIL import Image
            return Image.open(img_path).convert("RGB"), label

    def __repr__(self) -> str:
        smoke    = self.counts.get(1, 0)
        no_smoke = self.counts.get(0, 0)
        return (
            f"SmokeDataset(split='{self.split_dir.name}', "
            f"total={len(self)}, "
            f"SMOKE={smoke}, NO_SMOKE={no_smoke})"
        )


# ---------------------------------------------------------------------------
# Class weight calculation
# ---------------------------------------------------------------------------

def get_class_weights(dataset: SmokeDataset) -> dict[int, float]:
    """Compute balanced class weights from the training (or any) split.

    Uses the sklearn-compatible formula:
        w_j = n_samples / (n_classes * n_samples_j)

    This produces weights that compensate for class imbalance when passed
    to a loss function or sample weighter.

    Parameters
    ----------
    dataset:
        A SmokeDataset instance (typically the training split).

    Returns
    -------
    dict[int, float]
        {0: weight_no_smoke, 1: weight_smoke}
    """
    labels = np.array(dataset.labels, dtype=np.int64)
    n_samples = len(labels)
    n_classes = len(BINARY_CLASS_NAMES)

    weights: dict[int, float] = {}
    for cls in sorted(BINARY_CLASS_NAMES.keys()):
        n_cls = int((labels == cls).sum())
        if n_cls == 0:
            weights[cls] = 0.0
        else:
            weights[cls] = round(n_samples / (n_classes * n_cls), 6)
    return weights


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_split(
    split: str,
    dataset_root: Optional[str | Path] = None,
    preprocess: bool = True,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> SmokeDataset:
    """Load one split of the IIITDMJ_Smoke dataset as a SmokeDataset.

    Parameters
    ----------
    split:
        One of 'train', 'val', or 'test'.
    dataset_root:
        Override the default DATASET_ROOT path (optional).
    preprocess:
        Whether __getitem__ returns preprocessed arrays. Default: True.
    target_size:
        Resize target. Default: (224, 224).

    Returns
    -------
    SmokeDataset
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. Expected one of: {VALID_SPLITS}."
        )
    root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
    split_dir = root / split
    return SmokeDataset(split_dir, preprocess=preprocess, target_size=target_size)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_split_stats(dataset: SmokeDataset, split_name: str = "") -> None:
    """Print binary class counts and class balance for a split.

    Parameters
    ----------
    dataset:
        A SmokeDataset instance.
    split_name:
        Label used in the printed header.
    """
    label = split_name or dataset.split_dir.name
    total = len(dataset)
    smoke    = dataset.counts.get(1, 0)
    no_smoke = dataset.counts.get(0, 0)

    print(f"\n{'='*50}")
    print(f"  Split : {label.upper()}")
    print(f"{'='*50}")
    print(f"  Total images : {total:>6}")
    print(f"  SMOKE        : {smoke:>6}  ({smoke/total*100:.1f}%)")
    print(f"  NO_SMOKE     : {no_smoke:>6}  ({no_smoke/total*100:.1f}%)")
    print(f"{'='*50}")
