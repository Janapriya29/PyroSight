"""
src/preprocessing.py
--------------------
Image preprocessing pipeline for PyroSight.

Responsibilities:
  - Resize images to a common target size (default 224x224)
  - Convert to RGB, then to a float32 numpy array
  - Apply MobileNetV2-compatible normalisation:
      pixel = (pixel / 127.5) - 1.0   →  range [-1, 1]
  - Provide the four-to-two class mapping for binary classification

Binary label convention
-----------------------
  0  →  NO_SMOKE   (non_smoke, fog_non_smoke)
  1  →  SMOKE      (smoke, fog_smoke)
"""

from __future__ import annotations

import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE: tuple[int, int] = (224, 224)   # (width, height) for PIL.resize

# MobileNetV2 normalisation parameters  (scale to [-1, 1])
_MOBILENET_SCALE: float = 127.5
_MOBILENET_SHIFT: float = 1.0

# Four-class → binary mapping
#   key  : original folder name (lowercase)
#   value: (binary_label, binary_class_name)
FOUR_TO_BINARY: dict[str, tuple[int, str]] = {
    "smoke":         (1, "SMOKE"),
    "fog_smoke":     (1, "SMOKE"),
    "non_smoke":     (0, "NO_SMOKE"),
    "fog_non_smoke": (0, "NO_SMOKE"),
}

BINARY_CLASS_NAMES: dict[int, str] = {0: "NO_SMOKE", 1: "SMOKE"}


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(
    image_path: str,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Load a single image, resize it, and apply MobileNetV2 normalisation.

    Parameters
    ----------
    image_path:
        Absolute or relative path to a JPEG/PNG image file.
    target_size:
        (width, height) to resize to. Defaults to (224, 224).

    Returns
    -------
    np.ndarray
        Float32 array of shape (height, width, 3) with values in [-1, 1].

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    UnidentifiedImageError
        If the file cannot be opened as an image.
    ValueError
        If the resulting array has an unexpected shape.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, resample=Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)          # shape: (H, W, 3), [0, 255]
    arr = (arr / _MOBILENET_SCALE) - _MOBILENET_SHIFT  # scale to [-1, 1]

    if arr.shape != (target_size[1], target_size[0], 3):
        raise ValueError(
            f"Unexpected array shape {arr.shape} after preprocessing "
            f"(expected {(target_size[1], target_size[0], 3)})."
        )
    return arr


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def folder_to_binary_label(folder_name: str) -> int:
    """Map a four-class folder name to a binary integer label (0 or 1).

    Parameters
    ----------
    folder_name:
        One of: 'smoke', 'fog_smoke', 'non_smoke', 'fog_non_smoke'.

    Returns
    -------
    int
        1 for SMOKE, 0 for NO_SMOKE.

    Raises
    ------
    KeyError
        If *folder_name* is not a recognised category.
    """
    key = folder_name.lower().strip()
    if key not in FOUR_TO_BINARY:
        raise KeyError(
            f"Unknown category folder '{folder_name}'. "
            f"Expected one of: {list(FOUR_TO_BINARY.keys())}."
        )
    return FOUR_TO_BINARY[key][0]


def folder_to_class_name(folder_name: str) -> str:
    """Return the binary class name string for a four-class folder name.

    Parameters
    ----------
    folder_name:
        One of: 'smoke', 'fog_smoke', 'non_smoke', 'fog_non_smoke'.

    Returns
    -------
    str
        'SMOKE' or 'NO_SMOKE'.
    """
    key = folder_name.lower().strip()
    if key not in FOUR_TO_BINARY:
        raise KeyError(
            f"Unknown category folder '{folder_name}'. "
            f"Expected one of: {list(FOUR_TO_BINARY.keys())}."
        )
    return FOUR_TO_BINARY[key][1]
