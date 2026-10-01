# PyroSight — Data Directory

This directory holds datasets used for training, validation, and evaluation.  
**Do not commit raw data files or model outputs to Git.**  
Large files are covered by `.gitignore` rules at the project root.

---

## IIITDMJ_Smoke Dataset

### Storage (as found on disk)

The dataset is stored as a zip-within-zip:

```
data/
├── IIITDMJ_Smoke.zip                          # 787 MB  (outer wrapper)
└── IIITDMJ_Smoke/
    └── IIITDMJ_Smoke/
        └── IIITDMJ_Smoke.zip                  # 790 MB  (actual dataset)
```

The inner zip (`IIITDMJ_Smoke.zip`) contains all dataset files and must be
extracted before use.

---

### Internal Structure (inside the inner zip)

```
IIITDMJ_Smoke/
├── train/
│   ├── smoke/
│   ├── non_smoke/
│   ├── fog_smoke/
│   └── fog_non_smoke/
├── val/
│   ├── smoke/
│   ├── non_smoke/
│   ├── fog_smoke/
│   └── fog_non_smoke/
└── test/
    ├── smoke/
    ├── non_smoke/
    ├── fog_smoke/
    └── fog_non_smoke/
```

The dataset follows a standard ImageFolder layout: labels are encoded by
folder name. There are no separate CSV, JSON, TXT, or XML annotation files.

---

### Categories

| Category        | Description                                         |
|-----------------|-----------------------------------------------------|
| `smoke`         | Clear wildfire/combustion smoke                     |
| `non_smoke`     | No smoke present                                    |
| `fog_smoke`     | Smoke in foggy/hazy conditions                      |
| `fog_non_smoke` | Fog or haze without smoke (potential false-positive)|

---

### Image Counts per Split and Category

| Split   | smoke | non_smoke | fog_smoke | fog_non_smoke | Split Total |
|---------|------:|----------:|----------:|--------------:|------------:|
| train   | 3,827 |     5,456 |     4,278 |         5,353 |      18,914 |
| val     |   478 |       682 |       534 |           669 |       2,363 |
| test    |   479 |       682 |       536 |           670 |       2,367 |
| **Total** | **4,784** | **6,820** | **5,348** | **6,692** | **23,644** |

---

### Image Properties

| Property            | Value                                              |
|---------------------|----------------------------------------------------|
| Format              | JPEG (`.jpg`) — 100% of images                    |
| Dimensions — smoke / fog_smoke | 512 × 512 px (uniform)              |
| Dimensions — non_smoke / fog_non_smoke | Variable, approx. 300–400 px  |
| File size range     | 2.2 KB – 458.9 KB                                 |
| Average file size   | ~34.9 KB                                          |
| Total uncompressed  | ~806 MB                                           |
| Colour space        | RGB (standard JPEG)                               |

> Note: `non_smoke` and `fog_non_smoke` images have variable dimensions and
> will need to be resized to a common resolution before training.

---

### Metadata / Annotation Files

None found. The dataset contains only JPEG images organised into folders.
Labels are derived entirely from the directory structure.

---

### Duplicate Check

- **Duplicate filenames (basename):** 0 — all 23,644 filenames are unique
  across all categories and splits.
- Content-level duplicate detection (pixel-level hash) was not performed
  during this inspection pass.

---

### Inspection Date

Inspected: 2026-09-30  
Inspected by: Kiro (automated inspection — no files were modified)

---

## Phase 3 — Binary Classification Preprocessing

### Extracted Dataset Path

After extracting the inner zip, the dataset lives at:

```
data/IIITDMJ_Smoke/IIITDMJ_Smoke/IIITDMJ_Smoke/IIITDMJ_Smoke/
├── train/
├── val/
└── test/
```

This path is defined in `src/data_loader.py` as `DATASET_ROOT` and is
resolved automatically relative to the project root at runtime.

---

### Four-to-Binary Class Mapping

The four original folder categories are mapped to two binary labels:

| Original Folder  | Binary Label | Binary Class Name |
|------------------|:------------:|-------------------|
| `smoke`          | `1`          | `SMOKE`           |
| `fog_smoke`      | `1`          | `SMOKE`           |
| `non_smoke`      | `0`          | `NO_SMOKE`        |
| `fog_non_smoke`  | `0`          | `NO_SMOKE`        |

Labels are derived purely from folder names — no files are modified or moved.

---

### Binary Image Counts (verified on extracted dataset)

| Split | SMOKE (1) | NO_SMOKE (0) | Total  | SMOKE % |
|-------|----------:|-------------:|-------:|--------:|
| train |     8,105 |       10,809 | 18,914 |   42.9% |
| val   |     1,012 |        1,351 |  2,363 |   42.8% |
| test  |     1,015 |        1,352 |  2,367 |   42.9% |

The class distribution is consistent across splits (~43% SMOKE / ~57%
NO_SMOKE), with a mild imbalance toward the NO_SMOKE class.

---

### Preprocessing Pipeline

Implemented in `src/preprocessing.py`. Applied per image at load time
via `load_and_preprocess()`:

| Step | Detail |
|------|--------|
| Load | Open with Pillow, convert to RGB |
| Resize | Bilinear interpolation → **224 × 224 px** |
| To array | `numpy.float32`, shape `(224, 224, 3)` |
| Normalise | MobileNetV2: `pixel = (pixel / 127.5) − 1.0` → range `[−1, 1]` |

The normalisation is consistent with the MobileNetV2 pretraining convention
used by both TensorFlow/Keras (`tf.keras.applications.mobilenet_v2.preprocess_input`)
and torchvision-compatible implementations.

---

### Class Weights (computed from training split)

Balanced class weights are computed in `src/data_loader.get_class_weights()`
using the formula:

```
w_j = n_samples / (n_classes × n_samples_j)
```

| Class       | Label | Weight     |
|-------------|:-----:|------------|
| `NO_SMOKE`  |   0   | **0.8749** |
| `SMOKE`     |   1   | **1.1668** |

These weights should be applied to the loss function (or a weighted sampler)
during training to compensate for the class imbalance.

---

### Source Files

| File                    | Purpose                                             |
|-------------------------|-----------------------------------------------------|
| `src/preprocessing.py`  | Resize, normalise, and class-mapping utilities      |
| `src/data_loader.py`    | `SmokeDataset`, `load_split()`, `get_class_weights()`, `report_split_stats()` |
| `src/__init__.py`       | Package marker                                      |

---

### Dependencies Installed (Phase 3)

| Package       | Version | Purpose                          |
|---------------|---------|----------------------------------|
| Pillow        | 12.3.0  | Image loading and resizing       |
| numpy         | 2.4.6   | Array operations                 |
| scikit-learn  | 1.9.1   | Class weight formula (reference) |
| scipy         | 1.17.1  | scikit-learn dependency          |

All installed into `.venv` (Python 3.11.9).

---

### Phase 3 Date

Completed: 2026-09-30  
Completed by: Kiro (no dataset files were modified, renamed, or moved)
