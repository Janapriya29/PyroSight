# PyroSight
## Robust Lightweight CNN-Transformer for Early Wildfire Smoke Detection

PyroSight is a research-oriented deep learning system for binary wildfire smoke detection in still images. It combines a MobileNetV2 convolutional backbone with parallel CNN and Vision Transformer branches, fused into a lightweight binary classifier.

---

## Overview

Early detection of wildfire smoke is critical for rapid emergency response. Existing deep learning approaches often trade model size for accuracy, or perform well only on their training distribution. PyroSight investigates whether a lightweight hybrid architecture can achieve high accuracy while remaining robust to real-world image degradation and being practical for edge deployment.

**Key goals:**
- Lightweight inference (< 9 MB, < 1 ms/image on GPU)
- High accuracy with a low false alarm rate on the training distribution
- Improved robustness to brightness variation, contrast reduction, blur, and noise
- Honest evaluation of cross-domain generalisation limitations

---

## Architecture

```
Input: 224 × 224 × 3
    ↓
MobileNetV2 backbone (ImageNet pretrained, frozen)
    ↓  [B, 320, 7, 7]
    ├──────────────────────────────┐
    ↓                              ↓
CNN Branch                  Transformer Branch
DepthwiseSepConv (320→320)  Flatten → 49 tokens × 320
+ Global Average Pool       Linear projection → 128-dim
→ [B, 320]                  2× 4-head self-attention
                            CLS token → [B, 128]
    └──────────────┬───────────────┘
                   ↓
         Fusion Head: Linear(448 → 128 → 1)
                   ↓
         Output: SMOKE / NO SMOKE (sigmoid threshold 0.5)
```

---

## Robustness Training

The robust variant (`hybrid_robust_best.pt`) is trained with stochastic augmentations applied at training time:

- Random brightness scaling `U[0.75, 1.25]`
- Random contrast blending (up to 35% toward image mean)
- Random 3×3 average-pool blur (p = 0.4)
- Random Gaussian noise `N(0, σ)` with `σ ~ U[0, 0.04]` (p = 0.5)

---

## Results

### Internal test set — IIITDMJ\_Smoke (2,367 images)

| Model | Accuracy | F1 | False Alarm Rate | Params |
|---|:---:|:---:|:---:|---:|
| MobileNetV2 baseline | 94.34% | 0.9346 | 5.62% | 2,225,153 |
| CNN+Transformer hybrid | 98.56% | 0.9834 | 1.70% | 2,288,705 |
| Robust CNN+Transformer | **98.69%** | **0.9848** | **1.55%** | 2,288,705 |

Model size: 8.73 MB · Inference: ~0.68 ms/image (RTX 2050, batch=32)

### External evaluation — FIgLib (370 images, HPWREN cameras, Southern California)

External evaluation on the FIgLib wildfire camera dataset revealed a **significant domain gap**. Both models failed to generalise reliably to out-of-distribution HPWREN imagery. Cross-domain generalisation remains an open limitation of this project.

---

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app supports image upload (JPG/JPEG) and includes three built-in sample images.
Model inference runs on GPU if available, otherwise CPU.

---

## Project Structure

```
PyroSight/
├── app/
│   ├── streamlit_app.py        # Streamlit inference UI
│   └── samples/                # Three built-in sample images
├── src/
│   ├── preprocessing.py        # MobileNetV2 preprocessing pipeline
│   ├── data_loader.py          # IIITDMJ_Smoke dataset loader
│   ├── model.py                # MobileNetV2 baseline
│   ├── hybrid_model.py         # CNN+Transformer hybrid architecture
│   ├── train.py / train_hybrid.py / train_robust.py
│   ├── evaluate.py / evaluate_hybrid.py / evaluate_robust.py
│   ├── robustness_evaluation.py
│   ├── adapt_figlib.py         # FIgLib domain adaptation
│   └── evaluate_figlib*.py     # FIgLib evaluation scripts
├── data/                       # Dataset directory (not in Git — see below)
├── models/                     # Trained checkpoints (not in Git — see below)
├── results/                    # Evaluation outputs (not in Git — see below)
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

## Datasets and Model Weights

**Datasets and trained model weights are not included in this repository** due to their size.

- **IIITDMJ\_Smoke dataset** (~806 MB uncompressed): available from the original source.
- **FIgLib evaluation subset** (370 images, ~207 MB): downloaded from [hpwren.ucsd.edu/HPWREN-FIgLib/](https://hpwren.ucsd.edu/HPWREN-FIgLib/).  
  Attribution required: credit [http://hpwren.ucsd.edu/](http://hpwren.ucsd.edu/) in any derivative work.
- **Trained checkpoints** (`models/hybrid_robust_best.pt`, etc.): not committed to Git.

The `app/samples/` directory contains three small JPEG images for immediate UI testing without requiring the full dataset.

---

## Installation

```bash
# Create and activate a virtual environment (Python 3.11 recommended)
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

---

## Disclaimer

**PyroSight is a research and academic project.** It is not a certified wildfire warning system and must not be used as the sole basis for any emergency decision. Model predictions may be inaccurate, especially on images outside the training distribution. Always follow official wildfire guidance from local authorities.

---

## Reference

Training dataset:  
IIITDMJ\_Smoke — Indian Institute of Information Technology, Design and Manufacturing, Jabalpur.

External evaluation dataset:  
Dewangan, A. et al. "FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland Fire Smoke Detection." *Remote Sensing* 14(4), 1007, 2022. [https://doi.org/10.3390/rs14041007](https://doi.org/10.3390/rs14041007)
