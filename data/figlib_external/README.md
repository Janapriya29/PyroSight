# FIgLib External Evaluation Subset
## PyroSight — Phase 8.3

**This data is for EXTERNAL TESTING ONLY.**
It must never be used for model training or fine-tuning.

---

## Source

**Dataset:** HPWREN Fire Ignition image Library (FIgLib)  
**Maintained by:** Hans-Werner Braun, HPWREN, UC San Diego  
**Official page:** https://hpwren.ucsd.edu/HPWREN-FIgLib/  
**CDN index:** https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html  
**Attribution required:** All use must credit **https://www.hpwren.ucsd.edu/**

**Reference:**
> Dewangan, A.; Pande, Y.; Braun, H.-W.; Vernon, F.; Perez, I.; Altintas, I.;
> Cottrell, G.W.; Nguyen, M.H. FIgLib & SmokeyNet: Dataset and Deep Learning
> Model for Real-Time Wildland Fire Smoke Detection.
> *Remote Sens.* 2022, 14(4), 1007. https://doi.org/10.3390/rs14041007

---

## Downloaded Sequences

Five sequences were selected and verified in Phase 8.2 against the official
FIgLib index, `night_fires.txt`, and `mislabeled_fires.txt` before download.
All are daytime, colour-camera (`-mobo-c`) sequences from Southern California.

| Sequence | Camera | Fire date | Images | SMOKE | NO_SMOKE |
|---|---|---|:---:|:---:|:---:|
| `20221116-Willowfire-om-n-mobo-c` | `om-n-mobo-c` | 2022-11-16 | 80 | 41 | 39 |
| `20200806_SpringsFire_om-n-mobo-c` | `om-n-mobo-c` | 2020-08-06 | 65 | 39 | 26 |
| `20191006_FIRE_lp-n-mobo-c` | `lp-n-mobo-c` | 2019-10-06 | 73 | 35 | 38 |
| `20190924_FIRE_ml-w-mobo-c` | `ml-w-mobo-c` | 2019-09-24 | 80 | 40 | 40 |
| `20180806_Holy_sp-s-mobo-c` | `sp-s-mobo-c` | 2018-08-06 | 72 | 37 | 35 |
| **Total** | | | **370** | **192** | **178** |

---

## File Naming Convention

Every image filename encodes its temporal offset from the moment of visible
smoke/plume appearance:

```
<unix_epoch_timestamp>_<sign><offset_seconds>.jpg
```

Where `<sign>` is:
- `-` → image was captured **before** visible smoke → label **NO_SMOKE (0)**
- `+` (or `%2B`, the URL-encoded form) → image was captured **at or after**
  visible smoke appearance → label **SMOKE (1)**

Examples:
```
1668652555_-02398.jpg   → 2398 s before ignition → NO_SMOKE (0)
1668654953_%2B00000.jpg → exactly at ignition     → SMOKE    (1)
1668657353_%2B02400.jpg → 2400 s after ignition   → SMOKE    (1)
```

**Note:** The CDN serves positive-offset filenames with `%2B` (percent-encoded `+`).
Both `+` and `%2B` are treated as SMOKE by the label parser in `src/download_figlib.py`.

---

## Total Image Counts

| Metric | Value |
|---|---|
| Total images | 370 |
| SMOKE (label 1) | 192 (51.9%) |
| NO_SMOKE (label 0) | 178 (48.1%) |
| Duplicate filenames | 0 |
| Corrupt / unreadable files | 0 |
| Image format | JPEG only |
| Width range | 2048 – 3072 px |
| Height range | 1536 – 2048 px |

---

## Labelling Rule

Labels are derived **entirely from the filename** — no manual annotation required:

```python
# sign '-'            → NO_SMOKE (0)
# sign '+' or '%2B'   → SMOKE    (1)
```

This rule is implemented in `src/download_figlib.py :: assign_label()`.

---

## Preprocessing for PyroSight Inference

Before inference, each image must go through the standard pipeline defined in
`src/preprocessing.py`:

1. Open with Pillow and convert to RGB.
2. Resize to **224 × 224 px** (bilinear interpolation).
3. Apply MobileNetV2 normalisation: `pixel = (pixel / 127.5) − 1.0` → `[-1, 1]`.

FIgLib images are full-resolution (up to 3072 × 2048 px); the resize step handles
the scale difference from the IIITDMJ_Smoke training images (300–512 px).

---

## How This Differs from the Training Data (IIITDMJ_Smoke)

| Property | IIITDMJ_Smoke (train/val/test) | FIgLib subset (external) |
|---|---|---|
| Geography | India | Southern California, USA |
| Camera | Consumer / surveillance | Fixed mountaintop HPWREN cameras |
| Image size | 300–512 px | 2048–3072 px |
| Smoke type | Clear smoke, foggy smoke | Wildland fire ignition plumes |
| Label source | Folder name | Filename timestamp offset |
| Use in PyroSight | Training + held-out test | External evaluation only |

The domain gap makes FIgLib a meaningful test of cross-domain generalisation,
not just held-out performance on the same distribution.

---

## Download Date

Downloaded: 2026-09-30  
All 370 files verified (integrity check: PASS).

*Content was rephrased for compliance with licensing restrictions.*
