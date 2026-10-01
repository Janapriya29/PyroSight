"""
app/streamlit_app.py
--------------------
PyroSight — Lightweight Wildfire Smoke Detection
Streamlit UI with CNN + Transformer hybrid inference (Phase 10.2).

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/ when running from project root or app/
# ---------------------------------------------------------------------------
_APP_DIR     = Path(__file__).resolve().parent          # .../app/
_PROJECT_ROOT = _APP_DIR.parent                         # .../PyroSight/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.hybrid_model import build_hybrid_model         # noqa: E402
from src.train_robust import ROBUST_CONFIG              # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHECKPOINT   = _PROJECT_ROOT / "models" / "hybrid_robust_best.pt"
_TARGET_SIZE  = (224, 224)          # (W, H) — matches training pipeline
_SCALE        = 127.5               # MobileNetV2 normalisation
_THRESHOLD    = 0.5                 # fixed; matches evaluation threshold

# Sample images shipped with the app
_SAMPLES_DIR = _APP_DIR / "samples"
_SAMPLE_OPTIONS: dict[str, Path] = {
    "Clear Outdoor Scene":  _SAMPLES_DIR / "clear.jpg",
    "Smoke Outdoor Scene":  _SAMPLES_DIR / "smoke.jpg",
    "Foggy Outdoor Scene":  _SAMPLES_DIR / "fog.jpg",
}

# ---------------------------------------------------------------------------
# Model loading — cached so it runs once per session, not per interaction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PyroSight model…")
def _load_model() -> tuple[torch.nn.Module, torch.device]:
    """Load hybrid_robust_best.pt and return (model, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(_CHECKPOINT, map_location=device)
    model  = build_hybrid_model(
        freeze_backbone=ROBUST_CONFIG["freeze_backbone"],
        dropout_p=ROBUST_CONFIG["fusion_dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, device


# ---------------------------------------------------------------------------
# Preprocessing — operates on an in-memory PIL Image
# ---------------------------------------------------------------------------

def _preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a [1, 3, 224, 224] float32 tensor in [-1, 1].

    Mirrors the pipeline in src/preprocessing.py:
      1. Convert to RGB
      2. Resize to 224 x 224 (bilinear)
      3. Normalise: pixel = (pixel / 127.5) - 1.0
      4. HWC numpy → CHW tensor → add batch dim
    """
    img  = pil_image.convert("RGB").resize(_TARGET_SIZE, resample=Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32)          # (224, 224, 3)  [0, 255]
    arr  = (arr / _SCALE) - 1.0                     # (224, 224, 3)  [-1, 1]
    t    = torch.from_numpy(arr).permute(2, 0, 1)   # (3, 224, 224)
    return t.unsqueeze(0)                            # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    model:  torch.nn.Module,
    device: torch.device,
    tensor: torch.Tensor,
) -> tuple[str, float]:
    """Return (label, probability) for one image tensor.

    label       : "SMOKE" or "NO SMOKE"
    probability : float in [0, 1] — probability of SMOKE
    """
    with torch.no_grad():
        logit = model(tensor.to(device)).squeeze()
        prob  = float(torch.sigmoid(logit).cpu().item())
    label = "SMOKE" if prob >= _THRESHOLD else "NO SMOKE"
    return label, prob


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PyroSight",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔥 PyroSight")
st.markdown("#### Lightweight Wildfire Smoke Detection")
st.markdown(
    "A lightweight CNN-Transformer system for wildfire smoke detection."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About PyroSight")
    st.markdown(
        "A research prototype for early wildfire smoke detection using a "
        "hybrid deep learning architecture."
    )
    st.divider()

    # ---- Model Information ----
    st.subheader("Model Information")
    st.markdown(
        """
        | Property | Value |
        |---|---|
        | Backbone | MobileNetV2 (ImageNet) |
        | Architecture | CNN + Transformer hybrid |
        | Input size | 224 × 224 px |
        | Classification | Binary (SMOKE / NO SMOKE) |
        | Transformer layers | 2 × 4-head self-attention |
        | Fusion head | Linear 448 → 128 → 1 |
        | Trainable params | 476,993 |
        """
    )
    st.divider()

    # ---- Performance ----
    st.subheader("Performance")
    st.markdown("**Internal test set (IIITDMJ\\_Smoke)**")
    st.markdown(
        """
        | Metric | Value |
        |---|---|
        | Accuracy | 98.69% |
        | False Alarm Rate | 1.55% |
        | F1-score (SMOKE) | 98.48% |
        | Test images | 2,367 |

        *Measured on a held-out split of the IIITDMJ\\_Smoke dataset.
        Results reflect in-distribution performance only.*
        """
    )
    st.divider()

    # ---- Known Limitations ----
    st.subheader("Known Limitations")
    st.warning(
        "**Cross-domain generalisation is limited.**\n\n"
        "External evaluation on the FIgLib dataset (Southern California "
        "wildfire cameras) showed substantially lower performance due to "
        "domain shift. This model was trained on Indian surveillance imagery "
        "and does not generalise reliably to all camera types or geographies.\n\n"
        "Do not assume universal wildfire detection capability.",
        icon="⚠️",
    )
    st.divider()

    # Live model status
    if _CHECKPOINT.exists():
        st.success("Model checkpoint loaded", icon="✅")
    else:
        st.error("Checkpoint not found", icon="❌")
    st.caption("PyroSight · Research prototype · Not for operational use")

# ---------------------------------------------------------------------------
# Attempt to load the model at startup (cached)
# ---------------------------------------------------------------------------
model_ok  = False
model_err = ""
try:
    model, device = _load_model()
    model_ok = True
except Exception as exc:
    model_err = str(exc)

if not model_ok:
    st.error(
        f"**Model failed to load.**\n\n```\n{model_err}\n```\n\n"
        "Ensure `models/hybrid_robust_best.pt` exists and the `src/` package "
        "is accessible from the project root.",
        icon="❌",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Image uploader — full width above the two-column layout
# ---------------------------------------------------------------------------
st.subheader("Upload an Image")
st.markdown(
    "Upload a JPEG image to analyse for the presence of wildfire smoke. "
    "Supported formats: **JPG, JPEG**."
)

uploaded_file = st.file_uploader(
    label="Choose a JPEG image",
    type=["jpg", "jpeg"],
    accept_multiple_files=False,
    help="Supported formats: JPG, JPEG",
)

# ---------------------------------------------------------------------------
# Try a Sample Image
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Try a Sample Image")
st.markdown(
    "No image to hand? Select one of the built-in samples and click "
    "**Use Sample** to load it directly into the analysis pipeline."
)

sample_col, btn_col = st.columns([2, 1], gap="medium")
with sample_col:
    sample_choice = st.selectbox(
        "Sample image",
        options=list(_SAMPLE_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
with btn_col:
    use_sample = st.button(
        "📂 Use Sample", use_container_width=True
    )

if use_sample:
    sample_path = _SAMPLE_OPTIONS[sample_choice]
    if not sample_path.exists():
        st.warning(
            f"Sample file not found: `{sample_path.relative_to(_PROJECT_ROOT)}`\n\n"
            "Place a JPEG named `clear.jpg`, `smoke.jpg`, or `fog.jpg` "
            "in the `app/samples/` directory.",
            icon="⚠️",
        )
    else:
        try:
            st.session_state["active_image"] = Image.open(sample_path).convert("RGB")
            st.session_state["active_name"]  = f"{sample_choice}  (sample)"
        except Exception as exc:
            st.error(f"Could not open sample image.\n\n`{exc}`", icon="⚠️")

# ---------------------------------------------------------------------------
# Resolve which image to display and analyse:
#   priority 1 — freshly uploaded file
#   priority 2 — sample loaded into session state
# ---------------------------------------------------------------------------
image: Image.Image | None = None
image_name: str = ""

if uploaded_file is not None:
    try:
        image      = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
        image_name = uploaded_file.name
        # Clear any previously loaded sample so they don't conflict
        st.session_state.pop("active_image", None)
        st.session_state.pop("active_name",  None)
    except (UnidentifiedImageError, Exception) as exc:
        st.error(
            f"Could not open the uploaded file as an image.\n\n`{exc}`",
            icon="⚠️",
        )
        st.stop()
elif "active_image" in st.session_state:
    image      = st.session_state["active_image"]
    image_name = st.session_state.get("active_name", "sample")

# ---------------------------------------------------------------------------
# Two-column layout: left = image, right = results
# ---------------------------------------------------------------------------
if image is not None:

    w, h = image.size
    st.divider()

    col_img, col_result = st.columns([1, 1], gap="large")

    # ---- Left column: image ----------------------------------------
    with col_img:
        st.markdown("**Uploaded image**")
        st.image(
            image,
            caption=f"{image_name}  ({w} \u00d7 {h} px)",
            use_container_width=True,
        )

    # ---- Right column: prediction ----------------------------------
    with col_result:
        st.markdown("**Analysis**")

        analyze = st.button(
            "🔍 Analyze Image", type="primary", use_container_width=True
        )

        if analyze:
            with st.spinner("Running inference…"):
                try:
                    tensor        = _preprocess(image)
                    label, prob   = _run_inference(model, device, tensor)
                    confidence_pc = prob * 100 if label == "SMOKE" else (1 - prob) * 100
                except Exception as exc:
                    st.error(
                        f"Inference failed.\n\n`{exc}`",
                        icon="⚠️",
                    )
                    st.stop()

            # Prediction banner
            st.divider()
            if label == "SMOKE":
                st.error("## 🔥 SMOKE DETECTED", icon="🚨")
            else:
                st.success("## ✅ NO SMOKE DETECTED", icon="✅")

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction",  label)
            m2.metric("Smoke prob.", f"{prob * 100:.1f}%")
            m3.metric("Confidence",  f"{confidence_pc:.1f}%")

            # Probability bar
            st.markdown("**Smoke probability**")
            st.progress(
                min(max(prob, 0.0), 1.0),
                text=f"{prob * 100:.1f}%  (threshold: {_THRESHOLD * 100:.0f}%)",
            )

            # Disclaimer
            st.divider()
            st.warning(
                "**AI prediction — not a guarantee.**  \n"
                "This result is produced by a research model trained on a "
                "limited dataset. It should not be used as the sole basis for "
                "any emergency decision. Always follow official wildfire "
                "guidance from local authorities.",
                icon="⚠️",
            )
        else:
            st.info(
                "Click **Analyze Image** to run the smoke detection model.",
                icon="👆",
            )

else:
    st.info("Upload a JPEG image or use a sample above to get started.", icon="📂")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"PyroSight \u00b7 Research prototype \u00b7 "
    f"Model: hybrid_robust_best.pt \u00b7 "
    f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'} \u00b7 "
    f"Threshold: {_THRESHOLD} \u00b7 "
    f"Training data: IIITDMJ_Smoke"
)
