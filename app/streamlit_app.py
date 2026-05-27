from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from skimage.io import imread
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmed.config import load_config
from segmed.metrics import evaluate_pair
from segmed.pipeline import create_model
from segmed.viz import make_overlay


@st.cache_resource
def load_cached_model(config_path: str, weights_path: str):
    config = load_config(config_path)
    model = create_model(config)
    model.load_weights(weights_path)
    return config, model


def normalize_image(uploaded_file, size: tuple[int, int]) -> np.ndarray:
    image = imread(uploaded_file, as_gray=True)
    image = resize(image, size, preserve_range=True, anti_aliasing=True).astype("float32")
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def synthetic_cardiac_like_image(size: tuple[int, int]) -> np.ndarray:
    height, width = size
    y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    outer = np.exp(-((x / 0.62) ** 2 + (y / 0.78) ** 2) * 2.1)
    inner = np.exp(-((x / 0.30) ** 2 + (y / 0.42) ** 2) * 5.5)
    texture = 0.08 * np.sin(18 * x + 5 * y) + 0.05 * np.cos(14 * y)
    image = 0.18 + 0.55 * outer - 0.28 * inner + texture
    return np.clip(image, 0.0, 1.0).astype("float32")


st.set_page_config(page_title="Cardiac U-Net Segmentation", layout="wide")
st.title("Cardiac U-Net Segmentation")

with st.sidebar:
    target = st.selectbox("Target", ["endo", "epi"])
    default_config = ROOT / "configs" / f"{target}.yaml"
    default_weights = ROOT / "models" / (
        "endo_unet.h5" if target == "endo" else "epi_unet.h5"
    )
    config_path = st.text_input("Config path", str(default_config))
    weights_path = st.text_input("Weights path", str(default_weights))
    threshold = st.slider("Threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.05)

use_demo = st.button("Run demo sample")
uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
uploaded_mask = st.file_uploader("Upload mask", type=["png", "jpg", "jpeg"])

if (uploaded_image or use_demo) and weights_path:
    config, model = load_cached_model(config_path, weights_path)
    image = synthetic_cardiac_like_image(config.model.input_size) if use_demo else normalize_image(uploaded_image, config.model.input_size)
    prediction = model.predict(image.reshape(1, *config.model.input_size, 1), verbose=0)[0, :, :, 0]

    mask = None
    if uploaded_mask:
        mask = normalize_image(uploaded_mask, config.model.input_size)
        mask = (mask >= threshold).astype("float32")

    col_image, col_pred, col_overlay = st.columns(3)
    col_image.image(image, caption="Input image", clamp=True)
    col_pred.image(prediction >= threshold, caption="Predicted mask", clamp=True)
    if mask is not None:
        overlay = make_overlay(image, mask, prediction, threshold)
        col_overlay.image(overlay, caption="Overlay", clamp=True)
        metrics = evaluate_pair(mask, prediction, threshold)
        st.dataframe({key: [round(value, 4)] for key, value in metrics.items()}, use_container_width=True)
    else:
        col_overlay.image(prediction, caption="Prediction probability", clamp=True)
else:
    st.info("Run the demo sample or upload an image.")
