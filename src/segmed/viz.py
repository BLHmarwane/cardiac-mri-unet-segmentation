from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave


def make_overlay(image: np.ndarray, mask: np.ndarray, prediction: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    base = np.squeeze(image)
    true = np.squeeze(mask) >= threshold
    pred = np.squeeze(prediction) >= threshold
    rgb = np.stack([base, base, base], axis=-1).astype("float32")
    rgb[true, 1] = 1.0
    rgb[pred, 0] = 1.0
    rgb[np.logical_and(true, pred)] = [1.0, 1.0, 0.0]
    return np.clip(rgb, 0.0, 1.0)


def save_overlay(path: Path, image: np.ndarray, mask: np.ndarray, prediction: np.ndarray, threshold: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = (make_overlay(image, mask, prediction, threshold) * 255).astype("uint8")
    imsave(path, overlay)


def save_panel(path: Path, image: np.ndarray, mask: np.ndarray | None, prediction: np.ndarray, threshold: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 4 if mask is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(np.squeeze(image), cmap="gray")
    axes[0].set_title("Image")
    if mask is not None:
        axes[1].imshow(np.squeeze(mask), cmap="gray")
        axes[1].set_title("Ground truth")
        axes[2].imshow(np.squeeze(prediction) >= threshold, cmap="gray")
        axes[2].set_title("Prediction")
        axes[3].imshow(make_overlay(image, mask, prediction, threshold))
        axes[3].set_title("Overlay")
    else:
        axes[1].imshow(np.squeeze(prediction) >= threshold, cmap="gray")
        axes[1].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
