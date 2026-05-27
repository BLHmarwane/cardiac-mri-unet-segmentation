from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal local smoke tests
    binary_erosion = None
    distance_transform_edt = None
    generate_binary_structure = None


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(mask >= threshold, dtype=bool)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, epsilon: float = 1e-7) -> float:
    true = binarize(y_true, threshold)
    pred = binarize(y_pred, threshold)
    intersection = np.logical_and(true, pred).sum()
    denominator = true.sum() + pred.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * intersection + epsilon) / (denominator + epsilon))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, epsilon: float = 1e-7) -> float:
    true = binarize(y_true, threshold)
    pred = binarize(y_pred, threshold)
    union = np.logical_or(true, pred).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(true, pred).sum()
    return float((intersection + epsilon) / (union + epsilon))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, epsilon: float = 1e-7) -> float:
    true = binarize(y_true, threshold)
    pred = binarize(y_pred, threshold)
    tp = np.logical_and(true, pred).sum()
    fp = np.logical_and(~true, pred).sum()
    if tp + fp == 0:
        return 1.0 if true.sum() == 0 else 0.0
    return float((tp + epsilon) / (tp + fp + epsilon))


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, epsilon: float = 1e-7) -> float:
    true = binarize(y_true, threshold)
    pred = binarize(y_pred, threshold)
    tp = np.logical_and(true, pred).sum()
    fn = np.logical_and(true, ~pred).sum()
    if tp + fn == 0:
        return 1.0
    return float((tp + epsilon) / (tp + fn + epsilon))


def surface_distances(result: np.ndarray, reference: np.ndarray, connectivity: int = 1) -> np.ndarray:
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    if result.sum() == 0 or reference.sum() == 0:
        return np.array([np.inf], dtype=float)

    if binary_erosion is None or distance_transform_edt is None or generate_binary_structure is None:
        return _surface_distances_numpy(result, reference)

    footprint = generate_binary_structure(result.ndim, connectivity)
    result_border = np.logical_xor(result, binary_erosion(result, structure=footprint, iterations=1))
    reference_border = np.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=1))
    distances = distance_transform_edt(~reference_border)
    return distances[result_border]


def _surface_distances_numpy(result: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result_points = np.argwhere(_surface_mask(result))
    reference_points = np.argwhere(_surface_mask(reference))
    if len(result_points) == 0 or len(reference_points) == 0:
        return np.array([np.inf], dtype=float)
    distances = np.sqrt(((result_points[:, None, :] - reference_points[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = tuple(slice(1, -1) for _ in range(mask.ndim))
    interior = padded[center]
    surface = np.zeros_like(mask, dtype=bool)
    for axis in range(mask.ndim):
        before = list(center)
        after = list(center)
        before[axis] = slice(0, -2)
        after[axis] = slice(2, None)
        surface |= interior != padded[tuple(before)]
        surface |= interior != padded[tuple(after)]
    return np.logical_and(mask, surface)


def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    true = binarize(y_true, threshold)
    pred = binarize(y_pred, threshold)
    if true.sum() == 0 and pred.sum() == 0:
        return 0.0
    d1 = surface_distances(pred, true)
    d2 = surface_distances(true, pred)
    return float(max(np.max(d1), np.max(d2)))


def assd(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    true = binarize(y_true, threshold)
    pred = binarize(y_pred, threshold)
    if true.sum() == 0 and pred.sum() == 0:
        return 0.0
    d1 = surface_distances(pred, true)
    d2 = surface_distances(true, pred)
    return float((np.mean(d1) + np.mean(d2)) / 2.0)


def evaluate_pair(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    return {
        "dice": dice_score(y_true, y_pred, threshold),
        "iou": iou_score(y_true, y_pred, threshold),
        "precision": precision_score(y_true, y_pred, threshold),
        "recall": recall_score(y_true, y_pred, threshold),
        "hausdorff": hausdorff_distance(y_true, y_pred, threshold),
        "assd": assd(y_true, y_pred, threshold),
    }


def keras_dice_coef(y_true, y_pred):
    import tensorflow.keras.backend as K

    y_true_vect = K.flatten(y_true)
    y_pred_vect = K.flatten(y_pred)
    intersection = K.sum(y_true_vect * y_pred_vect)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_vect) + K.sum(y_pred_vect) + 1.0)
