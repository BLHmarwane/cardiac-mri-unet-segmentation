import math

import numpy as np

from segmed.metrics import assd, dice_score, evaluate_pair, hausdorff_distance, iou_score, precision_score, recall_score


def test_overlap_metrics_are_perfect_for_identical_masks():
    mask = np.zeros((8, 8), dtype=float)
    mask[2:5, 2:5] = 1.0

    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0
    assert precision_score(mask, mask) == 1.0
    assert recall_score(mask, mask) == 1.0
    assert hausdorff_distance(mask, mask) == 0.0
    assert assd(mask, mask) == 0.0


def test_empty_masks_are_treated_as_perfect_empty_prediction():
    true = np.zeros((8, 8), dtype=float)
    pred = np.zeros((8, 8), dtype=float)

    metrics = evaluate_pair(true, pred)

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["hausdorff"] == 0.0
    assert metrics["assd"] == 0.0


def test_surface_metrics_are_infinite_when_only_one_mask_is_empty():
    true = np.zeros((8, 8), dtype=float)
    pred = np.zeros((8, 8), dtype=float)
    pred[3:5, 3:5] = 1.0

    assert dice_score(true, pred) < 1.0
    assert math.isinf(hausdorff_distance(true, pred))
    assert math.isinf(assd(true, pred))
