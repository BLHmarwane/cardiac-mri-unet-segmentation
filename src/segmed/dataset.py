from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize


@dataclass(frozen=True)
class SegmentationPair:
    image: Path
    mask: Path


def list_pairs(dataset_root: Path, mask_dir_name: str, split: str) -> list[SegmentationPair]:
    frame_dir = dataset_root / "frames" / split
    mask_dir = dataset_root / mask_dir_name / split
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing frame directory: {frame_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    frames = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    pairs: list[SegmentationPair] = []
    missing: list[str] = []
    for frame in frames:
        mask = mask_dir / frame.name
        if mask.exists():
            pairs.append(SegmentationPair(frame, mask))
        else:
            missing.append(frame.name)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} masks missing in {mask_dir}: {preview}")
    return pairs


def read_grayscale(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = imread(path, as_gray=True)
    image = resize(image, size, preserve_range=True, anti_aliasing=True)
    image = image.astype("float32")
    max_value = float(image.max())
    if max_value > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def read_mask(path: Path, size: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    mask = read_grayscale(path, size)
    return (mask >= threshold).astype("float32")


def load_arrays(
    dataset_root: Path,
    mask_dir_name: str,
    split: str,
    size: tuple[int, int],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[SegmentationPair]]:
    pairs = list_pairs(dataset_root, mask_dir_name, split)
    images = np.zeros((len(pairs), size[0], size[1], 1), dtype="float32")
    masks = np.zeros_like(images)
    for index, pair in enumerate(pairs):
        images[index, :, :, 0] = read_grayscale(pair.image, size)
        masks[index, :, :, 0] = read_mask(pair.mask, size, threshold)
    return images, masks, pairs


def make_tf_datasets(
    dataset_root: Path,
    mask_dir_name: str,
    size: tuple[int, int],
    batch_size: int,
    threshold: float,
    seed: int,
    augment: bool,
):
    import tensorflow as tf

    train_x, train_y, _ = load_arrays(dataset_root, mask_dir_name, "Train", size, threshold)
    val_x, val_y, _ = load_arrays(dataset_root, mask_dir_name, "Val", size, threshold)

    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    if augment:
        train = train.map(lambda image, mask: _augment(image, mask, seed), num_parallel_calls=tf.data.AUTOTUNE)
    train = train.shuffle(len(train_x), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train, val


def _augment(image, mask, seed: int):
    import tensorflow as tf

    pair = tf.concat([image, mask], axis=-1)
    pair = tf.image.stateless_random_flip_left_right(pair, seed=[seed, 1])
    pair = tf.image.stateless_random_flip_up_down(pair, seed=[seed, 2])
    image_aug = pair[:, :, :1]
    mask_aug = pair[:, :, 1:]
    return image_aug, mask_aug
