from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np

from segmed.config import ProjectConfig
from segmed.dataset import load_arrays, read_grayscale
from segmed.metrics import evaluate_pair, keras_dice_coef
from segmed.model import build_model
from segmed.viz import save_panel


def create_model(config: ProjectConfig):
    import tensorflow as tf

    model = build_model(
        config.model.architecture,
        config.model.input_size,
        config.model.base_filters,
        config.model.batch_norm,
        config.model.dropout,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[keras_dice_coef])
    return model


def train(config: ProjectConfig, weights: str | None = None) -> Path:
    import tensorflow as tf

    from segmed.dataset import make_tf_datasets

    tf.keras.utils.set_random_seed(config.training.seed)
    output_dir = config.output_dir / config.target
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(config)
    if weights:
        model.load_weights(weights)

    train_ds, val_ds = make_tf_datasets(
        config.dataset_root,
        config.mask_dir_name,
        config.model.input_size,
        config.training.batch_size,
        config.training.threshold,
        config.training.seed,
        config.training.augment,
    )
    checkpoint_path = checkpoint_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True, save_weights_only=True, monitor="val_loss"),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        tf.keras.callbacks.EarlyStopping(patience=config.training.patience, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, config.training.patience // 3)),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=config.training.epochs, callbacks=callbacks)
    final_path = checkpoint_dir / "final.weights.h5"
    model.save_weights(final_path)
    return final_path


def predict_array(config: ProjectConfig, weights: str, image: np.ndarray) -> np.ndarray:
    model = create_model(config)
    model.load_weights(weights)
    batch = image.reshape(1, config.model.input_size[0], config.model.input_size[1], 1)
    return model.predict(batch, verbose=0)[0, :, :, 0]


def predict_file(config: ProjectConfig, weights: str, image_path: str | Path, output_path: str | Path | None = None) -> Path:
    image = read_grayscale(Path(image_path), config.model.input_size)
    prediction = predict_array(config, weights, image)
    if output_path is None:
        output_path = config.output_dir / config.target / "predictions" / f"{Path(image_path).stem}_prediction.png"
    output_path = Path(output_path)
    save_panel(output_path, image, None, prediction, config.training.threshold)
    return output_path


def evaluate(config: ProjectConfig, weights: str, split: str = "Val", sample_count: int = 8) -> dict[str, object]:
    model = create_model(config)
    model.load_weights(weights)
    images, masks, pairs = load_arrays(
        config.dataset_root,
        config.mask_dir_name,
        split,
        config.model.input_size,
        config.training.threshold,
    )
    predictions = model.predict(images, batch_size=config.training.batch_size, verbose=1)
    rows: list[dict[str, float | str]] = []
    output_dir = config.output_dir / config.target / split.lower()
    panels_dir = output_dir / "panels"
    for index, pair in enumerate(pairs):
        metrics = evaluate_pair(masks[index, :, :, 0], predictions[index, :, :, 0], config.training.threshold)
        rows.append({"image": pair.image.name, **metrics})
        if index < sample_count:
            save_panel(
                panels_dir / f"{index:03d}_{pair.image.stem}.png",
                images[index, :, :, 0],
                masks[index, :, :, 0],
                predictions[index, :, :, 0],
                config.training.threshold,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics_per_image.csv", rows)
    summary = {
        "target": config.target,
        "split": split,
        "weights": str(weights),
        "count": len(rows),
        "metrics": {
            column: {
                "mean": _finite_mean(np.array([row[column] for row in rows], dtype=float)),
                "std": _finite_std(np.array([row[column] for row in rows], dtype=float)),
            }
            for column in ["dice", "iou", "precision", "recall", "hausdorff", "assd"]
        },
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("inf")


def _finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else float("inf")


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
