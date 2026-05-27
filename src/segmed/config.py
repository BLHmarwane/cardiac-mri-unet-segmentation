from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only on minimal systems
    yaml = None


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "modern_unet"
    input_size: tuple[int, int] = (256, 256)
    base_filters: int = 32
    batch_norm: bool = True
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 80
    learning_rate: float = 0.001
    seed: int = 123
    threshold: float = 0.5
    augment: bool = True
    patience: int = 15


@dataclass(frozen=True)
class ProjectConfig:
    target: str
    dataset_root: Path
    output_dir: Path
    mask_dir_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _tuple_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Invalid input_size: {value!r}")


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            raw = yaml.safe_load(handle) or {}
        else:
            raw = _load_simple_yaml(handle.read())

    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})

    model = ModelConfig(
        architecture=model_raw.get("architecture", "modern_unet"),
        input_size=_tuple_size(model_raw.get("input_size", (256, 256))),
        base_filters=int(model_raw.get("base_filters", 32)),
        batch_norm=bool(model_raw.get("batch_norm", True)),
        dropout=float(model_raw.get("dropout", 0.1)),
    )
    training = TrainingConfig(
        batch_size=int(training_raw.get("batch_size", 8)),
        epochs=int(training_raw.get("epochs", 80)),
        learning_rate=float(training_raw.get("learning_rate", 0.001)),
        seed=int(training_raw.get("seed", 123)),
        threshold=float(training_raw.get("threshold", 0.5)),
        augment=bool(training_raw.get("augment", True)),
        patience=int(training_raw.get("patience", 15)),
    )

    dataset_root = Path(raw["dataset_root"]).expanduser()
    output_dir = Path(raw.get("output_dir", "outputs")).expanduser()
    return ProjectConfig(
        target=str(raw["target"]),
        dataset_root=dataset_root,
        output_dir=output_dir,
        mask_dir_name=str(raw["mask_dir_name"]),
        model=model,
        training=training,
    )


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small config subset used by this project when PyYAML is absent."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            key = stripped[:-1]
            root[key] = {}
            current = root[key]
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        target = current if line.startswith(" ") and current is not None else root
        target[key.strip()] = _parse_scalar(raw_value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar(part.strip()) for part in value[1:-1].split(",") if part.strip()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
