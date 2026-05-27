from pathlib import Path

from segmed.config import load_config


def test_load_config_parses_project_config():
    config = load_config(Path("configs/endo.yaml"))

    assert config.target == "endo"
    assert config.mask_dir_name == "masks endo"
    assert config.model.input_size == (256, 256)
    assert config.model.architecture == "legacy_unet"
    assert config.training.threshold == 0.5
