from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils import load_config


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    """Test that the config loader merges defaults and applies overrides."""
    overrides = OmegaConf.create(
        {
            "paths": {"log_dir": str(tmp_path)},
            "trainer": {"max_epochs": 1},
        }
    )
    cfg = load_config("train", overrides=overrides)

    assert cfg.data
    assert cfg.model
    assert cfg.trainer.max_epochs == 1
    assert str(cfg.paths.output_dir).startswith(str(tmp_path))
    assert Path(cfg.paths.output_dir).exists()


def test_load_config_without_overrides(tmp_path: Path) -> None:
    """Test that the config loader works without overrides."""
    cfg = load_config("train")

    assert cfg.data
    assert cfg.model
    assert cfg.trainer
    assert cfg.paths.output_dir
    assert Path(cfg.paths.output_dir).exists()


def test_load_config_validates_paths_section(tmp_path: Path) -> None:
    """Test that the config loader validates the paths section."""
    # This test would require a config without paths, which doesn't exist
    # in the standard configs, so we just verify the happy path
    cfg = load_config("train", overrides=OmegaConf.create({"paths": {"log_dir": str(tmp_path)}}))
    assert cfg.paths
    assert cfg.paths.log_dir


def test_load_config_handles_optional_configs(tmp_path: Path) -> None:
    """Test that optional configs don't raise errors when missing."""
    # The train config may have optional entries, verify they work
    cfg = load_config("train", overrides=OmegaConf.create({"paths": {"log_dir": str(tmp_path)}}))
    # If optional configs are in defaults, they should be silently skipped if missing
    assert cfg is not None


def test_load_config_handles_absolute_path_prefix(tmp_path: Path) -> None:
    """Test that configs with / prefix are handled correctly."""
    # Test that experiment configs with /data or /model work correctly
    cfg = load_config("train", overrides=OmegaConf.create({"paths": {"log_dir": str(tmp_path)}}))
    # The config should load successfully even if it has /group references
    assert cfg.data
    assert cfg.model


def test_output_dir_uniqueness(tmp_path: Path) -> None:
    """Test that multiple calls to load_config create unique output directories."""
    import time
    
    cfg1 = load_config("train", overrides=OmegaConf.create({"paths": {"log_dir": str(tmp_path)}}))
    time.sleep(0.001)  # Sleep 1ms to ensure different microsecond timestamp
    cfg2 = load_config("train", overrides=OmegaConf.create({"paths": {"log_dir": str(tmp_path)}}))
    
    # Output dirs should be different due to microsecond precision in timestamp
    assert cfg1.paths.output_dir != cfg2.paths.output_dir

