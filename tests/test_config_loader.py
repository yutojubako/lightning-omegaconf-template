from pathlib import Path

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
