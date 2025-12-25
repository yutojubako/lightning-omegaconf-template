from pathlib import Path

from omegaconf import DictConfig

from src.utils import setup_output_dir


def test_setup_output_dir_creates_and_sets(tmp_path: Path) -> None:
    """Ensure the output directory is created and stored in the config."""
    cfg_train = DictConfig(
        {
            "paths": {
                "log_dir": str(tmp_path),
                # This will be set by setup_output_dir
                "output_dir": None,
            },
            "task_name": "dummy_task",
        }
    )
    output_dir = setup_output_dir(cfg_train)

    assert output_dir.exists()
    assert Path(cfg_train.paths.output_dir) == output_dir
    assert output_dir.parent.name == cfg_train.task_name
    assert output_dir.parent.parent == Path(cfg_train.paths.log_dir)
