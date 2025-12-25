from pathlib import Path

from omegaconf import DictConfig

from src.utils import setup_output_dir


def test_setup_output_dir_creates_and_sets(cfg_train: DictConfig) -> None:
    """Ensure the output directory is created and stored in the config.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    output_dir = setup_output_dir(cfg_train)

    assert output_dir.exists()
    assert Path(cfg_train.paths.output_dir) == output_dir
    assert output_dir.parent.name == cfg_train.task_name
    assert output_dir.parent.parent == Path(cfg_train.paths.log_dir)
