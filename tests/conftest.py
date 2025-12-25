"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils import load_config


@pytest.fixture(scope="package")
def cfg_train_global(tmp_path_factory: pytest.TempPathFactory) -> DictConfig:
    """A pytest fixture for setting up a default DictConfig for training.

    :param tmp_path_factory: Pytest fixture for creating temporary directories.
    :return: A DictConfig object containing a default configuration for training.
    """
    # Use a temporary directory for the package-scoped fixture
    tmp_log_dir = tmp_path_factory.mktemp("train_logs")
    
    cfg = load_config(
        "train",
        overrides=OmegaConf.create({"paths": {"log_dir": str(tmp_log_dir)}}),
    )

    # set defaults for all tests
    with open_dict(cfg):
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 0.01
        cfg.trainer.limit_val_batches = 0.1
        cfg.trainer.limit_test_batches = 0.1
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global(tmp_path_factory: pytest.TempPathFactory) -> DictConfig:
    """A pytest fixture for setting up a default DictConfig for evaluation.

    :param tmp_path_factory: Pytest fixture for creating temporary directories.
    :return: A DictConfig containing a default configuration for evaluation.
    """
    # Use a temporary directory for the package-scoped fixture
    tmp_log_dir = tmp_path_factory.mktemp("eval_logs")
    
    cfg = load_config(
        "eval",
        overrides=OmegaConf.create({
            "ckpt_path": ".",
            "paths": {"log_dir": str(tmp_log_dir)},
        }),
    )

    # set defaults for all tests
    with open_dict(cfg):
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_test_batches = 0.1
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg



@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg
