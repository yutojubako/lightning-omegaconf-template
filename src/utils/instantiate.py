"""Utilities for instantiating objects from configuration."""

from importlib import import_module
from typing import Any

from omegaconf import DictConfig, OmegaConf


def instantiate(cfg: DictConfig, **kwargs: Any) -> Any:
    """Instantiate an object from a DictConfig with _target_ key.

    :param cfg: A DictConfig containing _target_ and parameters.
    :param kwargs: Additional keyword arguments to override config values.
    :return: The instantiated object.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config must be a DictConfig, got {type(cfg)}")

    if "_target_" not in cfg:
        raise ValueError("Config must contain '_target_' key")

    target = cfg._target_
    params = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(params, dict):
        params.pop("_target_", None)
        params.update(kwargs)
    else:
        params = kwargs

    # Import the target class or function
    module_path, class_name = target.rsplit(".", 1)
    module = import_module(module_path)
    target_class = getattr(module, class_name)

    return target_class(**params)
