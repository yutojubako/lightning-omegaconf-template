"""Utilities for instantiating targets from configuration."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, TypeVar, cast

from omegaconf import DictConfig, OmegaConf


class TargetConfig(Protocol):
    """Protocol for configs that include a target path."""

    def get(self, key: str, default: Any | None = None) -> Any: ...


T = TypeVar("T")


def _locate_target(target_path: str) -> Any:
    """Resolve a dotted path into a Python object.

    :param target_path: Dotted path to the target object.
    :return: Resolved Python object.
    """
    if not target_path or "." not in target_path:
        raise ValueError(f"Invalid target path: {target_path!r}")

    module_path, attr_name = target_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr_name)


def _extract_params(cfg: DictConfig) -> dict[str, Any]:
    """Extract parameters from a DictConfig, excluding the target key.

    :param cfg: Configuration to unpack.
    :return: Parameters for the target callable.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Config must resolve to a dictionary.")
    params = {key: value for key, value in container.items() if key != "_target_"}
    return params


def instantiate(cfg: DictConfig, **overrides: Any) -> T:
    """Instantiate a target from config.

    :param cfg: DictConfig containing a ``_target_`` entry.
    :param overrides: Optional keyword overrides for parameters.
    :return: Instantiated object.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError("Config must be a DictConfig.")

    target_path = cast(TargetConfig, cfg).get("_target_")
    if not isinstance(target_path, str):
        raise ValueError("Config must include a string '_target_' entry.")

    target = _locate_target(target_path)
    params = _extract_params(cfg)
    params.update(overrides)
    return cast(T, target(**params))
