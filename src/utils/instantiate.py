"""Utilities for instantiating targets from configuration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _locate_target(target_path: str) -> Any:
    """Resolve a dotted path into a Python object.

    :param target_path: Dotted path to the target object.
    :return: Resolved Python object.
    :raises ValueError: If ``target_path`` is empty or otherwise invalid.
    :raises ModuleNotFoundError: If the specified module cannot be imported.
    :raises AttributeError: If the requested attribute is not found on the resolved module or in builtins.
    """
    if not target_path:
        raise ValueError(f"Invalid target path: {target_path!r}")

    if "." not in target_path:
        # Handle built-in types or single-name targets by checking builtins first
        try:
            import builtins
            return getattr(builtins, target_path)
        except AttributeError:
            # If not a builtin, try importing as a module
            try:
                module = import_module(target_path)
                return module
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Name {target_path!r} is neither a builtin nor an importable module. "
                    "If this was meant to be a module, ensure it is installed and importable; "
                    "if it was meant to be a builtin, check that the name is correct."
                ) from e

    module_path, attr_name = target_path.rsplit(".", 1)
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import module {module_path!r} when resolving target {target_path!r}. "
            "Ensure the module is installed and importable."
        ) from e

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Cannot resolve target path {target_path!r}: "
            f"attribute {attr_name!r} not found in module {module_path!r}"
        ) from exc


def _extract_params(cfg: DictConfig) -> dict[str, Any]:
    """Extract parameters from a DictConfig, excluding the target key.

    :param cfg: Configuration to unpack.
    :return: Parameters for the target callable.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            "Expected OmegaConf.to_container to return a dict for parameter extraction, "
            f"but got {type(container).__name__}."
        )
    params = {key: value for key, value in container.items() if key != "_target_"}
    return params


def instantiate(cfg: DictConfig, **overrides: Any) -> Any:
    """Instantiate a target from config.

    :param cfg: DictConfig containing a ``_target_`` entry.
    :param overrides: Optional keyword overrides for parameters.
    :return: Instantiated object.
    :raises TypeError: If ``cfg`` is not an instance of :class:`DictConfig`.
    :raises ValueError: If ``cfg`` does not include a string ``\"_target_\"`` entry.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError("Config must be a DictConfig.")

    target_path = cfg.get("_target_")
    if not isinstance(target_path, str):
        raise ValueError("Config must include a string '_target_' entry.")

    target = _locate_target(target_path)
    params = _extract_params(cfg)
    params.update(overrides)
    return target(**params)
