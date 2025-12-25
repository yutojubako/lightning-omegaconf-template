from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import rootutils
from omegaconf import DictConfig, OmegaConf, open_dict


def load_config(
    config_name: str,
    config_dir: str | Path = "configs",
    overrides: Optional[DictConfig] = None,
) -> DictConfig:
    """Load and merge configs from the config directory.

    :param config_name: Name of the top-level config file without the extension.
    :param config_dir: Directory containing the config tree.
    :param overrides: Optional OmegaConf configuration containing overrides.
    :return: The merged DictConfig object.
    """
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    project_root = Path(rootutils.find_root(indicator=".project-root"))
    config_root = project_root / config_dir

    cfg = _load_config_file(config_root / f"{config_name}.yaml", config_root)
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)

    _populate_runtime_paths(cfg, project_root)
    return cfg


def _load_config_file(config_path: Path, config_root: Path) -> DictConfig:
    """Load a config file and recursively merge its defaults."""
    cfg = OmegaConf.load(config_path)
    defaults = cfg.get("defaults") or []

    configs = []
    base_cfg = _remove_defaults(cfg)
    has_self = False
    current_group_dir = config_path.parent

    for entry in defaults:
        if entry == "_self_":
            configs.append(base_cfg)
            has_self = True
            continue

        loaded = _load_default_entry(entry, config_root, current_group_dir)
        if loaded is not None:
            configs.append(loaded)

    if not has_self:
        configs.append(base_cfg)

    return OmegaConf.merge(*configs)


def _load_default_entry(
    entry: Any,
    config_root: Path,
    current_group_dir: Path,
) -> Optional[DictConfig]:
    """Load a default config entry from a defaults list."""
    if isinstance(entry, str):
        config_path = current_group_dir / f"{entry}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        return _load_config_file(config_path, config_root)

    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(f"Invalid defaults entry: {entry}")

        group_key, value = next(iter(entry.items()))
        if value is None:
            return None

        group_name, optional, allow_missing = _parse_group_key(group_key)
        config_path = config_root / group_name / f"{value}.yaml"
        if not config_path.exists():
            if optional or allow_missing:
                return None
            raise FileNotFoundError(f"Missing config file: {config_path}")
        return _load_config_file(config_path, config_root)

    raise TypeError(f"Unsupported defaults entry type: {type(entry)}")


def _parse_group_key(group_key: str) -> tuple[str, bool, bool]:
    """Parse a Hydra defaults group key."""
    optional_prefix = "optional "
    override_prefix = "override "
    if group_key.startswith(optional_prefix):
        return group_key[len(optional_prefix) :].strip().lstrip("/"), True, False
    if group_key.startswith(override_prefix):
        return group_key[len(override_prefix) :].strip().lstrip("/"), False, True
    return group_key.lstrip("/"), False, False


def _remove_defaults(cfg: DictConfig) -> DictConfig:
    """Return a config without its defaults list."""
    data = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(data, dict):
        data.pop("defaults", None)
    return OmegaConf.create(data)


def _populate_runtime_paths(cfg: DictConfig, project_root: Path) -> None:
    """Set runtime paths and create the output directory."""
    with open_dict(cfg):
        cfg.paths.root_dir = str(project_root)
        cfg.paths.work_dir = str(Path.cwd())

    # Validate that paths section exists and has log_dir
    if not cfg.get("paths"):
        raise ValueError("Config must contain a 'paths' section")
    
    # Register "now" resolver before resolving paths
    _register_now_resolver()
    
    resolved_paths = OmegaConf.to_container(cfg.paths, resolve=True)
    if not isinstance(resolved_paths, dict) or "log_dir" not in resolved_paths:
        raise ValueError("Config paths section must contain 'log_dir' key")
    
    log_dir = Path(str(resolved_paths["log_dir"]))
    task_name = str(cfg.get("task_name", "task"))
    output_dir = log_dir / task_name / "runs" / _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.paths.output_dir = str(output_dir)

    _register_hydra_resolver(output_dir=output_dir, cwd=Path.cwd())


def _register_now_resolver() -> None:
    """Register the 'now' OmegaConf resolver."""

    def resolve_now(pattern: str = "%Y-%m-%d_%H-%M-%S") -> str:
        return datetime.now().strftime(pattern)

    OmegaConf.register_new_resolver("now", resolve_now, replace=True)


def _register_hydra_resolver(output_dir: Path, cwd: Path) -> None:
    """Register the 'hydra' OmegaConf resolver for runtime paths."""

    def resolve_hydra(value: str) -> str:
        mapping = {
            "runtime.output_dir": str(output_dir),
            "runtime.cwd": str(cwd),
        }
        if value not in mapping:
            raise ValueError(f"Unsupported hydra resolver key: {value}")
        return mapping[value]

    OmegaConf.register_new_resolver("hydra", resolve_hydra, replace=True)


def _timestamp() -> str:
    """Return a timestamp string for output directory creation."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
