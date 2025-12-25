"""Tests for instantiate utility."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

# Import directly from the instantiate module using importlib to avoid pulling
# in all dependencies from src/utils/__init__.py (particularly Lightning)
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "instantiate",
    Path(__file__).parent.parent / "src" / "utils" / "instantiate.py"
)
instantiate_module = importlib.util.module_from_spec(spec)
sys.modules["instantiate"] = instantiate_module
spec.loader.exec_module(instantiate_module)

instantiate = instantiate_module.instantiate


def test_instantiate_class() -> None:
    """Instantiate a class from config."""
    cfg = OmegaConf.create(
        {"_target_": "tests.helpers.instantiate_targets.Example", "value": 3}
    )
    result = instantiate(cfg)
    assert result.value == 3


def test_instantiate_function() -> None:
    """Instantiate a function from config."""
    cfg = OmegaConf.create(
        {"_target_": "tests.helpers.instantiate_targets.add", "a": 2, "b": 5}
    )
    result = instantiate(cfg)
    assert result == 7


def test_instantiate_overrides() -> None:
    """Override config values during instantiation."""
    cfg = OmegaConf.create(
        {"_target_": "tests.helpers.instantiate_targets.Example", "value": 1}
    )
    result = instantiate(cfg, value=10)
    assert result.value == 10


def test_instantiate_requires_dictconfig() -> None:
    """Raise when config is not a DictConfig."""
    with pytest.raises(TypeError, match="Config must be a DictConfig"):
        instantiate({"_target_": "tests.helpers.instantiate_targets.Example"})


def test_instantiate_requires_target() -> None:
    """Raise when config has no _target_."""
    cfg = DictConfig({"value": 1})
    with pytest.raises(ValueError, match="Config must include a string '_target_' entry"):
        instantiate(cfg)


def test_instantiate_invalid_module() -> None:
    """Raise when module does not exist."""
    cfg = OmegaConf.create(
        {"_target_": "nonexistent.module.Class"}
    )
    with pytest.raises(ModuleNotFoundError, match="Could not import module 'nonexistent.module'"):
        instantiate(cfg)


def test_instantiate_invalid_attribute() -> None:
    """Raise when attribute does not exist in module."""
    cfg = OmegaConf.create(
        {"_target_": "tests.helpers.instantiate_targets.NonExistentClass"}
    )
    with pytest.raises(AttributeError, match="attribute 'NonExistentClass' not found"):
        instantiate(cfg)


def test_instantiate_target_without_dots() -> None:
    """Handle target path without dots (built-in types)."""
    # Built-in types like dict should work
    cfg = OmegaConf.create(
        {"_target_": "dict"}
    )
    result = instantiate(cfg)
    assert result == {}
