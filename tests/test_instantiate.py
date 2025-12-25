"""Tests for instantiate utility."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

from src.utils.instantiate import instantiate


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
    with pytest.raises(TypeError):
        instantiate({"_target_": "tests.helpers.instantiate_targets.Example"})


def test_instantiate_requires_target() -> None:
    """Raise when config has no _target_."""
    cfg = DictConfig({"value": 1})
    with pytest.raises(ValueError):
        instantiate(cfg)
