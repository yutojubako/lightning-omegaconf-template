"""Targets for instantiate tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Example:
    """Example dataclass for instantiation tests."""

    value: int


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""

    return a + b
