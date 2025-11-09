"""Utility helpers shared by the benchmarking suite."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import typing as t

import numpy as np

RNGSeed = int | None


def set_random_seed(seed: RNGSeed) -> None:
    """Seed the standard random module and NumPy for reproducibility."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def batched(iterable: t.Iterable[t.Any], batch_size: int) -> t.Iterable[list[t.Any]]:
    """Yield lists of ``batch_size`` items from ``iterable``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    batch: list[t.Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(slots=True)
class MetricResult:
    name: str
    value: float
    metadata: dict[str, t.Any] | None = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for rows of ``a`` and ``b``."""

    if a.shape != b.shape:
        raise ValueError("Inputs must have matching shapes")
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    denom = np.clip(a_norm * b_norm, a_min=1e-12, a_max=None)
    sim = (a * b).sum(axis=1, keepdims=True) / denom
    return sim.squeeze(-1)


@contextmanager
def numpy_seed(seed: RNGSeed):
    """Context manager that temporarily sets the NumPy random seed."""

    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def log_uniform(min_value: float, max_value: float, num: int) -> np.ndarray:
    """Return ``num`` points spaced uniformly in log space between the bounds."""

    if min_value <= 0 or max_value <= 0:
        raise ValueError("Bounds must be positive for log uniform spacing")
    if num < 2:
        raise ValueError("num must be >= 2")
    return np.exp(np.linspace(math.log(min_value), math.log(max_value), num=num))


__all__ = [
    "RNGSeed",
    "MetricResult",
    "batched",
    "cosine_similarity",
    "ensure_path",
    "log_uniform",
    "numpy_seed",
    "set_random_seed",
]

