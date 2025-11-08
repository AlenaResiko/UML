"""
generate_synthetic_position_labels() takes in a numpy array and generates synthetic position labels, based
on index similarity.

TODO do we want sigmoid-based similarity rather than linear?
"""

from __future__ import annotations
from functools import cache
from typing import Any
from numpy.typing import NDArray
import numpy as np
import pandas as pd


try:
    from typing import TypedDict, Unpack
except Exception:  # pragma: no cover - fallback for older Python
    from typing_extensions import TypedDict, Unpack


def synthetic_similarity_labels(arr: NDArray[Any], **kwargs: Unpack[SimilarityCutoffParams]) -> pd.DataFrame:
    """
    Generate synthetic position labels for a numpy array based on index similarity.
    Accepts optional kwargs: min, max, p (passed through to _calculate_similarity_cutoff).
    """
    cutoff = _calculate_similarity_cutoff(size=arr.shape[0], **kwargs)
    idx = np.arange(arr.shape[0])
    ret = pd.DataFrame({"arr": arr, "idx": idx, "min_idx": idx - cutoff, "max_idx": idx + cutoff})
    return ret


class SimilarityCutoffParams(TypedDict, total=False):
    min: int
    max: int
    p: float


@cache
def _calculate_similarity_cutoff(size: int, min: int = 3, max: int = 15, p: float = 0.01):
    """
    Calculate the similarity cutoff based on array size.
    """
    return float(np.clip(size * p, min, max))
