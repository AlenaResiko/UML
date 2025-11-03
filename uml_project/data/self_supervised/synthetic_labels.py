"""
generate_synthetic_position_labels() takes in a numpy array and generates synthetic position labels, based
on index similarity.
"""

from functools import cache
from typing import Any
from numpy.typing import NDArray
import numpy as np


def generate_synthetic_position_labels(arr: NDArray[Any], target_index: int):
    """
    Generate synthetic position labels for a numpy array based on index similarity.
    """
    cutoff = _calculate_similarity_cutoff(size=arr.shape[0])
    distances = np.arange(arr.shape[0]) - target_index
    return np.abs(distances) <= cutoff


@cache
def _calculate_similarity_cutoff(size: int, min: int = 5, max: int = 15, p: float = 0.01):
    """
    Calculate the similarity cutoff based on array size.
    """
    return float(np.clip(size * p, min, max))


# def _similarity(i: int, j: int) -> float:
#     return 1.0 / (1.0 + abs(i - j))  # Example: inverse of index distance
