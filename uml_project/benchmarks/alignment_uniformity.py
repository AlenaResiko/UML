"""Alignment and uniformity metrics for embedding spaces.

The implementation follows ``Understanding Contrastive Representation Learning
through Alignment and Uniformity on the Hypersphere`` (Wang & Isola, 2020).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import MetricResult, RNGSeed, batched, numpy_seed


@dataclass(slots=True)
class AlignmentUniformityConfig:
    alpha: float = 2.0
    temperature: float = 2.0
    batch_size: int = 64
    max_uniformity_samples: int | None = 8192
    seed: RNGSeed = 42
    show_progress_bar: bool = False


@dataclass(slots=True)
class AlignmentUniformityResult:
    alignment: float
    uniformity: float
    positive_pairs: int
    sampled_points: int
    config: AlignmentUniformityConfig

    def as_metrics(self) -> list[MetricResult]:
        return [
            MetricResult(
                name="alignment",
                value=self.alignment,
                metadata={
                    "alpha": self.config.alpha,
                    "positive_pairs": self.positive_pairs,
                },
            ),
            MetricResult(
                name="uniformity",
                value=self.uniformity,
                metadata={
                    "temperature": self.config.temperature,
                    "sampled_points": self.sampled_points,
                },
            ),
        ]


def _encode_pairs(
    model: SentenceTransformer,
    sentence_pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    texts = [s for pair in sentence_pairs for s in pair]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return embeddings.reshape(len(sentence_pairs), 2, -1)


def _alignment(emb_a: np.ndarray, emb_b: np.ndarray, alpha: float) -> float:
    diff = emb_a - emb_b
    norms = np.linalg.norm(diff, axis=1)
    powered = norms ** alpha
    return float(np.mean(powered))


def _uniformity(embeddings: np.ndarray, temperature: float) -> float:
    # pairwise squared distance matrix
    sq_norms = np.sum(embeddings**2, axis=1, keepdims=True)
    sq_dists = sq_norms - 2 * embeddings @ embeddings.T + sq_norms.T
    np.fill_diagonal(sq_dists, np.inf)
    values = np.exp(-temperature * sq_dists)
    finite_values = values[np.isfinite(values)]
    return float(np.log(np.mean(finite_values)))


def evaluate_alignment_uniformity(
    model: SentenceTransformer,
    sentence_pairs: Sequence[tuple[str, str]],
    *,
    config: AlignmentUniformityConfig | None = None,
) -> AlignmentUniformityResult:
    """Evaluate alignment and uniformity on provided positive pairs."""

    cfg = config or AlignmentUniformityConfig()

    if not sentence_pairs:
        raise ValueError("No sentence pairs provided for alignment/uniformity evaluation")

    encoded_pairs = []
    for batch in batched(sentence_pairs, cfg.batch_size):
        embeddings = _encode_pairs(
            model,
            batch,
            batch_size=cfg.batch_size,
            show_progress_bar=cfg.show_progress_bar,
        )
        encoded_pairs.append(embeddings)

    pair_embeddings = np.concatenate(encoded_pairs, axis=0)
    emb_a = pair_embeddings[:, 0, :]
    emb_b = pair_embeddings[:, 1, :]

    alignment_value = _alignment(emb_a, emb_b, cfg.alpha)

    stacked = np.concatenate([emb_a, emb_b], axis=0)

    if cfg.max_uniformity_samples is not None and stacked.shape[0] > cfg.max_uniformity_samples:
        with numpy_seed(cfg.seed):
            indices = np.random.choice(
                stacked.shape[0],
                size=cfg.max_uniformity_samples,
                replace=False,
            )
            uniformity_embeddings = stacked[indices]
    else:
        uniformity_embeddings = stacked

    uniformity_value = _uniformity(uniformity_embeddings, cfg.temperature)

    return AlignmentUniformityResult(
        alignment=alignment_value,
        uniformity=uniformity_value,
        positive_pairs=len(sentence_pairs),
        sampled_points=uniformity_embeddings.shape[0],
        config=cfg,
    )


__all__ = [
    "AlignmentUniformityConfig",
    "AlignmentUniformityResult",
    "evaluate_alignment_uniformity",
]

