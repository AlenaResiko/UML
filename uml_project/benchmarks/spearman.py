"""Spearman rank correlation evaluation on Semantic Textual Similarity (STS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

from .utils import MetricResult, RNGSeed, batched, set_random_seed


STS_DATASET_NAME = "mteb/stsbenchmark"


@dataclass(slots=True)
class SpearmanEvaluationResult:
    spearman: float
    gold_scores: list[float]
    predicted_scores: list[float]
    split: str
    dataset_name: str

    def to_metric(self) -> MetricResult:
        return MetricResult(
            name=f"spearman@{self.split}",
            value=self.spearman,
            metadata={
                "dataset": self.dataset_name,
                "split": self.split,
                "num_examples": len(self.gold_scores),
            },
        )


def _encode_pairs(
    model: SentenceTransformer,
    sentence_pairs: list[tuple[str, str]],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    texts = [s for pair in sentence_pairs for s in pair]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
    return embeddings.reshape(len(sentence_pairs), 2, -1)


def run_sts_spearman(
    model: SentenceTransformer,
    *,
    dataset_name: str = STS_DATASET_NAME,
    split: str = "test",
    batch_size: int = 64,
    seed: RNGSeed = 13,
    show_progress_bar: bool = False,
) -> SpearmanEvaluationResult:
    """Compute Spearman correlation between cosine similarities and gold labels."""

    set_random_seed(seed)

    dataset = load_dataset(dataset_name, split=split)
    sentence_pairs = list(zip(dataset["sentence1"], dataset["sentence2"]))
    gold_scores = [float(score) for score in dataset["score"]]

    predicted_scores: list[float] = []
    for batch in batched(sentence_pairs, batch_size):
        embeddings = _encode_pairs(
            model,
            batch,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        a = embeddings[:, 0, :]
        b = embeddings[:, 1, :]
        sims = (a * b).sum(axis=1)
        predicted_scores.extend(sims.tolist())

    corr, _ = spearmanr(gold_scores, predicted_scores)

    return SpearmanEvaluationResult(
        spearman=float(corr),
        gold_scores=gold_scores,
        predicted_scores=predicted_scores,
        split=split,
        dataset_name=dataset_name,
    )


__all__ = [
    "STS_DATASET_NAME",
    "SpearmanEvaluationResult",
    "run_sts_spearman",
]

