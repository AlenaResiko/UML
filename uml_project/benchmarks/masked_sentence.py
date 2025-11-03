"""Masked sentence (dropout) evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn import metrics

from .utils import MetricResult, RNGSeed, set_random_seed


@dataclass(slots=True)
class MaskedSentenceExample:
    left_context: list[str]
    right_context: list[str]
    candidates: list[str]
    correct_index: int
    domain: str = "generic"
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class MaskedSentenceConfig:
    batch_size: int = 32
    seed: RNGSeed = 11
    show_progress_bar: bool = False
    average: str = "macro"


@dataclass(slots=True)
class MaskedSentenceResult:
    overall_metrics: dict[str, float]
    domain_metrics: dict[str, dict[str, float]]
    total_examples: int

    def as_metrics(self) -> list[MetricResult]:
        rows = [
            MetricResult(name=f"masked/{metric}", value=value, metadata={"domain": "all"})
            for metric, value in self.overall_metrics.items()
        ]
        for domain, metric_values in self.domain_metrics.items():
            for metric, value in metric_values.items():
                rows.append(MetricResult(name=f"masked/{metric}", value=value, metadata={"domain": domain}))
        return rows


def load_masked_sentence_examples(path: Path | str) -> list[MaskedSentenceExample]:
    """Load examples from a JSON Lines file.

    Each row should contain the keys ``left_context`` (list[str]), ``right_context``
    (list[str]), ``candidates`` (list[str]) and ``answer`` (int index).
    """

    examples: list[MaskedSentenceExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            examples.append(
                MaskedSentenceExample(
                    left_context=list(record.get("left_context", [])),
                    right_context=list(record.get("right_context", [])),
                    candidates=list(record.get("candidates", [])),
                    correct_index=int(record["answer"]),
                    domain=record.get("domain", "generic"),
                    metadata=record.get("metadata"),
                )
            )
    return examples


def _encode_sentences(
    model: SentenceTransformer,
    sentences: Sequence[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    return model.encode(
        list(sentences),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )


def _candidate_scores(
    candidate_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
) -> np.ndarray:
    if context_embeddings.size == 0:
        return np.zeros(candidate_embeddings.shape[0])
    sims = candidate_embeddings @ context_embeddings.T
    return sims.mean(axis=1)


def evaluate_masked_sentence(
    model: SentenceTransformer,
    examples: Sequence[MaskedSentenceExample],
    *,
    config: MaskedSentenceConfig | None = None,
) -> MaskedSentenceResult:
    cfg = config or MaskedSentenceConfig()
    set_random_seed(cfg.seed)

    unique_sentences: dict[str, str] = {}
    for example in examples:
        for sentence in (*example.left_context, *example.right_context, *example.candidates):
            unique_sentences.setdefault(sentence, sentence)

    sentence_list = list(unique_sentences)
    embeddings = _encode_sentences(
        model,
        sentence_list,
        batch_size=cfg.batch_size,
        show_progress_bar=cfg.show_progress_bar,
    )
    embedding_lookup = {sentence: emb for sentence, emb in zip(sentence_list, embeddings)}

    y_true: list[int] = []
    y_pred: list[int] = []
    domain_buckets: dict[str, list[int]] = {}
    domain_predictions: dict[str, list[int]] = {}

    for example in examples:
        if not example.candidates:
            raise ValueError("Masked sentence example has no candidate sentences")
        candidate_vectors = np.stack([embedding_lookup[s] for s in example.candidates])
        context_items = [embedding_lookup[s] for s in (*example.left_context, *example.right_context)]
        if context_items:
            context_vectors = np.stack(context_items)
        else:
            context_vectors = np.zeros((0, candidate_vectors.shape[1]))
        scores = _candidate_scores(candidate_vectors, context_vectors)
        predicted_index = int(np.argmax(scores))

        y_true.append(example.correct_index)
        y_pred.append(predicted_index)

        domain_buckets.setdefault(example.domain, []).append(example.correct_index)
        domain_predictions.setdefault(example.domain, []).append(predicted_index)

    labels = sorted(set(y_true) | set(y_pred))
    overall_metrics = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, labels=labels, average=cfg.average, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, labels=labels, average=cfg.average, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, labels=labels, average=cfg.average, zero_division=0)),
    }

    domain_metrics: dict[str, dict[str, float]] = {}
    for domain, truths in domain_buckets.items():
        preds = domain_predictions[domain]
        domain_labels = sorted(set(truths) | set(preds))
        domain_metrics[domain] = {
            "accuracy": float(metrics.accuracy_score(truths, preds)),
            "precision": float(
                metrics.precision_score(truths, preds, labels=domain_labels, average=cfg.average, zero_division=0)
            ),
            "recall": float(
                metrics.recall_score(truths, preds, labels=domain_labels, average=cfg.average, zero_division=0)
            ),
            "f1": float(
                metrics.f1_score(truths, preds, labels=domain_labels, average=cfg.average, zero_division=0)
            ),
        }

    return MaskedSentenceResult(
        overall_metrics=overall_metrics,
        domain_metrics=domain_metrics,
        total_examples=len(examples),
    )


__all__ = [
    "MaskedSentenceConfig",
    "MaskedSentenceExample",
    "MaskedSentenceResult",
    "evaluate_masked_sentence",
    "load_masked_sentence_examples",
]

