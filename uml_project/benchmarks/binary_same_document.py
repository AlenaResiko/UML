"""Binary classification benchmark: do two sentences originate from the same document?"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import random
import re
from typing import Sequence

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn import metrics

from .utils import MetricResult, RNGSeed, set_random_seed


MIN_WORDS_PER_SENTENCE = 3


@dataclass(slots=True)
class Document:
    domain: str
    doc_id: str
    sentences: list[str]


@dataclass(slots=True)
class SameDocumentPair:
    sentence_a: str
    sentence_b: str
    label: int
    domain: str


@dataclass(slots=True)
class BinaryBenchmarkConfig:
    pairs_per_domain: int = 2048
    seed: RNGSeed = 7
    batch_size: int = 64
    show_progress_bar: bool = False
    threshold: float = 0.5


def simple_sentence_tokenize(text: str) -> list[str]:
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [c.strip() for c in candidates if len(c.split()) >= MIN_WORDS_PER_SENTENCE]
    return sentences


def _load_scientific_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        raw_text = data.get("raw_tex") or data.get("text") or ""
        sentences = simple_sentence_tokenize(raw_text)
        if sentences:
            docs.append(Document(domain="scientific", doc_id=path.stem, sentences=sentences))
    return docs


def _load_music_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.glob("**/*.txt")):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        sentences = simple_sentence_tokenize(text)
        if sentences:
            docs.append(Document(domain="music", doc_id=str(path.relative_to(root)), sentences=sentences))
    return docs


def _load_imdb_documents(root: Path, *, split: str = "train") -> list[Document]:
    data_files = {split: str(root / f"{split}.parquet")}
    dataset = load_dataset("parquet", data_files=data_files, split=split)
    docs: list[Document] = []
    for idx, text in enumerate(dataset["text"]):
        sentences = simple_sentence_tokenize(text)
        if sentences:
            docs.append(Document(domain="imdb", doc_id=f"{split}-{idx}", sentences=sentences))
    return docs


DEFAULT_DOMAIN_LOADERS = {
    "scientific": _load_scientific_documents,
    "music": _load_music_documents,
    "imdb": _load_imdb_documents,
}


def load_documents(data_root: Path | str = "data", domains: Sequence[str] | None = None) -> list[Document]:
    root = Path(data_root)
    selected_domains = list(domains) if domains is not None else list(DEFAULT_DOMAIN_LOADERS)
    documents: list[Document] = []
    for domain in selected_domains:
        loader = DEFAULT_DOMAIN_LOADERS.get(domain)
        if loader is None:
            raise ValueError(f"Unknown domain '{domain}'. Available: {sorted(DEFAULT_DOMAIN_LOADERS)}")
        domain_root = root / domain
        if not domain_root.exists():
            continue
        if domain == "imdb":
            documents.extend(loader(domain_root, split="train"))
        else:
            documents.extend(loader(domain_root))
    return documents


def _sample_pairs_for_domain(
    documents: Sequence[Document],
    *,
    pairs_per_domain: int,
    rng: random.Random,
) -> list[SameDocumentPair]:
    by_doc = [doc for doc in documents if len(doc.sentences) >= 2]
    if not by_doc:
        return []

    positives: list[SameDocumentPair] = []
    while len(positives) < pairs_per_domain:
        doc = rng.choice(by_doc)
        idxs = rng.sample(range(len(doc.sentences)), 2)
        a, b = doc.sentences[idxs[0]], doc.sentences[idxs[1]]
        positives.append(SameDocumentPair(sentence_a=a, sentence_b=b, label=1, domain=doc.domain))

    all_sentences = [(doc, sent) for doc in documents for sent in doc.sentences]
    negatives: list[SameDocumentPair] = []
    while len(negatives) < pairs_per_domain:
        (doc_a, sent_a), (doc_b, sent_b) = rng.sample(all_sentences, 2)
        if doc_a.doc_id == doc_b.doc_id:
            continue
        negatives.append(SameDocumentPair(sentence_a=sent_a, sentence_b=sent_b, label=0, domain=doc_a.domain))

    return positives + negatives


def build_same_document_pairs(
    documents: Sequence[Document],
    *,
    config: BinaryBenchmarkConfig,
) -> list[SameDocumentPair]:
    rng = random.Random(config.seed)
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        grouped[doc.domain].append(doc)

    pairs: list[SameDocumentPair] = []
    for domain, docs in grouped.items():
        domain_pairs = _sample_pairs_for_domain(docs, pairs_per_domain=config.pairs_per_domain, rng=rng)
        pairs.extend(domain_pairs)
    rng.shuffle(pairs)
    return pairs


@dataclass(slots=True)
class BinaryEvaluationResult:
    domain_metrics: dict[str, dict[str, float]]
    overall_metrics: dict[str, float]
    threshold: float
    total_pairs: int

    def as_metric_list(self) -> list[MetricResult]:
        results = [
            MetricResult(name=f"binary/{metric}", value=value, metadata={"domain": "all"})
            for metric, value in self.overall_metrics.items()
        ]
        for domain, metric_values in self.domain_metrics.items():
            for metric, value in metric_values.items():
                results.append(
                    MetricResult(name=f"binary/{metric}", value=value, metadata={"domain": domain})
                )
        return results


@dataclass(slots=True)
class DomainStatistics:
    domain: str
    num_documents: int
    total_sentences: int
    avg_sentences_per_document: float


def summarise_documents(documents: Sequence[Document]) -> list[DomainStatistics]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        grouped[doc.domain].append(doc)

    summaries: list[DomainStatistics] = []
    for domain, docs in grouped.items():
        sentence_count = sum(len(doc.sentences) for doc in docs)
        avg_sentences = sentence_count / len(docs) if docs else 0.0
        summaries.append(
            DomainStatistics(
                domain=domain,
                num_documents=len(docs),
                total_sentences=sentence_count,
                avg_sentences_per_document=avg_sentences,
            )
        )
    return summaries


def _compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy": float(metrics.accuracy_score(labels, preds)),
        "precision": float(metrics.precision_score(labels, preds, zero_division=0)),
        "recall": float(metrics.recall_score(labels, preds, zero_division=0)),
        "f1": float(metrics.f1_score(labels, preds, zero_division=0)),
    }


def evaluate_same_document_binary(
    model: SentenceTransformer,
    documents: Sequence[Document],
    *,
    config: BinaryBenchmarkConfig | None = None,
) -> BinaryEvaluationResult:
    cfg = config or BinaryBenchmarkConfig()
    set_random_seed(cfg.seed)

    pairs = build_same_document_pairs(documents, config=cfg)
    sentences = [pair.sentence_a for pair in pairs] + [pair.sentence_b for pair in pairs]
    embeddings = model.encode(
        sentences,
        batch_size=cfg.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=cfg.show_progress_bar,
    )
    embeddings = embeddings.reshape(len(pairs), 2, -1)
    scores = (embeddings[:, 0, :] * embeddings[:, 1, :]).sum(axis=1)

    labels = np.array([pair.label for pair in pairs], dtype=int)
    overall_metrics = _compute_metrics(labels, scores, cfg.threshold)

    domain_scores: dict[str, list[float]] = defaultdict(list)
    domain_labels: dict[str, list[int]] = defaultdict(list)
    for pair, score in zip(pairs, scores):
        domain_scores[pair.domain].append(float(score))
        domain_labels[pair.domain].append(pair.label)

    domain_metrics = {
        domain: _compute_metrics(np.array(domain_labels[domain]), np.array(domain_scores[domain]), cfg.threshold)
        for domain in domain_scores
    }

    return BinaryEvaluationResult(
        domain_metrics=domain_metrics,
        overall_metrics=overall_metrics,
        threshold=cfg.threshold,
        total_pairs=len(pairs),
    )


__all__ = [
    "BinaryBenchmarkConfig",
    "BinaryEvaluationResult",
    "Document",
    "SameDocumentPair",
    "DomainStatistics",
    "build_same_document_pairs",
    "evaluate_same_document_binary",
    "load_documents",
    "summarise_documents",
]

