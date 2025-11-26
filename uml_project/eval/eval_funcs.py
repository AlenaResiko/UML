import random

import numpy as np
import torch
import tqdm
from scipy.stats import spearmanr
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertModel

from uml_project.eval.sent_embedder_nn import SentenceEmbedder
from uml_project.eval.sentence_helper import ContrastiveSentenceDataset
from uml_project.torch_device import DEVICE_TORCH_STR


def lalign(x: Tensor, y: Tensor, alpha: float = 2.0):
    # x, y: [bsz, d] normalized embeddings
    return (x - y).norm(dim=1).pow(alpha).mean()


def lunif(x: Tensor, t: float = 2.0):
    # x: [N, d] normalized embeddings
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def encode_sentences_depr(
    model: SentenceEmbedder,
    tokenizer: BertModel,
    sentences: tuple[list[str], list[str]],
    batch_size: int = 64,
    max_length: int = 64,
    show_progress: bool = False,
):
    """
    Returns a (N, d) numpy array of L2-normalized embeddings.
    """
    model.eval()
    all_embs = []

    with torch.no_grad():
        raw_iter = range(0, len(sentences), batch_size)
        iterable = tqdm.tqdm(raw_iter) if show_progress else raw_iter
        for i in iterable:
            batch_sents = sentences[i : i + batch_size]
            toks = tokenizer(batch_sents, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
                DEVICE_TORCH_STR
            )

            emb = model(
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
                return_encoder_output=False,  # just projected embedding
            )  # (B, d), already normalized in forward()
            all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0).numpy()


def encode_sentences(
    model: SentenceEmbedder,
    tokenizer: BertModel,
    sentences: tuple[list[str], list[str]] | list[str],
    batch_size: int = 64,
    max_length: int = 64,
    show_progress: bool = True,
):
    """
    Encode a list of strings into [N, dim] numpy embeddings.
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    model.eval()
    all_embs = []

    dataloader = DataLoader(sentences, batch_size=batch_size)  # type: ignore
    iterator = tqdm(dataloader) if show_progress else dataloader  # type: ignore

    with torch.no_grad():
        for batch in iterator:
            enc = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE_TORCH_STR) for k, v in enc.items()}

            # SentenceEmbedder forward returns normalized embeddings already
            z = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_encoder_output=False,
            )  # [B, dim]

            all_embs.append(z.cpu())

    embs = torch.cat(all_embs, dim=0)  # [N, dim]
    return embs.numpy()


def evaluate_sts_spearman(
    model: SentenceEmbedder,
    tokenizer: BertModel,
    sentences1: list[str],
    sentences2: list[str],
    scores: Tensor,
    batch_size: int = 64,
    max_length: int = 64,
):
    assert len(sentences1) == len(sentences2) == len(scores)

    emb1 = encode_sentences(
        model, tokenizer, sentences1, batch_size=batch_size, max_length=max_length, show_progress=True
    )
    emb2 = encode_sentences(
        model, tokenizer, sentences2, batch_size=batch_size, max_length=max_length, show_progress=True
    )

    cos_sim = (emb1 * emb2).sum(axis=1)  # cosine = dot because normalized

    rho, pval = spearmanr(scores, cos_sim)
    return {
        "spearman_rho": float(rho),  # type: ignore
        "p_value": float(pval),  # type: ignore
    }


def sample_pos_pairs_from_docs(
    docs,
    num_pairs: int = 10_000,
    neighbor_window: int = 10,
    num_neighbors: int = 1,
    sample_neighbors_prob: float = 1.0,
):
    dataset = ContrastiveSentenceDataset(
        docs=docs,
        neighbor_window=neighbor_window,
        num_neighbors=num_neighbors,
        sample_neighbors_prob=sample_neighbors_prob,
    )

    pairs = []
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        item = dataset[idx]
        anchor_text = item["anchor_text"]
        for flat_idx in item["neighbor_pos_indices"]:
            neighbor_text = dataset.corpus.get_sentence(flat_idx)
            pairs.append((anchor_text, neighbor_text))
            if len(pairs) >= num_pairs:
                return pairs

    return pairs


def eval_alignment_from_docs(
    model: SentenceEmbedder,
    tokenizer: BertModel,
    docs: list[list[str]],
    num_pairs: int = 10_000,
    neighbor_window: int = 10,
    alpha: float = 2.0,
    batch_size: int = 64,
    max_length: int = 64,
):

    pos_pairs = sample_pos_pairs_from_docs(
        docs,
        num_pairs=num_pairs,
        neighbor_window=neighbor_window,
        num_neighbors=1,
        sample_neighbors_prob=1.0,
    )
    if not pos_pairs:
        raise ValueError("No positive pairs found in docs.")

    s1 = [p[0] for p in pos_pairs]
    s2 = [p[1] for p in pos_pairs]

    emb1 = encode_sentences(
        model,
        tokenizer,
        s1,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=True,
    )
    emb2 = encode_sentences(
        model,
        tokenizer,
        s2,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=True,
    )

    x = torch.from_numpy(emb1)
    y = torch.from_numpy(emb2)

    return lalign(x, y, alpha=alpha).item()


def uniformity_on_sentences(
    model,
    tokenizer,
    sentences,
    t: float = 2.0,
    max_samples: int = 8192,
    batch_size: int = 64,
    max_length: int = 64,
    show_progress: bool = True,
):

    if len(sentences) > max_samples:
        sentences = random.sample(sentences, max_samples)

    emb = encode_sentences(
        model,
        tokenizer,
        sentences,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=show_progress,
    )
    x = torch.from_numpy(emb)
    return lunif(x, t=t).item()


import pandas as pd


def uniformity_from_parquet(
    model: SentenceEmbedder,
    tokenizer: BertModel,
    path: str,
    text_col: str,
    dataset_name: str,
    t: float = 2.0,
):
    df = pd.read_parquet(path)
    sentences = df[text_col].astype(str).tolist()
    print(f"{dataset_name}: {len(sentences)} sentences")

    u = uniformity_on_sentences(
        model,
        tokenizer,
        sentences,
        t=t,
        max_samples=8192,
        batch_size=64,
        max_length=64,
    )
    print(f"Uniformity ({dataset_name}) = {u:.4f}")
    return u


def within_doc_similarity_stats(
    model,
    tokenizer,
    parquet_path: str,
    text_col: str = "sentence",
    doc_col: str = "doc_id",
    batch_size: int = 64,
    max_length: int = 64,
    max_pairs_per_doc: int = 100,
    max_docs: int | None = None,  # if you want to cap docs for speed; or None for all
    device=None,
):
    """
    For a parquet with (doc_id, text), compute cosine similarity statistics
    for random sentence pairs sampled *within the same doc*.

    Returns a dict with { 'num_docs', 'num_pairs', 'mean', 'median', 'std', 'min', 'max' }.
    """
    if device is None:
        device = next(model.parameters()).device

    df = pd.read_parquet(parquet_path)
    df = df[[doc_col, text_col]].dropna()

    # Optionally subsample docs for speed
    doc_ids = df[doc_col].unique().tolist()
    if max_docs is not None and len(doc_ids) > max_docs:
        doc_ids = random.sample(doc_ids, max_docs)

    df = df[df[doc_col].isin(doc_ids)].reset_index(drop=True)

    print(f"Loaded {len(df)} sentences across {len(doc_ids)} docs from {parquet_path}")

    # Encode all sentences once
    sentences = df[text_col].astype(str).tolist()
    embeddings = encode_sentences(
        model,
        tokenizer,
        sentences,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=True,
    )  # [N, dim]

    # Pre-map row index -> embedding
    emb_t = torch.from_numpy(embeddings)  # [N, dim], still normalized

    # Build index lists per doc
    doc_to_indices = {}
    for idx, doc_id in enumerate(df[doc_col].tolist()):
        doc_to_indices.setdefault(doc_id, []).append(idx)

    cos_sims = []

    for doc_id, idxs in doc_to_indices.items():
        if len(idxs) < 2:
            continue  # need at least 2 sentences to make a pair

        # Sample pairs within this doc
        # If doc is small, use all pairs; if large, subsample.
        if len(idxs) * (len(idxs) - 1) // 2 <= max_pairs_per_doc:
            # all unordered pairs
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a = emb_t[idxs[i]]
                    b = emb_t[idxs[j]]
                    cos = torch.dot(a, b).item()  # cosine because normalized
                    cos_sims.append(cos)
        else:
            # random sampling of pairs
            for _ in range(max_pairs_per_doc):
                i, j = random.sample(idxs, 2)
                a = emb_t[i]
                b = emb_t[j]
                cos = torch.dot(a, b).item()
                cos_sims.append(cos)

    if not cos_sims:
        raise ValueError("No within-doc pairs found (check doc_id / text columns).")

    cos_sims = np.array(cos_sims, dtype=np.float32)
    stats = {
        "num_docs": len(doc_to_indices),
        "num_pairs": int(len(cos_sims)),
        "mean": float(cos_sims.mean()),
        "median": float(np.median(cos_sims)),
        "std": float(cos_sims.std()),
        "min": float(cos_sims.min()),
        "max": float(cos_sims.max()),
    }
    return stats
