import random

import torch
import tqdm
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertModel

from uml_project.eval.sent_embedder_nn import SentenceEmbedder
from uml_project.eval.sentence_helper import ContrastiveSentenceDataset
from uml_project.torch_device import DEVICE_TORCH_STR


def encode_sentences(
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


def encode_sentences2(
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
    sentences1,
    sentences2,
    scores,
    batch_size: int = 64,
    max_length: int = 64,
):
    assert len(sentences1) == len(sentences2) == len(scores)

    emb1 = encode_sentences2(
        model, tokenizer, sentences1, batch_size=batch_size, max_length=max_length, show_progress=True
    )
    emb2 = encode_sentences2(
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
