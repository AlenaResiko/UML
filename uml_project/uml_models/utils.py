import typing as t

import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, models


def df_to_input_examples(df: pd.DataFrame, use_labels=True):
    # vectorized: convert columns to arrays and build InputExample list via comprehension
    s1 = df["sentence1"].astype(str).to_numpy()
    s2 = df["sentence2"].astype(str).to_numpy()
    if use_labels:
        labels = (df["label"].astype(float) / 5.0).to_numpy()
        ex = [InputExample(texts=[a, b], label=float(l)) for a, b, l in zip(s1, s2, labels)]
    else:
        ex = [InputExample(texts=[a, b]) for a, b in zip(s1, s2)]
    return ex


def freeze_encoder_only(model: SentenceTransformer):
    # SentenceTransformer stores modules in model._modules (OrderedDict-like). The transformer is index 0.
    # Simpler: freeze parameters in modules that are instances of models.Transformer
    for module in model._modules.values():
        if isinstance(module, models.Transformer):
            for p in module.parameters():
                p.requires_grad = False


def freeze_model_weights(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_weights(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True


class NotebookVars(t.TypedDict, total=True):
    DEVICE: str
    BATCH_SIZE: int
    EPOCHS_POOLER: int
    EPOCHS_FINETUNE: int
    FINETUNE_LR: float
    POOLER_LR: float
