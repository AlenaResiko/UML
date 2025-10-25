import pandas as pd
from sentence_transformers import SentenceTransformer, models

from sentence_transformers import InputExample


def df_to_input_examples(df: pd.DataFrame, use_labels=True):
    ex = []
    for _, r in df.iterrows():
        if use_labels:
            # STS labels are 0..5; evaluator typically expects 0..1
            ex.append(InputExample(texts=[r["sentence1"], r["sentence2"]], label=float(r["label"]) / 5.0))
        else:
            ex.append(InputExample(texts=[r["sentence1"], r["sentence2"]]))
    return ex


def freeze_encoder_only(model: SentenceTransformer):
    # SentenceTransformer stores modules in model._modules (OrderedDict-like). The transformer is index 0.
    # Simpler: freeze parameters in modules that are instances of models.Transformer
    for module in model._modules.values():
        if isinstance(module, models.Transformer):
            for p in module.parameters():
                p.requires_grad = False
