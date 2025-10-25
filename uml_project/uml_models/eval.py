import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, evaluation
from sklearn.decomposition import PCA


# -------------------------
# Evaluation helpers
# -------------------------
def evaluate_sts(model: SentenceTransformer, examples: list[InputExample]):
    """Compute Pearson & Spearman on STS-style examples using sentence-transformers evaluator utilities."""
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(examples, name="sts-eval")
    return evaluator(model)


def compute_pca_explained_variance(embeddings: np.ndarray, n_components: int = 50):
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)
    return pd.DataFrame({"explained_var": explained, "cum_explained_var": cum})


def participation_ratio(singular_values: np.ndarray) -> float:
    """
    Participation ratio = (sum_i s_i^2)^2 / sum_i s_i^4
    When s_i are singular values of embedding matrix (or eigenvalues).
    Higher -> more dimensions effectively used.
    """
    s2 = singular_values**2
    num = (s2.sum()) ** 2
    den = (s2**2).sum()
    if den == 0:
        return 0.0
    return num / den
