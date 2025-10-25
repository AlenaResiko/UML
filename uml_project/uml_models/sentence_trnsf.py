from sentence_transformers import SentenceTransformer, models
from torch import nn


def build_model(base_model_name: str, target_dim: int, DEVICE: str = "cpu"):
    """
    Build a SentenceTransformer where we append a Dense projection after pooling
    to obtain exactly `target_dim` output dimensions.
    """
    # Transformer (encoder)
    word_embedding_model = models.Transformer(base_model_name, max_seq_length=128)
    # Mean pooling (or use cls pooling if you prefer)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,  # sentence embedding = mean of word
        # ↪embeddings in sentence, that's rule of thumb for sentence similarity but if
        # ↪we want to do classification prob cls is better
        pooling_mode_cls_token=False,  # instead of cls or max use mean here;
        # ABOBA: can vary and see changes
        pooling_mode_max_tokens=False,
    )
    # The pooler (projector)
    dense = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=target_dim,
        activation_function=nn.Tanh(),
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense], device=DEVICE)  # type: ignore
    return model
