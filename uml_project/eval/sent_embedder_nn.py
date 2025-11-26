import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from uml_project.torch_device import DEVICE_LITERAL


class SentenceEmbedder(nn.Module):
    """
    BERT-based sentence embedder with mean-pooling or cls pooling.
    The model is built so passing the same input twice with dropout yields different embeddings
    (SimCSE unsupervised trick).

    We explicitly separate:
      - the encoder (self.enc): maps tokens -> hidden states
      - the pooler/projection (self.projection): maps hidden states -> low-dim sentence embedding

    Following Wang et al. (2023), the performance loss in low-dimensional settings can be
    understood as the sum of:
      - Performance Loss of the Encoder
      - Performance Loss of the Pooler

    This motivates training schemes where we freeze one part and optimize the other.
    """

    def __init__(
        self,
        model_name="bert-base-uncased",
        pooling: str = "mean",
        device: DEVICE_LITERAL = "cuda",
        proj_size: int | None = None,
    ):
        super().__init__()
        assert pooling in ("mean", "cls")
        self.device = device
        self.enc = AutoModel.from_pretrained(model_name).to(device)
        self.pooling = pooling
        hidden_size = self.enc.config.hidden_size

        # Projection head = "pooler" in the Wang et al. sense
        if proj_size is None or proj_size == hidden_size:
            # No separate projection if proj_size is None or matches hidden_size
            self.projection = nn.Identity()
            self.proj_size = hidden_size
        else:
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # Optional intermediate layer
                nn.Tanh(),  # Or ReLU or other activation
                nn.Linear(hidden_size, proj_size),
            )
            self.proj_size = proj_size

        self.to(device)

    # --- NEW: convenience methods to (un)freeze encoder vs pooler ---

    def freeze_encoder(self):
        for p in self.enc.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.enc.parameters():
            p.requires_grad = True

    def freeze_pooler(self):
        for p in self.projection.parameters():
            p.requires_grad = False

    def unfreeze_pooler(self):
        for p in self.projection.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask, return_encoder_output: bool = False):
        """
        Encode batch of tokenized sentences and return normalized embeddings.

        Args:
            input_ids, attention_mask: tensors (batch, seq_len)
            return_encoder_output:
                - False (default): return final pooled+projected embedding (pooler output)
                - True: return a tuple (encoder_embedding, pooler_embedding)
                  where encoder_embedding is the pooled encoder output before projection,
                  and pooler_embedding is the final low-dim embedding.

        Returns:
            If return_encoder_output is False:
                z: L2-normalized embeddings (batch, proj_size)
            If True:
                (enc_norm, z): both L2-normalized, shapes (batch, H) and (batch, proj_size)
        """
        outputs = self.enc(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True
        )
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        if self.pooling == "mean":
            # mean pooling over tokens with attention mask
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
            summed = (last_hidden * mask).sum(1)  # (B, H)
            denom = mask.sum(1).clamp(min=1e-9)
            pooled = summed / denom  # encoder-level sentence embedding
        elif self.pooling == "cls":
            pooled = last_hidden[:, 0]
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        z = self.projection(pooled)
        z = F.normalize(z, p=2, dim=1)

        if return_encoder_output:
            enc_norm = F.normalize(pooled, p=2, dim=1)
            return enc_norm, z

        return z

    def get_sentence_embedding_dimension(self):
        """Returns the dimension of the final sentence embeddings after pooling and projection."""
        return self.proj_size
