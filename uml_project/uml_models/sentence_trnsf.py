import pathlib

from sentence_transformers import SentenceTransformer, evaluation, losses, models
from torch import nn
from torch.utils.data import DataLoader

from uml_project.uml_models.utils import *

# from datasets import Dataset


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


def train_pooler_then_finetune(
    model: SentenceTransformer,
    train_examples: list[InputExample],
    val_examples: list[InputExample],
    out_dir: pathlib.Path | str,
    notebook_vars: NotebookVars,
):
    # Step A: train pooler only (encoder frozen)
    freeze_encoder_only(model)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader = DataLoader(train_examples, batch_size=notebook_vars["BATCH_SIZE"], shuffle=True)  # type: ignore
    # Use CosineSimilarityLoss for contrastive-style or MSELoss for regression
    # (STS)
    loss_fct = losses.CosineSimilarityLoss(model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name="sts-val"
    )  # note this benchmark compares against human-annotated similarity scores;
    # ABOBA: we can't self-annotate sim for Swift or Verma so we can't get
    # encoder error
    model.fit(
        train_objectives=[(train_dataloader, loss_fct)],
        evaluator=evaluator,
        epochs=notebook_vars["EPOCHS_POOLER"],
        warmup_steps=100,
        output_path=str(out_dir / "stepA_pooler_only"),
        optimizer_params={"lr": notebook_vars["POOLER_LR"]},
    )
    # Step B: unfreeze encoder and finetune whole model
    unfreeze_model_weights(model)
    # Recreate dataloader (sentence-transformers expects InputExamples in an
    # in-memory list)
    train_dataloader = DataLoader(train_examples, batch_size=notebook_vars["BATCH_SIZE"], shuffle=True)  # type: ignore
    loss_fct2 = losses.MultipleNegativesRankingLoss(
        model
    )  # good objective for contrastive training (requires positive pairs)
    model.fit(
        train_objectives=[(train_dataloader, loss_fct2)],
        evaluator=evaluator,
        epochs=notebook_vars["EPOCHS_FINETUNE"],
        warmup_steps=100,
        output_path=str(out_dir / "stepB_finetune"),
        optimizer_params={"lr": notebook_vars["FINETUNE_LR"]},
    )
    return model
