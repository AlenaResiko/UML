"""Utilities for discovering and loading fine-tuned sentence models.

This module standardises how benchmark code accesses locally saved models.
Model trainers are asked to follow the naming convention discussed in the
project brief: ``{base_model}_{training_dataset}_{embedding_dim}_{version}``.
For example ``bert-base-uncased_ALL_128_v0``.  All artefacts for the model
should live inside ``models/{model_name}/`` in the repository.

At the moment we focus on SentenceTransformer style checkpoints because the
fine-tuning notebooks save models in that format.  The helper below still keeps
the parsing utilities generic enough that we can extend the loader to other
model types later if required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import typing as t

from sentence_transformers import SentenceTransformer

ModelPathLike = t.Union[str, Path]


MODEL_NAME_PATTERN = re.compile(
    r"^(?P<backbone>[\w-]+)_(?P<dataset>[\w-]+)_(?P<embedding_dim>\d+)_(?P<version>v\d+)$"
)


@dataclass(frozen=True, slots=True)
class ParsedModelName:
    """Structured view of a model name following the agreed convention."""

    backbone: str
    dataset: str
    embedding_dim: int
    version: str

    @property
    def tag(self) -> str:
        """Return the canonical tag for this model.

        The tag matches the folder name inside ``models/``.
        """

        return f"{self.backbone}_{self.dataset}_{self.embedding_dim}_{self.version}"


@dataclass(slots=True)
class ModelArtifacts:
    """Wrap a loaded model together with optional metadata."""

    name: ParsedModelName
    path: Path
    model: SentenceTransformer
    metadata: dict[str, t.Any]

    def to_summary(self) -> dict[str, t.Any]:
        """Return a dictionary with high-level attributes for quick reporting."""

        config = self.metadata.get("config", {})
        return {
            "model_name": self.name.tag,
            "backbone": self.name.backbone,
            "training_dataset": self.name.dataset,
            "embedding_dim": self.name.embedding_dim,
            "version": self.name.version,
            "sentence_embedding_dim": self.model.get_sentence_embedding_dimension(),
            "metadata": self.metadata,
            "config": config,
        }


def parse_model_name(model_name: str) -> ParsedModelName:
    """Validate and parse a model name.

    Parameters
    ----------
    model_name:
        Folder name (``models/<model_name>``) that should match the agreed
        convention.

    Returns
    -------
    ParsedModelName
        Structured information extracted from the string.

    Raises
    ------
    ValueError
        If the string does not match the expected pattern.
    """

    match = MODEL_NAME_PATTERN.match(model_name)
    if not match:
        raise ValueError(
            "Model name '%s' does not follow the required pattern "
            "{backbone}_{dataset}_{embedding_dim}_{version}." % model_name
        )

    groups = match.groupdict()
    return ParsedModelName(
        backbone=groups["backbone"],
        dataset=groups["dataset"],
        embedding_dim=int(groups["embedding_dim"]),
        version=groups["version"],
    )


def list_available_models(models_dir: ModelPathLike = "models") -> list[ParsedModelName]:
    """List all models in ``models_dir`` that follow the naming convention."""

    base_path = Path(models_dir)
    if not base_path.exists():
        return []

    parsed: list[ParsedModelName] = []
    for entry in base_path.iterdir():
        if not entry.is_dir():
            continue
        try:
            parsed.append(parse_model_name(entry.name))
        except ValueError:
            # Skip folders that are not part of the canonical registry.
            continue
    parsed.sort(key=lambda item: (item.backbone, item.dataset, item.embedding_dim, item.version))
    return parsed


def _load_metadata(model_path: Path) -> dict[str, t.Any]:
    metadata_file = model_path / "metadata.json"
    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Surface invalid metadata to the caller while keeping the loader usable.
                raise ValueError(f"Invalid JSON metadata in {metadata_file}")
    return {}


def load_sentence_transformer(
    model_name: str,
    *,
    models_dir: ModelPathLike = "models",
    device: str | None = None,
    strict: bool = True,
) -> ModelArtifacts:
    """Load a sentence-transformer style checkpoint and return structured artefacts.

    Parameters
    ----------
    model_name:
        Name of the model folder, following the convention.  The function will
        raise an error if the pattern is invalid.
    models_dir:
        Base directory that contains all model folders.  Defaults to ``models``
        at the repository root.
    device:
        Optional device override passed to :class:`SentenceTransformer`.  When
        omitted the library decides (CPU / CUDA).
    strict:
        When ``True`` (default) the loader will fail fast if the folder does not
        exist.  When ``False`` the function returns ``None`` instead of raising.

    Returns
    -------
    ModelArtifacts
        Wrapper containing the instantiated :class:`SentenceTransformer` and
        auxiliary metadata.
    """

    parsed_name = parse_model_name(model_name)
    model_path = Path(models_dir) / parsed_name.tag

    if not model_path.exists():
        if strict:
            raise FileNotFoundError(f"Model artefacts not found at {model_path}")
        return None  # type: ignore[return-value]

    model = SentenceTransformer(str(model_path), device=device)
    metadata = _load_metadata(model_path)

    # Add the raw config file if available; ignore parse errors silently because
    # they should not prevent evaluations from running.
    config_path = model_path / "config.json"
    if config_path.exists() and "config" not in metadata:
        try:
            with config_path.open("r", encoding="utf-8") as f:
                metadata["config"] = json.load(f)
        except json.JSONDecodeError:
            metadata.setdefault("config", str(config_path))

    return ModelArtifacts(name=parsed_name, path=model_path, model=model, metadata=metadata)


__all__ = [
    "MODEL_NAME_PATTERN",
    "ModelArtifacts",
    "ParsedModelName",
    "list_available_models",
    "load_sentence_transformer",
    "parse_model_name",
]

