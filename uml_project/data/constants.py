import os
import typing as t
from pathlib import Path

# PATHS
DATA_DIR = Path(os.environ.get("DATA_DIR") or Path(__file__).parent.parent.parent).resolve() / "data"
MUSIC_DIR = DATA_DIR / "music/"
SCRATCH_DIR = DATA_DIR / "scratch/"  # for temporary files

# Scientific papers
SCIENTIFIC_DIR = DATA_DIR / "scientific/"
SCIENTIFIC_REGISTRY = SCIENTIFIC_DIR / "REGISTRY.json"  # JSON file with list of paper URLs

# HuggingFace
HF_DIR = DATA_DIR / "huggingface/"  # for HuggingFace datasets/models
HF_REGISTRY = HF_DIR / "REGISTRY.json"  # JSON file with list of datasets/models

# KAGGLE
TAYLOR_SWIFT_KAGGLE = "ishikajohari/taylor-swift-all-lyrics-30-albums"

# NLTK
NLTK_DATA_DIR = DATA_DIR / "nltk/"  # for NLTK

# GENIUS
GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN")  # Store in .env file


# Dtypes
class TexSourceDict(t.TypedDict, total=True):
    id: str  # e.g. "2503.19280v1"
    files: list[dict[str, str]]  # {"name": str, "text": str}
    main: str | None  # main .tex file name
    title: str | None  # extracted title, if any


class ArxivSearchResultDict(t.TypedDict, total=True):
    id: str | None  # e.g. "2503.19280"
    title: str
    authors: list[str]
    year: int | None  # e.g. 2025
    published: str | t.Any | None  # e.g. "2025-03-28T17:59:59Z"
    summary: str  # same as abstract
    categories: list[str]  # e.g. ["cs.LG","stat.ML"]
    url_abs: str | t.Any | None  # e.g. "https://arxiv.org/abs/2503.19280"
    url_pdf: str | t.Any | None  # e.g. "https://arxiv.org/pdf/2503.19280.pdf"


class ArxivRegistry(t.TypedDict, total=True):
    arxiv: list[str]  # list of arXiv abs URLs


class HuggingFaceRegistry(t.TypedDict, total=True):
    datasets: list[str]  # list of dataset repo IDs, e.g. ["imdb", "squad", "org/repo_id"]
    # models: list[str]  # list of model repo IDs, e.g. ["gpt2", "bert-base-uncased", "org/repo_id"]
