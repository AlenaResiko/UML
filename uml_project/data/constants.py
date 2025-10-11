import os
from pathlib import Path
import typing as t

# PATHS
ROOT_DIR = Path(os.environ.get("DATA_DIR") or Path(__file__).parent.parent.parent).resolve() / "data"
MUSIC_DIR = ROOT_DIR / "music/"
SCRATCH_DIR = ROOT_DIR / "scratch/"  # for temporary files
SCIENTIFIC_DIR = ROOT_DIR / "scientific/"

# KAGGLE
TAYLOR_SWIFT_KAGGLE = "ishikajohari/taylor-swift-all-lyrics-30-albums"


# GENIUS
GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN")  # Store in .env file


# Dtypes
class TexSource(t.TypedDict):
    id: str | None  # e.g. "2503.19280v1"
    files: list[dict[str, str]]  # {"name": str, "text": str}
    main: str | None  # main .tex file name
    title: str | None  # extracted title, if any
