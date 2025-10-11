import os
from pathlib import Path
import typing as t

# PATHS
ROOT_DIR = Path(os.environ.get("DATA_DIR") or Path(__file__).parent.parent).resolve()
MUSIC_DIR = ROOT_DIR / "music/"
SCRATCH_DIR = ROOT_DIR / "scratch/"  # for temporary files

# KAGGLE
TAYLOR_SWIFT_KAGGLE = "ishikajohari/taylor-swift-all-lyrics-30-albums"


# GENIUS
GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN")  # Store in .env file


# Dtypes
class TexSource(t.TypedDict):
    id: t.Optional[str]  # e.g. "2503.19280v1"
    files: list[dict[str, str]]  # {"name": str, "text": str}
    main: t.Optional[str]  # main .tex file name
