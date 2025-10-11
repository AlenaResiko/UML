import os
from pathlib import Path


# PATHS
ROOT = Path(os.environ.get("DATA_DIR") or Path(__file__).parent.parent).resolve()
MUSIC_DIR = ROOT / "music/"


# KAGGLE
TAYLOR_SWIFT_KAGGLE = "ishikajohari/taylor-swift-all-lyrics-30-albums"


# GENIUS
GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN")  # Store in .env file
