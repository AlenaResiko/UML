import os
from pathlib import Path

# PATHS
ROOT_DIR = Path(os.environ.get("DATA_DIR") or Path(__file__).parent.parent.parent).resolve() / "data"
MODEL_DIR = ROOT_DIR / "models/"
