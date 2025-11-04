import os
from pathlib import Path

# PATHS
ROOT_DIR = Path(os.environ.get("ROOT_DIR") or Path(__file__).parent.parent.parent).resolve()
MODEL_DIR = ROOT_DIR / "models/"
