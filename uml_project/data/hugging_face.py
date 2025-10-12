import json
import shutil
import typing as t
from pathlib import Path

from datasets import DatasetDict, load_dataset

from .constants import HF_DIR, HF_REGISTRY, HuggingFaceRegistry

# -----------------------------
# HuggingFace registry utilities
# -----------------------------


def save_all_hf_datasets_from_registry(
    registry_path: Path | str = HF_REGISTRY,
    dest_root: Path | str = HF_DIR,
    *,
    fmt: t.Literal["arrow", "parquet", "json", "csv"] = "parquet",
    compression: str | None = None,
    overwrite: bool = False,
    token: str | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Load every dataset listed under the "datasets" key of REGISTRY.json
    and materialize each split to disk under `dest_root / subdir / <repo_id>/`.

    Returns a mapping:
        { repo_id: { split_name: path_to_saved_artifact } }

    Parameters
    ----------
    registry_path : Path | str
        Path to REGISTRY.json (defaults to HF_REGISTRY).
    dest_root : Path | str
        Root folder where artifacts are saved (defaults to HF_DIR).
    fmt : str
        One of {"arrow","parquet","json","csv"}.
    compression : str | None
        Optional compression for Parquet/JSON/CSV (e.g., "snappy", "zstd", "gzip").
    overwrite : bool
        If True, remove any existing output and re-write.
    token : str | None
        Optional HF token for gated/private datasets.
    """
    out_index: dict[str, dict[str, Path]] = {}
    reg = _read_hf_registry(registry_path)
    repo_ids = reg.get("datasets", [])

    base = Path(dest_root)
    base.mkdir(parents=True, exist_ok=True)

    for repo_id in repo_ids:
        ds = load_dataset(repo_id, token=token)
        repo_map: dict[str, Path] = {}
        if not isinstance(ds, DatasetDict):
            raise TypeError(f"This function only supports DatasetDict, got {type(ds)} for {repo_id}")

        repo_folder = base / repo_id
        repo_folder.mkdir(parents=True, exist_ok=True)

        for split, subset in ds.items():
            split = str(split)
            out_path = repo_folder / f"{split}.{fmt}"
            if out_path.exists() and overwrite:
                if out_path.is_dir():
                    shutil.rmtree(out_path)
                else:
                    out_path.unlink()

            if fmt == "arrow":
                subset.save_to_disk(out_path)
            elif fmt == "parquet":
                subset.to_parquet(out_path, compression=compression)
            elif fmt == "json":
                subset.to_json(out_path, compression=compression)
            elif fmt == "csv":
                subset.to_csv(out_path, compression=compression)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            repo_map[split] = out_path

        out_index[repo_id] = repo_map

    return out_index


def _read_hf_registry(path: Path | str = HF_REGISTRY) -> HuggingFaceRegistry:
    """
    Read the HuggingFace REGISTRY.json file.

    Expected schema:
    {
      "datasets": ["repo_id_1", "org/repo_id_2", ...]
    }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Registry file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data: HuggingFaceRegistry = json.load(f)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError("Registry file must be a dict with a 'datasets' key.")
    data["datasets"] = [str(x) for x in data["datasets"] if x]
    return data
