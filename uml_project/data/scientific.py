from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from uml_project.data.utils.latex_helpers import (
    _download_arxiv_source,
    _extract_tex_from_tar,
    _extract_title_from_tex,
    _fetch_arxiv_metadata_safe,
    _guess_main_tex,
    _resolve_arxiv_id_and_version,
    _slugify_title,
)

from .constants import *


def batch_fetch_tex_sources_from_json(json_file: str | Path | None = None, max_workers: int = 8, timeout: int = 60):
    """
    Read a JSON file listing paper URLs and fetch TeX sources in parallel.

    JSON formats supported:
      - List of URLs:
            ["https://arxiv.org/abs/2503.19280", ...]
      - Object with key 'urls':
            {"urls": ["https://arxiv.org/abs/2503.19280", ...]}
      - Mapping of name->URL:
            {"logic_learner": "https://arxiv.org/abs/2503.19280"}

    Parameters
    ----------
    json_file : str or Path
        Path to the JSON file.
    max_workers : int
        Number of parallel threads (default: 8).
    timeout : int
        Timeout in seconds for each individual fetch.

    Returns
    -------
    dict[str, dict]
        Mapping from ID (or key) to the TexSource dict returned by fetch_tex_sources.
        If a fetch fails, returns {"error": "ExceptionName: message"} for that entry.
    """
    json_path = Path(json_file or SCIENTIFIC_REGISTRY)
    with open(json_path, encoding="utf-8") as fp:
        payload = json.load(fp)

    # Normalize into list of (key, url)
    pairs = []
    if isinstance(payload, list):
        pairs = [(url, url) for url in payload]
    elif isinstance(payload, dict):
        if "urls" in payload and isinstance(payload["urls"], list):
            pairs = [(url, url) for url in payload["urls"]]
        else:
            pairs = [(k, v) for k, v in payload.items()]
    else:
        raise ValueError("Unsupported JSON structure; expected list or dict of URLs.")

    results: dict[str, TexSource] = {}

    def _worker(name, url):
        try:
            src = fetch_tex_sources(url, timeout=timeout)
            key = src.get("id") or name or url
            return key, src
        except Exception as e:
            return name or url, TexSource({"id": f"{type(e).__name__}: {e}", "files": [], "main": None, "title": None})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, name, url) for name, url in pairs]
        for fut in as_completed(futures):
            key, value = fut.result()
            results[str(key)] = value

    return results


def fetch_tex_sources(url: str, timeout: int = 60):
    """Fetch raw .tex files from a paper URL (currently supports arXiv).

    Returns a dict: {"id": Optional[str], "files": [{"name": str, "text": str}, ...], "main": Optional[str]}
    The `.text` values are returned **as-is** from the archive (no LaTeX cleaning).
    """
    arxiv = _resolve_arxiv_id_and_version(url, timeout=timeout)
    if not arxiv:
        raise ValueError("Only arXiv URLs are supported at the moment.")
    arx_id, ver = arxiv
    tar_bytes = _download_arxiv_source(arx_id, ver, timeout=timeout)
    tex_members, all_members = _extract_tex_from_tar(tar_bytes)
    main = _guess_main_tex([n for n, _ in tex_members])
    # Try to extract title from main tex, fallback to arXiv metadata if needed
    raw_main_text = None
    tex_title = None
    if main:
        for n, raw in tex_members:
            if n == main:
                try:
                    raw_main_text = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    raw_main_text = raw.decode("latin-1", errors="strict")
                break
        if raw_main_text is not None:
            tex_title = _extract_title_from_tex(raw_main_text)
    meta_title = None
    if not tex_title:
        # fallback to arXiv metadata
        meta = _fetch_arxiv_metadata_safe(arx_id)
        if meta:
            meta_title = meta.get("title")
    # Clean up files as before
    files = []
    for name, raw in tex_members:
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        text = text.replace("%", "")
        text = text.replace("=", "")
        # Collapse all consecutive spaces (2 or more) into exactly 3 spaces
        text = re.sub(r" {2,}", "   ", text)
        files.append({"name": name, "text": text})
    # Title selection: prefer tex_title, then meta_title, else None
    title = tex_title or meta_title or None
    return TexSource(id=f"{arx_id}{ver or ''}", files=files, main=main, title=title)


def batch_save_texsource_jsons(
    texsrc_list: dict[str, TexSource], out_dir: Path | None = None, timeout: int = 30
) -> dict[str, list[str]]:
    """
    Save multiple TexSource dicts to JSON files using `save_texsource_jsons`.

    Parameters
    ----------
    texsrc_list : dict[str, TexSource]
        Mapping from ID (or key) to TexSource dict.
    out_dir : str or Path, optional
        Directory to save the JSON files (default: SCIENTIFIC_DIR).
    timeout : int
        Timeout in seconds for each individual save operation.

    Returns
    -------
    dict[str, list[str]]
        Mapping from ID (or key) to list of saved file paths.
    """
    results: dict[str, list[str]] = {}

    def _worker(key, texsrc):
        try:
            paths = save_texsource_jsons(texsrc, out_dir=out_dir, timeout=timeout)
            return key, paths
        except Exception as e:
            return key, [f"Error: {type(e).__name__}: {e}"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker, key, texsrc) for key, texsrc in texsrc_list.items()]
        for fut in as_completed(futures):
            key, value = fut.result()
            results[str(key)] = value

    return results


def save_texsource_jsons(texsrc: TexSource, out_dir: Path | None = None, timeout: int = 30) -> list[str]:
    """
    Save one JSON per TeX document in `texsrc["files"]` to SCIENTIC_DIR by default.

    Each JSON contains:
      1) title (str)
      2) authors (List[str])
      3) date_published (str | None)  # ISO date if available, else year, else None
      4) raw_tex (str)  # single string with the file's LaTeX content

    Filenames are derived from the paper title: lowercase, spaces -> '_'.
    If multiple .tex files exist, a suffix based on the teX basename is appended to avoid overwrites.
    Returns the list of saved file paths.
    """
    # Resolve output dir
    outdir = out_dir if out_dir is not None else SCIENTIFIC_DIR
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Try to fetch metadata from arXiv if an id is present
    # Title preference: texsrc.get("title"), then arXiv metadata, then main tex stem
    title = texsrc.get("title")
    authors: list[str] = []
    date_published: str | None = None

    arx_id_with_ver = texsrc.get("id")
    meta = None
    if arx_id_with_ver:
        m = re.search(r"(\d{4}\.\d{4,5})", arx_id_with_ver)
        if m:
            base_id = m.group(1)
            meta = _fetch_arxiv_metadata_safe(base_id, timeout=timeout)
            if meta:
                if not title:
                    title = meta.get("title") or title
                authors = meta.get("authors") or authors
                # Prefer full published date if available; fallback to year
                if meta.get("published"):
                    date_published = meta["published"]
                elif meta.get("year"):
                    date_published = str(meta["year"])

    # Reasonable fallbacks if metadata missing
    if not title:
        # Use main tex name (without extension) or the arXiv id
        main = texsrc.get("main") or (texsrc["files"][0]["name"] if texsrc.get("files") else "paper")
        title = Path(main).stem
    if not authors:
        authors = []
    if date_published is None:
        date_published = None

    # Build slug from title
    slug = _slugify_title(title)

    saved_paths: list[str] = []
    for f in texsrc.get("files", []):
        raw_tex = f.get("text", "")
        # Build filename; if multiple files, differentiate by basename
        basename = Path(f.get("name", "doc.tex")).stem
        fname = f"{slug}.json"
        if len(texsrc.get("files", [])) > 1:
            # Avoid overwriting by appending file stem
            fname = f"{slug}__{_slugify_title(basename)}.json"
        outpath = str(Path(outdir) / fname)

        payload = {
            "title": title,
            "authors": authors,
            "date_published": date_published,
            "raw_tex": raw_tex,
        }
        with open(outpath, "w", encoding="utf-8") as fp:
            print(f"Saving to {outpath}")
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        saved_paths.append(outpath)

    return saved_paths
