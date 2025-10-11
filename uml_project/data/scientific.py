from __future__ import annotations
import re
from .constants import *
import json
from pathlib import Path
from uml_project.data.utils.latex_helpers import (
    _resolve_arxiv_id_and_version,
    _download_arxiv_source,
    _extract_tex_from_tar,
    _guess_main_tex,
    _fetch_arxiv_metadata_safe,
    _slugify_title,
    _extract_title_from_tex,
)


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
