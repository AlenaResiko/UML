from __future__ import annotations
from typing import Dict, Optional, List, Tuple
import re
import requests
import io
import tarfile
from .constants import *

# Existing code above ...

# ------------------------
# TeX source fetching (arXiv)
# ------------------------

ARXIV_ABS_RE = re.compile(r"https?://arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?")
ARXIV_PDF_RE2 = re.compile(r"https?://arxiv\.org/pdf/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?(?:\.pdf)?")
ARXIV_SRC_RE = re.compile(r"https?://arxiv\.org/src/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?")
ARXIV_PDF_RE = re.compile(r"https?://arxiv\.org/pdf/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?")


def fetch_tex_sources(url: str, timeout: int = 60) -> Dict[str, object]:
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
    files = []
    for name, raw in tex_members:
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        files.append({"name": name, "text": text})
    main = _guess_main_tex([n for n, _ in tex_members])
    return {"id": f"{arx_id}{ver or ''}", "files": files, "main": main}


def _resolve_arxiv_id_and_version(url: str, timeout: int = 30) -> Optional[Tuple[str, Optional[str]]]:
    # Direct matches
    for rx in (ARXIV_ABS_RE, ARXIV_PDF_RE2, ARXIV_SRC_RE, ARXIV_PDF_RE):
        m = rx.match(url)
        if m:
            return m.group("id"), (m.group("ver") or None)
    # If it's an abs page with fragments, fetch HTML and regex the src link
    if "arxiv.org/abs/" in url:
        try:
            html = requests.get(url.split("#")[0], timeout=timeout).text
            m = re.search(r'href="/src/(\d{4}\.\d{4,5})(v\d+)?\?download=1"', html)
            if m:
                return m.group(1), (m.group(2) or None)
        except Exception:
            pass
    return None


def _download_arxiv_source(arxiv_id: str, version: Optional[str], timeout: int = 60) -> bytes:
    ver = version or ""
    # Preferred modern endpoint
    url1 = f"https://arxiv.org/src/{arxiv_id}{ver}?download=1"
    r = requests.get(url1, timeout=timeout, allow_redirects=True)
    if r.status_code == 200 and r.content:
        return r.content
    # Fallback legacy endpoint
    url2 = f"https://arxiv.org/e-print/{arxiv_id}{ver}"
    r2 = requests.get(url2, timeout=timeout, allow_redirects=True)
    r2.raise_for_status()
    return r2.content


def _extract_tex_from_tar(tar_bytes: bytes) -> Tuple[List[Tuple[str, bytes]], List[str]]:
    tex_members: List[Tuple[str, bytes]] = []
    all_names: List[str] = []
    bio = io.BytesIO(tar_bytes)
    # Some arXiv src are gzipped tar; tarfile can auto-detect via mode="r:*"
    with tarfile.open(fileobj=bio, mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name
            all_names.append(name)
            if name.lower().endswith(".tex"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                tex_members.append((name, f.read()))
    return tex_members, all_names


_DEF_MAIN_KEYS = ("main", "paper", "ms", "manuscript", "arxiv")


def _guess_main_tex(tex_names: List[str]) -> Optional[str]:
    if not tex_names:
        return None

    # Score by keyword presence and path depth (shallower is better)
    def score(name: str) -> Tuple[int, int, int]:
        base = name.split("/")[-1].lower()
        keyscore = max((base.startswith(k) or base.endswith(k + ".tex")) for k in _DEF_MAIN_KEYS)
        # prefer shortest basename
        return (int(not keyscore), name.count("/"), len(base))

    return sorted(tex_names, key=score)[0]


# ------------------------ # Utility: Save TeX file from raw text # ------------------------
import re


def save_tex_file(raw_text: str, base_name: str = "paper", strict_unescape: bool = False):
    """
    Save a raw LaTeX text string as a .tex file in the scratch directory,
    ensuring that literal '\\n' characters are treated as newlines **without**
    corrupting LaTeX control sequences like ``\newcommand``/``\newtheorem``.

    Parameters
    ----------
    raw_text : str
        The LaTeX text content, possibly containing literal '\\n' sequences.
    base_name : str
        The base name for the file (without extension). Defaults to 'paper'.
    strict_unescape : bool
        If True, aggressively replace *all* literal "\\n" with real newlines.
        Default False (safe mode): only unescape when "\\n" is **not** starting
        a LaTeX control word (i.e., not followed by a letter).

    Returns
    -------
    str
        The full path to the saved .tex file.

    Notes
    -----
    - The naive replacement of all "\\n" → "\\n" breaks LaTeX commands like
      \newcommand or \newtheorem, which must remain as literal backslash + n.
    - This function only replaces "\\n" when it is *not* followed by a letter,
      leaving LaTeX commands untouched. See comments below for details.
    - If strict_unescape=True, all "\\n" are replaced, for trusted JSON input.
    """
    import os
    from datetime import datetime

    scratch_dir = SCRATCH_DIR
    os.makedirs(scratch_dir, exist_ok=True)

    # Conservatively convert escaped newlines.
    # The naive replacement of all "\\n" -> "\n" breaks LaTeX commands like \newcommand / \newtheorem
    # (which literally begin with the two characters backslash + 'n').
    #
    # Safe rule: only convert "\\n" when it is **not** followed by a letter.
    # This handles JSON-style line breaks (often followed by '\\', digits, space, etc.)
    # while preserving control words like "\\newcommand".
    #
    # Do not touch "\\\n" (double backslash + n) which is a LaTeX linebreak.
    if "\\n" in raw_text:
        if strict_unescape:
            # Strict: replace all (legacy)
            text_content = raw_text.replace("\\r\\n", "\n").replace("\\n", "\n")
        else:
            text_content = raw_text
            # Normalize escaped CRLF first
            text_content = text_content.replace("\\r\\n", "\n")
            # Replace \n not followed by a letter (negative lookahead)
            # This preserves e.g. \newcommand, \noindent, \newtheorem, etc.
            text_content = re.sub(r"\\n(?![A-Za-z])", "\n", text_content)
    else:
        text_content = raw_text

    # Create timestamped filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.tex"
    file_path = os.path.join(scratch_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    return text_content, file_path
