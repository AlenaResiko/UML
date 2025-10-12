def _extract_title_from_tex(tex: str) -> str | None:
    """
    Extracts the LaTeX title from a TeX document string.
    - Strips comments (ignoring escaped percent).
    - Finds \title[optional]{...} (captures multiline).
    - Cleans result: removes \thanks{...}, \footnote{...}, collapses \\ and ~ to spaces, collapses whitespace.
    - Returns cleaned title or None if not found.
    """
    import re

    # Remove comments (ignore escaped %)
    tex_nocomment = re.sub(r"(?<!\\)%.*", "", tex)
    # Match \title[optional]{...}
    m = re.search(r"\\title(?:\[[^\]]*\])?\{(.*?)\}", tex_nocomment, flags=re.S)
    if not m:
        return None
    title = m.group(1)
    if not title:
        return None
    # Remove \thanks{...}, \footnote{...}, etc.
    title = re.sub(r"\\(thanks|footnote)\{.*?\}", "", title, flags=re.S)
    # Replace \\ and ~ with spaces
    title = re.sub(r"\\\\|~", " ", title)
    # Collapse all whitespace to single space
    title = re.sub(r"\s+", " ", title)
    # Strip
    title = title.strip()
    if not title:
        return None
    return title


import html
import io
import re
import tarfile

import requests

# ------------------------
# TeX source fetching (arXiv)
# ------------------------

ARXIV_ABS_RE = re.compile(r"https?://arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?")
ARXIV_PDF_RE2 = re.compile(r"https?://arxiv\.org/pdf/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?(?:\.pdf)?")
ARXIV_SRC_RE = re.compile(r"https?://arxiv\.org/src/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?")
ARXIV_PDF_RE = re.compile(r"https?://arxiv\.org/pdf/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?")


def _slugify_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "paper"


def _clean_arxiv_title(title: str) -> str:
    # Unescape HTML entities
    title = html.unescape(title)

    # Remove any remaining XML/HTML tags
    title = re.sub(r"<.*?>", "", title)

    # Remove LaTeX commands wrapping text: \emph{...}, \textbf{...}, \texorpdfstring{A}{B}
    # For \texorpdfstring, keep first brace content
    def replace_latex_cmd(m):
        cmd = m.group(1)
        content = m.group(2)
        if cmd == "texorpdfstring":
            # Extract first brace content only
            first_brace = re.match(r"\{(.*?)\}", content)
            if first_brace:
                return first_brace.group(1)
            return ""
        return content

    # Match commands with one or two brace groups, keep first content
    # This regex matches commands like \emph{...}, \textbf{...}, \texorpdfstring{...}{...}
    title = re.sub(
        r"\\(emph|textbf|textit|textrm|texttt|texorpdfstring)\{([^{}]*)(?:\}\{[^{}]*\})?\}", replace_latex_cmd, title
    )

    # Remove inline math $...$ and display math \( ... \), \[ ... \]
    title = re.sub(r"\$(?:[^$]|\\\$)+\$", "", title)
    title = re.sub(r"\\\((?:[^\\)]|\\.)*\\\)", "", title)
    title = re.sub(r"\\\[(?:[^\]\\]|\\.)*\\\]", "", title)

    # Collapse all whitespace (including newlines and tabs) to single spaces
    title = re.sub(r"\s+", " ", title)

    # Trim spaces around punctuation (e.g., " , " -> ", ")
    title = re.sub(r"\s*([,;:.!?])\s*", r"\1 ", title)

    # Strip leading/trailing whitespace again
    title = title.strip()

    return title


def _fetch_arxiv_metadata_safe(arxiv_id: str, timeout: int = 30) -> dict | None:
    """
    Lightweight arXiv metadata fetcher that returns:
      {"title": str, "authors": List[str], "year": int | None, "published": str | None}
    Returns None on failure.
    """
    try:
        api = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        r = requests.get(api, timeout=timeout)
        r.raise_for_status()
        xml = r.text
        # extract title (2nd <title>), published (full date), authors, and year
        titles = re.findall(r"<title>(.*?)</title>", xml, flags=re.S)
        title = titles[1].strip() if len(titles) >= 2 else None
        if title:
            title = _clean_arxiv_title(title)
        else:
            title = None
        authors = [re.sub(r"<.*?>", "", a).strip() for a in re.findall(r"<name>(.*?)</name>", xml)]
        published = None
        m = re.search(r"<published>(\d{4}-\d{2}-\d{2})", xml)
        if m:
            published = m.group(1)
        year = None
        my = re.search(r"<published>(\d{4})-", xml)
        if my:
            year = int(my.group(1))
        return {"title": title, "authors": authors, "year": year, "published": published}
    except Exception:
        return None


def _resolve_arxiv_id_and_version(url: str, timeout: int = 30) -> tuple[str, str | None] | None:
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


def _download_arxiv_source(arxiv_id: str, version: str | None, timeout: int = 60) -> bytes:
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


def _extract_tex_from_tar(tar_bytes: bytes) -> tuple[list[tuple[str, bytes]], list[str]]:
    tex_members: list[tuple[str, bytes]] = []
    all_names: list[str] = []
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


def _guess_main_tex(tex_names: list[str]) -> str | None:
    if not tex_names:
        return None

    # Score by keyword presence and path depth (shallower is better)
    def score(name: str) -> tuple[int, int, int]:
        base = name.split("/")[-1].lower()
        keyscore = max((base.startswith(k) or base.endswith(k + ".tex")) for k in _DEF_MAIN_KEYS)
        # prefer shortest basename
        return (int(not keyscore), name.count("/"), len(base))

    return sorted(tex_names, key=score)[0]
