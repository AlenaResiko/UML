import html
import json
import re
from pathlib import Path

import requests

from uml_project.data.constants import SCIENTIFIC_REGISTRY, ArxivSearchResultDict, RegistryDict


def append_arxiv_results_to_registry(
    results: list[ArxivSearchResultDict], registry_path: str | Path = SCIENTIFIC_REGISTRY
) -> None:
    """
    Append arXiv search results to REGISTRY.json under the 'arxiv' key.

    Parameters
    ----------
    results : list[dict]
        List of search results from `search_arxiv()`. Each must include an 'id' and 'url_abs'.
    registry_path : str or Path
        Path to REGISTRY.json (defaults to SCIENTIFIC_REGISTRY).

    Behavior
    --------
    - Loads existing REGISTRY.json (creates it if missing).
    - Appends any new arXiv URLs (from `url_abs`) that aren't already present.
    - Only adds entries where 'url_abs' starts with 'https://arxiv.org/'.
    - Saves the updated JSON back to disk.
    """
    reg_path = Path(registry_path)
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing registry if it exists, else initialize
    if reg_path.exists():
        with open(reg_path, encoding="utf-8") as f:
            try:
                registry: RegistryDict = json.load(f)
            except json.JSONDecodeError:
                registry: RegistryDict = {"arxiv": []}
    else:
        registry: RegistryDict = {"arxiv": []}

    # # Ensure proper structure
    # if "arxiv" not in registry or not isinstance(registry["arxiv"], list):
    #     registry["arxiv"] = []

    existing = set(registry["arxiv"])

    new_links = []
    for item in results:
        url = item.get("url_abs")
        if url and re.match(r"^https?://arxiv\.org/abs/", url) and url not in existing:
            new_links.append(url)
            existing.add(url)

    if new_links:
        registry["arxiv"].extend(new_links)
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        print(f"✅ Added {len(new_links)} new arXiv links to registry.")
    else:
        print("No new arXiv links to add — registry already up to date.")


def search_arxiv(
    query: str | None = None,
    author: str | None = None,
    title: str | None = None,
    abstract: str | None = None,
    categories: list[str] | None = None,  # e.g. ["cs.LG","stat.ML"]
    start: int = 0,
    max_results: int = 50,  # arXiv caps at 30000 total; typical page 50–200
    sort_by: str = "submittedDate",  # "relevance", "lastUpdatedDate", "submittedDate"
    sort_order: str = "descending",  # "ascending" or "descending"
    timeout: int = 30,
):
    """
    Search arXiv by author/keyword and return a list of matches with abs/pdf links.

    Returns: List[dict] with keys: id, title, authors, year, published, url_abs, url_pdf, summary, categories
    """
    # Build the search_query param
    parts = []
    if query:
        parts.append(f'all:"{query}"')
    if author:
        parts.append(f'au:"{author}"')
    if title:
        parts.append(f'ti:"{title}"')
    if abstract:
        parts.append(f'abs:"{abstract}"')
    if categories:
        parts.append(" OR ".join(f"cat:{c}" for c in categories))

    if not parts:
        raise ValueError("Provide at least one of: query, author, title, abstract, or categories")

    search_query = " AND ".join(f"({p})" for p in parts)

    base = "http://export.arxiv.org/api/query"
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    # Query
    r = requests.get(base, params=params, timeout=timeout, headers={"User-Agent": "arXiv-Searcher/1.0"})
    r.raise_for_status()
    xml = r.text

    # Parse entries with light regex (enough for the stable Atom structure)
    entries = re.split(r"<entry>", xml)[1:]  # first split is feed header
    results: list[ArxivSearchResultDict] = []
    for e in entries:
        # id (abs URL) & arxiv id
        m_id = re.search(r"<id>(.*?)</id>", e, flags=re.S)
        url_abs = html.unescape(m_id.group(1).strip()) if m_id else None
        arxiv_id = url_abs.rsplit("/", 1)[-1] if url_abs else None

        # title
        m_title = re.search(r"<title>(.*?)</title>", e, flags=re.S)
        title_txt = html.unescape(re.sub(r"\s+", " ", (m_title.group(1) if m_title else "")).strip())

        # authors
        authors = [html.unescape(a.strip()) for a in re.findall(r"<name>(.*?)</name>", e)]

        # published (ISO), year
        m_pub = re.search(r"<published>(\d{4}-\d{2}-\d{2})", e)
        published = m_pub.group(1) if m_pub else None
        year = int(published[:4]) if published else None

        # summary (abstract)
        m_sum = re.search(r"<summary>(.*?)</summary>", e, flags=re.S)
        summary = html.unescape(re.sub(r"\s+", " ", (m_sum.group(1) if m_sum else "")).strip())

        # categories
        cats = re.findall(r'term="([^"]+)"', e)  # from <category term="cs.LG" scheme="..."/>
        # pdf link
        m_pdf = re.search(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', e)
        url_pdf = (
            html.unescape(m_pdf.group(1)) if m_pdf else (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None)
        )

        results.append(
            {
                "id": arxiv_id,
                "title": title_txt,
                "authors": authors,
                "year": year,
                "published": published,
                "summary": summary,
                "categories": cats,
                "url_abs": url_abs,
                "url_pdf": url_pdf,
            }
        )

    return results
