import re
import spacy
import numpy as np
from spacy.language import Language

# ---------- compile once ----------
# remove whole environments that are never prose
_ENV_BLOCK = re.compile(
    r"""
    \\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|array|tabular\*?|figure\*?|table\*?|algorithm\*?)\}
    [\s\S]*?
    \\end\{(?:equation\*?|align\*?|gather\*?|multline\*?|array|tabular\*?|figure\*?|table\*?|algorithm\*?)\}
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

# remove display math: $$…$$ or \[ … \]
_DISPLAY_MATH = re.compile(r"(\$\$.*?\$\$|\\\[.*?\\\])", re.DOTALL)

# strip inline math: $…$ or \( … \)
_INLINE_MATH = re.compile(r"(\$[^$]+\$|\\\([^)]*\\\))")

# strip commands that are never useful in prose
_CMD_DROP = re.compile(
    r"""
    \\(label|ref|eqref|citep?|citet|footnote|url|href|nonumber|hline|cline)\b
    (\[[^\]]*\])?
    (\{[^{}]*\})*
    """,
    re.IGNORECASE | re.VERBOSE,
)


# unwrap commands that *do* contain readable text, e.g. \text{…}, \emph{…}
def _unwrap_text_commands(s: str) -> str:
    # repeatedly unwrap a few safe commands
    for _ in range(3):
        s = re.sub(r"\\(text|emph|underline|mathbf|mathrm|mathit)\{([^{}]*)\}", r"\2", s)
    return s


# table-like lines (alignment ampersands + line breaks)
_TABLEY = re.compile(r"&.*\\\\")  # e.g., rows from tabular
_LATEX_LEAD = re.compile(r"^\s*\\[A-Za-z@]+")

_MULTI_SPACE = re.compile(r"\s+")


def strip_latex_prose(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    # 1) remove big non-prose blocks
    txt = _ENV_BLOCK.sub(" ", text)
    txt = _DISPLAY_MATH.sub(" ", txt)

    # 2) drop obvious table-like lines and pure LaTeX lines
    lines = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _TABLEY.search(line):
            continue
        if _LATEX_LEAD.match(line):  # \section, \newcommand, etc.
            continue
        # strip inline bits and unwrap readable content
        line = _INLINE_MATH.sub(" ", line)
        line = _CMD_DROP.sub(" ", line)
        line = _unwrap_text_commands(line)
        line = _MULTI_SPACE.sub(" ", line).strip(" ,.;")
        if line:
            lines.append(line)
    # join to paragraphs; leave periods to help sentencizer
    return ".\n".join(lines)


# optional: minimal sentencizer (no downloads)
def build_sentencizer() -> Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def latex_to_clean_sentences(raw_tex: str, nlp: Language | None = None) -> np.ndarray:
    """Return prose-only sentences from LaTeX source."""
    if nlp is None:
        nlp = build_sentencizer()
    prose = strip_latex_prose(raw_tex)
    if not prose:
        return np.array([], dtype=object)
    # spaCy split
    sents = [s.text.strip() for s in nlp(prose).sents if s.text.strip()]
    # drop tiny math crumbs that survived
    sents = [s for s in sents if len(s) > 3 and not s.startswith("\\")]
    return np.array(sents, dtype=object)
