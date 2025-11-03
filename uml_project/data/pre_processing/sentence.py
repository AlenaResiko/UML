"""
Take a string input 'text' and returns a numpy array of pre-processed sentences.

1. Run sentences through a deduplication process to remove duplicate sentences (case-insensitive).
    Ex: Taylor Swift lyrics - avoid repeated chorus

2. Remove sentences that start with a special character (non-alphanumeric).
    Ex: LaTeX commands in scientific
        [Verse 1:]

3. Trim leading and trailing whitespace from each sentence.
"""

# uml_project/data/pre_processing/sentence.py
import re
import numpy as np
from collections.abc import Iterable

import spacy
from spacy.language import Language

IGNORE_SENTENCE_START_CHARS = set("!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~")


# ---------- main API ----------
def define_sentence(
    text: str, nlp: Language | None = None, min_chars: int = 2, dedupe_case_insensitive: bool = True
) -> np.ndarray:
    """
    Split `text` into sentence-like units using spaCy, apply lyric/LaTeX/URL filters,
    and deduplicate. Returns a numpy array of strings.
    """
    if not isinstance(text, str) or not text.strip():
        return np.array([], dtype=object)

    if nlp is None:
        nlp = build_sentencizer(use_small_model=False)

    # Strategy: be line-aware (good for lyrics). For each line:
    #   - If it already looks like a full sentence (has . ! ?), use spaCy to split it.
    #   - Otherwise, keep the line as a single unit after cleaning.
    sentences: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _clean(raw)
        if not line:
            continue
        if _looks_like_garbage(line):
            continue
        # prose-like line → let spaCy split on punctuation
        if _END_PUNCT.search(line):
            for s in nlp(line).sents:
                seg = _clean(str(s))
                if seg and not _looks_like_garbage(seg) and len(seg) >= min_chars:
                    sentences.append(seg)
        else:
            # lyric-like line without end punctuation → take as-is
            if len(line) >= min_chars:
                sentences.append(line)

    sentences = _dedupe(sentences, casefold=dedupe_case_insensitive)
    return np.array(sentences, dtype=object)


# ---------- spaCy setup ----------
def build_sentencizer(use_small_model: bool = False) -> Language:
    """
    Return an English pipeline with only a sentencizer.
    use_small_model=True if you've installed 'en_core_web_sm' and want slightly better tokenization.
    """
    if use_small_model:
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    else:
        nlp = spacy.blank("en")  # no download, fast
    # make sentence boundaries on punctuation; works well for prose
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# ---------- regex filters ----------
_BRACKETED_TAG = re.compile(
    r"^\s*$begin:math:display$[^$end:math:display$]+\]\s*:?\s*$"
)  # [Chorus], [Verse 1], [Intro]
_LATEX_LINE = re.compile(r"^\s*\\[A-Za-z@]+")  # \section, \begin{...}, \newcommand
_URL_LINE = re.compile(r"^\s*(https?://|www\.)", re.I)
_ONLY_PUNCT = re.compile(r"^[\W_]+$")
_MULTI_SPACE = re.compile(r"\s+")
_END_PUNCT = re.compile(r"[.!?]")  # to detect sentencey lines


def _clean(s: str) -> str:
    return _MULTI_SPACE.sub(" ", s.replace("\u200b", "")).strip()


def _looks_like_garbage(s: str) -> bool:
    if not s:
        return True
    if _BRACKETED_TAG.match(s):
        return True
    if _LATEX_LINE.match(s):
        return True
    if _URL_LINE.match(s):
        return True
    if _ONLY_PUNCT.match(s):
        return True
    if not s[0].isalnum() or s[0] in IGNORE_SENTENCE_START_CHARS:
        return True
    # common lyric-site cruft
    low = s.lower()
    if "get tickets" in low or "you might also like" in low or "embed" == low:
        return True
    return False


def _dedupe(strings: Iterable[str], casefold: bool = True) -> list[str]:
    seen = set()
    out = []
    for s in strings:
        k = s.casefold() if casefold else s
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out
