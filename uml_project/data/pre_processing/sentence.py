"""
Take a string input 'text' and return a numpy array of pre-processed sentences.

1. Deduplicate only **adjacent** repeated sentences (case-insensitive).
   Ex: Taylor Swift lyrics – remove repeated chorus lines that appear back-to-back,
   but keep the same line if it reappears later in the song.

2. Remove sentences that start with a special character (non-alphanumeric) or match
   common site/markup noise (e.g., [Chorus], LaTeX commands, URLs, header junk).

3. Trim leading and trailing whitespace from each sentence.
"""

import re
import numpy as np
from typing import Literal, Iterable
import spacy
from spacy.language import Language

IGNORE_SENTENCE_START_CHARS = set("!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~")
from uml_project.data.pre_processing.latex_helper import strip_latex_prose, latex_to_clean_sentences


# ---------- main API ----------
def define_sentence(
    text: str,
    nlp: Language | None = None,
    min_chars: int = 2,
    dedupe_case_insensitive: bool = True,
    *,
    latex_mode: Literal["strip", "sentences"] | None = None,
) -> np.ndarray:
    """
    Split `text` into sentence-like units using spaCy, apply lyric/LaTeX/URL filters,
    and deduplicate. Returns a numpy array of strings.

    Args
    ----
    latex_mode:
        None (default): no special handling; run the existing line-aware logic.
        "strip":       pre-clean LaTeX into prose with `strip_latex_prose`, then run the
                       normal sentence splitting/filters/dedup from this module.
        "sentences":   bypass this module's splitter and directly use
                       `latex_to_clean_sentences` (which returns spaCy-split sentences)
                       and then only do adjacent dedupe here.
    """
    if not isinstance(text, str) or not text.strip():
        return np.array([], dtype=object)

    nlp = build_sentencizer(use_better_model=True) if nlp is None else nlp

    # --- LaTeX specialized paths ---
    if latex_mode == "sentences":
        # Use the LaTeX pipeline end-to-end for sentence extraction
        sents = latex_to_clean_sentences(text, nlp=nlp).tolist()
        sents = _dedupe_consecutive(sents, casefold=dedupe_case_insensitive)
        return np.array(sents, dtype=object)

    if latex_mode == "strip":
        # Pre-clean LaTeX into readable prose, then fall through to normal flow
        text = strip_latex_prose(text)

    # --- Normal flow (lyrics/prose, line-aware) ---
    sentences: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _clean(raw)
        if not line:
            continue
        if _looks_like_garbage(line):
            continue
        if _END_PUNCT.search(line):
            for s in nlp(line).sents:
                seg = _clean(str(s))
                if seg and not _looks_like_garbage(seg) and len(seg) >= min_chars:
                    sentences.append(seg)
        else:
            if len(line) >= min_chars:
                sentences.append(line)

    sentences = _dedupe_consecutive(sentences, casefold=dedupe_case_insensitive)
    return np.array(sentences, dtype=object)


# ---------- spaCy setup ----------
def build_sentencizer(use_better_model: bool = False) -> Language:
    """
    Return an English pipeline with only a sentencizer.
    use_better_model=True if you've installed 'en_core_web_sm' and want slightly better tokenization.
    """
    if use_better_model:
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    else:
        nlp = spacy.blank("en")  # no download, fast
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# ---------- regex filters ----------
# [Chorus], [Verse 1], [Intro]
_BRACKETED_TAG = re.compile(r"^\s*\[[^\]]+\]\s*:?\s*$")
# Genius/lyrics header like: "123 Contributors..."
_HEADER_JUNK = re.compile(r"^\s*\d+\s+Contributors", re.I)
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
    if _HEADER_JUNK.match(s):
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


def _dedupe_consecutive(strings: Iterable[str], casefold: bool = True) -> list[str]:
    """
    Only remove **consecutive** duplicates (case-insensitive if specified).
    Keeps repeated lines that appear later in the text.
    """
    out: list[str] = []
    prev_key: str | None = None
    for s in strings:
        k = s.casefold() if casefold else s
        if prev_key is not None and k == prev_key:
            continue  # skip only immediately repeated lines
        out.append(s)
        prev_key = k
    return out
