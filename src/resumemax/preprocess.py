"""Text preprocessing utilities: tokenization, lemmatization, normalization."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#\-]*")


@lru_cache(maxsize=1)
def _nlp():
    """Lazily load spaCy model. Falls back to a blank pipeline if not installed."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_lg")
        except OSError:
            return spacy.load("en_core_web_sm")
    except Exception:
        return None


def simple_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def normalize_token(tok: str) -> str:
    """Lowercase, but preserve all-uppercase acronyms (AWS, SQL, ML)."""
    if tok.isupper() and len(tok) <= 5:
        return tok
    return tok.lower()


def lemmatize(text: str) -> List[str]:
    nlp = _nlp()
    if nlp is None:
        return [normalize_token(t) for t in simple_tokenize(text)]
    doc = nlp(text)
    out: List[str] = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if tok.text.isupper() and len(tok.text) <= 5:
            out.append(tok.text)
        else:
            out.append(tok.lemma_.lower())
    return out


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()
