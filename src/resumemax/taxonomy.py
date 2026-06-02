"""Canonical skill taxonomy with alias-based fuzzy lookup."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "skills_taxonomy"
DEFAULT_PATH = DATA_DIR / "skills_seed.json"


@lru_cache(maxsize=1)
def load_taxonomy(path: str = str(DEFAULT_PATH)) -> Dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def alias_index(taxonomy: Optional[Dict[str, dict]] = None) -> Dict[str, str]:
    """Map every alias (lowercase) to its canonical skill id."""
    tax = taxonomy or load_taxonomy()
    idx: Dict[str, str] = {}
    for canonical, info in tax.items():
        for alias in info.get("aliases", []) + [canonical]:
            idx[alias.lower()] = canonical
    return idx


def match_skill(text: str, threshold: int = 88) -> Optional[Tuple[str, int]]:
    """Return (canonical, score) if `text` matches any alias above `threshold`."""
    text_l = text.lower().strip()
    idx = alias_index()
    if text_l in idx:
        return idx[text_l], 100
    if fuzz is None:
        return None
    best, best_score = None, 0
    for alias, canonical in idx.items():
        s = fuzz.ratio(text_l, alias)
        if s > best_score:
            best, best_score = canonical, s
    if best and best_score >= threshold:
        return best, best_score
    return None


def extract_skills_by_dict(text: str) -> List[str]:
    """Dictionary based skill extraction. Returns a sorted unique list of canonical ids."""
    import re as _re
    idx = alias_index()
    text_l = text.lower()
    found = set()
    for alias, canonical in idx.items():
        # Word-boundary match that tolerates surrounding punctuation.
        pattern = r"(?<![A-Za-z0-9])" + _re.escape(alias) + r"(?![A-Za-z0-9])"
        if _re.search(pattern, text_l):
            found.add(canonical)
    return sorted(found)
