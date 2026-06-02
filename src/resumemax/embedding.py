"""Sentence-BERT semantic similarity layer.

Provides graceful fallback to a TF-IDF cosine similarity if sentence-transformers
or torch are unavailable, so the pipeline can still run for development.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

import numpy as np


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|[\n\r]+|•|·")


def split_sentences(text: str) -> List[str]:
    parts = [p.strip(" -•\t") for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if len(p) > 2]


@lru_cache(maxsize=2)
def _model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)
    except Exception:
        return None


def encode(texts: Iterable[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    texts = list(texts)
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    m = _model(model_name)
    if m is not None:
        emb = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
    # Fallback: TF-IDF cosine surrogate so pipeline runs without torch.
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(stop_words="english").fit(texts)
    M = v.transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    if a.shape[1] != b.shape[1]:
        # Pad the narrower matrix (only happens with TF-IDF fallback).
        n = max(a.shape[1], b.shape[1])
        def pad(M):
            if M.shape[1] == n:
                return M
            out = np.zeros((M.shape[0], n), dtype=np.float32)
            out[:, : M.shape[1]] = M
            return out
        a, b = pad(a), pad(b)
    return a @ b.T


def semantic_score(resume_text: str, jd_sentences: List[str]) -> dict:
    """Per-JD-requirement max similarity, averaged for an overall score."""
    r_sents = split_sentences(resume_text)
    if not r_sents or not jd_sentences:
        return {"semantic_score": 0.0, "per_requirement": []}
    r_emb = encode(r_sents)
    j_emb = encode(jd_sentences)
    sim = cosine_matrix(j_emb, r_emb)
    per_req = sim.max(axis=1) if sim.size else np.zeros(len(jd_sentences))
    best_idx = sim.argmax(axis=1) if sim.size else np.zeros(len(jd_sentences), dtype=int)
    return {
        "semantic_score": float(per_req.mean()),
        "per_requirement": [
            {
                "jd_sentence": jd_sentences[i],
                "best_resume_sentence": r_sents[int(best_idx[i])] if r_sents else "",
                "similarity": float(per_req[i]),
            }
            for i in range(len(jd_sentences))
        ],
    }
