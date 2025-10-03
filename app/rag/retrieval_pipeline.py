# app/rag/retrieval_pipeline.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import os
import re
import logging

import numpy as np

from app.filters.lexicon import DOMAIN_KEYWORDS
from app.rag.vector_store import search as _vec_search, search_batch as _vec_search_batch

logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("TOP_K", 4))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 4))
CE_MODEL = os.getenv("CROSS_ENCODER_MODEL", "")  # напр. "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BATCH = int(os.getenv("CROSS_ENCODER_BATCH", 32))
CE_WEIGHT = float(os.getenv("CROSS_ENCODER_WEIGHT", 0.55))  # вклад CE в итоговый скор
FAISS_WEIGHT = 1.0 - CE_WEIGHT

_CE = None
def _get_cross_encoder():
    global _CE
    if _CE is None and CE_MODEL:
        try:
            from sentence_transformers import CrossEncoder
            _CE = CrossEncoder(CE_MODEL)
            logger.info("CrossEncoder loaded: %s", CE_MODEL)
        except Exception:
            logger.exception("Failed to load CrossEncoder: %s", CE_MODEL)
            _CE = None
    return _CE


def _normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    q = q.replace("как найти папку со скриншотами", "где папка со скриншотами")
    q = q.replace("как поставить apk", "установка apk файл")
    return q

def _expand_query(base: str) -> List[str]:
    base_l = base.lower()
    variants = {base}

    if any(w in base_l for w in ("папк", "путь", "скрин", "screenshots", "oculus")):
        variants.add(base_l.replace("где", "путь к").strip())
        variants.add(base_l + " oculus/screenshots")
        variants.add(base_l + " android/data")

    for kw in DOMAIN_KEYWORDS:
        if kw in base_l:
            variants.add(base_l.replace(kw, kw + " инструкция"))
            variants.add(base_l.replace(kw, kw + " путь"))

    out = []
    for v in variants:
        v2 = re.sub(r"\s+", " ", v).strip()
        if v2 and v2 not in out:
            out.append(v2)
    # не раздуваем — 1 базовый + 2–3 расширения более чем достаточно
    return out[:3]

def _lex_overlap(q: str, text: str) -> float:
    tok = re.findall(r'\b[а-яёa-z0-9]+(?:[-/][а-яёa-z0-9]+)*\b', (text or "").lower())
    qok = re.findall(r'\b[а-яёa-z0-9]+(?:[-/][а-яёa-z0-9]+)*\b', (q or "").lower())
    qset = {t for t in qok if len(t) > 2}
    tset = {t for t in tok if len(t) > 2}
    return (len(qset & tset) / (len(qset) or 1))

def _merge_hits(list_of_hitlists: List[List[Dict]], k: int) -> List[Dict]:

    best_by_id: Dict[int, Dict] = {}
    for hits in list_of_hitlists:
        for h in hits:
            hid = int(h["id"])
            cur = best_by_id.get(hid)
            if (cur is None) or (h["score"] > cur["score"]):
                best_by_id[hid] = h
    merged = list(best_by_id.values())
    # сортируем по faiss-score убыв.
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:k*3]  # оставим запас для re-rank

def _ce_rerank(query: str, hits: List[Dict]) -> List[Dict]:

    if not hits:
        return []
    ce = _get_cross_encoder()
    if ce is None:
        # без CE — комбинируем faiss + лёгкий лексический сигнал (нормируем).
        lex = np.array([_lex_overlap(query, h.get("text") or "") for h in hits], dtype="float32")
        faiss = np.array([float(h.get("score", 0.0)) for h in hits], dtype="float32")
        # нормировка на [0..1]
        if faiss.size and (faiss.max() - faiss.min()) > 1e-6:
            faiss = (faiss - faiss.min()) / (faiss.max() - faiss.min())
        if lex.size and (lex.max() - lex.min()) > 1e-6:
            lex = (lex - lex.min()) / (lex.max() - lex.min())
        comb = 0.75 * faiss + 0.25 * lex
        for h, s in zip(hits, comb.tolist()):
            h["combined_score"] = float(s)
        hits.sort(key=lambda x: x["combined_score"], reverse=True)
        return hits

    pairs = [(query, h.get("text") or "") for h in hits]
    try:
        scores = ce.predict(pairs, batch_size=CE_BATCH)
    except Exception:
        logger.exception("CrossEncoder.predict failed — fallback to FAISS")
        if max(float(h.get("score", 0.0)) for h in hits[:2]) >= 0.85:
            for h in hits:
                h["combined_score"] = float(h.get("score", 0.0))
            hits.sort(key=lambda x: x["combined_score"], reverse=True)
            return hits

    scores = np.asarray(scores, dtype="float32")
    if (scores.max() - scores.min()) > 1e-6:
        ce_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        ce_norm = scores * 0.0

    faiss = np.array([float(h.get("score", 0.0)) for h in hits], dtype="float32")
    if (faiss.max() - faiss.min()) > 1e-6:
        faiss_norm = (faiss - faiss.min()) / (faiss.max() - faiss.min())
    else:
        faiss_norm = faiss * 0.0

    comb = CE_WEIGHT * ce_norm + FAISS_WEIGHT * faiss_norm
    for h, s, ce_s in zip(hits, comb.tolist(), ce_norm.tolist()):
        h["rerank_score"] = float(ce_s)
        h["combined_score"] = float(s)

    hits.sort(key=lambda x: x["combined_score"], reverse=True)
    return hits


FAST_MODE = os.getenv("FAST_MODE", "1") == "1"
FAST_TOP_SCORE = float(os.getenv("FAST_TOP_SCORE", "0.83"))  # порог «и так хорошо»

def retrieve_with_rerank(query: str, *, chat_tail: Optional[str] = None, k: int = RETRIEVE_TOP_K) -> Tuple[str, List[Dict]]:
    q2 = _normalize_query(query)

    if FAST_MODE:
        try:
            hits_fast = _vec_search(q2, k=k)
        except Exception:
            logger.exception("vector_search failed (fast path)")
            hits_fast = []
        if hits_fast:
            best = float(hits_fast[0].get("score", 0.0))
            if best >= FAST_TOP_SCORE:
                # проставим combined_score для совместимости
                for h in hits_fast:
                    h["combined_score"] = float(h.get("score", 0.0))
                return q2, hits_fast

    variants = _expand_query(q2)
    try:
        all_lists = _vec_search_batch(variants, k=k)  # ← вместо цикла по _vec_search
    except Exception:
        logger.exception("batch search failed; fallback to per-variant")
        all_lists = []
        for v in variants:
            try:
                all_lists.append(_vec_search(v, k=k))
            except Exception:
                all_lists.append([])

    merged = _merge_hits(all_lists, k)
    reranked = _ce_rerank(q2, merged)
    topk = (reranked or [])[:k]
    for h in topk:
        if "combined_score" in h:
            h["score"] = float(h["combined_score"])
    return q2, topk
