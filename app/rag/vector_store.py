# app/rag/vector_store.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer

try:
    from app.data.source_map import SOURCE_MAP
except ImportError:
    SOURCE_MAP = {}

# ====== конфиг по умолчанию (переопределяется через env) ======
INDEX_DIR = os.environ.get("INDEX_DIR", "app/data/index")
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-m3")
TOP_K = int(os.environ.get("TOP_K", 4))
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", 4))

_METAS: Optional[List[Dict]] = None
_INDEX = None
_EMB: Optional[SentenceTransformer] = None


def warmup_vector_store():
    _load_metas()
    _load_index()
    _load_emb()


def _load_metas() -> List[Dict]:
    global _METAS
    if _METAS is None:
        with open(Path(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
            _METAS = json.load(f)
    return _METAS


def _load_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    return _INDEX


def _load_emb() -> SentenceTransformer:
    global _EMB
    if _EMB is None:
        _EMB = SentenceTransformer(EMB_MODEL)
    return _EMB


@dataclass
class Hit:
    rank: int
    score: float
    text: str
    file: str
    title: str
    id: int
    url: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _make_hit(m: Dict, rank: int, score: float) -> Dict:
    """Единый конструктор словаря хита из меты FAISS и ранга/скора."""
    fname = Path(m["file"]).name  # нужен для SOURCE_MAP
    return Hit(
        rank=rank,
        score=float(score),
        text=m.get("text", ""),
        file=m["file"],
        title=m["title"],
        id=m["id"],
        url=SOURCE_MAP.get(fname),
    ).to_dict()


def search(query: str, k: int = TOP_K) -> List[Dict]:
    metas = _load_metas()
    index = _load_index()
    emb = _load_emb()

    qv = emb.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")
    D, I = index.search(qv, k)

    hits: List[Dict] = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0:
            continue
        m = metas[idx]
        hits.append(_make_hit(m, rank, score))
    return hits


def search_batch(queries: List[str], k: int = RETRIEVE_TOP_K) -> List[List[Dict]]:
    metas = _load_metas()
    index = _load_index()
    emb = _load_emb()

    qv = emb.encode(queries, normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")
    D, I = index.search(qv, k)

    out: List[List[Dict]] = []
    for d_row, i_row in zip(D, I):
        hits: List[Dict] = []
        rank = 1
        for idx, score in zip(i_row, d_row):
            if idx < 0:
                continue
            m = metas[idx]
            hits.append(_make_hit(m, rank, score))
            rank += 1
        out.append(hits)
    return out


def norm_source(path_str: str) -> str:
    return Path(path_str).name
