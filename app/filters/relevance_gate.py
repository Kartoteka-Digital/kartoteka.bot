# app/filters/relevance_gate.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import os
import re
import logging

from app.config import LLMRelevanceSettings, llm_gate_settings
from app.filters.lexicon import DOMAIN_KEYWORDS, SMALLTALK_PATTERNS, QUESTION_PREFIXES
from app.filters.rules import parse_question, NLUResult

logger = logging.getLogger(__name__)

try:
    # используем тот же поиск, что и раньше
    from app.rag.vector_store import search as _rag_search
except Exception as exc:  # pragma: no cover
    _rag_search = None
    logger.warning("RAG поиск недоступен: %s", exc)

# ---- ENV-настройки ранних проверок (не про LLM gate) ----
MIN_WORDS = int(os.getenv("ASK_MIN_WORDS", 2))
MIN_CHARS = int(os.getenv("ASK_MIN_CHARS", 8))
OFFTOP_THRESHOLD = float(os.getenv("ASK_OFFTOP_THRESHOLD", 0.2))

LexOverlapFn = Callable[[str, str], float]

def _looks_like_smalltalk(text: str) -> bool:
    t = (text or "").strip()
    return any(re.match(pat, t, re.IGNORECASE) for pat in SMALLTALK_PATTERNS)

def _domain_hint(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in DOMAIN_KEYWORDS)

def _looks_like_question(text: str) -> bool:
    t = (text or "").lower().strip()
    return t.endswith("?") or any(t.startswith(p) for p in QUESTION_PREFIXES)

def _default_lex_overlap(query: str, text: str) -> float:
    tok = re.findall(r'\b[а-яёa-z0-9]+(?:[-/][а-яёa-z0-9]+)*\b', (text or "").lower())
    qok = re.findall(r'\b[а-яёa-z0-9]+(?:[-/][а-яёa-z0-9]+)*\b', (query or "").lower())
    q = {t for t in qok if len(t) > 2}
    t = {t for t in tok if len(t) > 2}
    return (len(q & t) / (len(q) or 1))

@dataclass
class GateVerdict:
    allow: bool
    reason: Optional[str]
    hits: List[Dict]
    settings_used: LLMRelevanceSettings
    nlu: NLUResult


class RelevanceGate:

    """
    1) ранние проверки,
    2) поиск ,
    3) пороги на top-hit (score/overlap),
    4) LLM-фолбэк (опционально).
    """
    def __init__(
        self,
        *,
        settings: Optional[LLMRelevanceSettings] = None,
        lex_overlap: LexOverlapFn = _default_lex_overlap,
        ask_relevance_llm: Optional[Callable[..., bool]] = None,
    ) -> None:
        self._base_settings = settings or llm_gate_settings()
        self._lex_overlap = lex_overlap
        self._ask_relevance_llm = ask_relevance_llm

    def _settings_for(self, q: str) -> LLMRelevanceSettings:
        low = (q or "").lower()
        token_count = len(re.findall(r'\b[а-яёa-z0-9]+\b', low))
        is_short = token_count <= 6
        is_where = re.search(r"\bгде\b", low) is not None
        has_fs_hint = re.search(r"(папк|путь|скриншот|screenshots|видео|oculus)", low) is not None

        if is_short or is_where or has_fs_hint:
            return LLMRelevanceSettings(
                min_score=0.40,
                min_overlap=0.08,
                fallback_enabled=True,
                fallback_score_threshold=0.55,
                fallback_overlap_trigger=0.20,
            )
        return self._base_settings

    def check(
        self,
        *,
        question: str,
        chat_tail: Optional[str] = None,
        prefetched_hits: Optional[List[Dict]] = None,
        search_fn: Optional[Callable[[str], List[Dict]]] = None,
    ) -> GateVerdict:
        q = (question or "").strip()

        # --- 0) ранние проверки (бывший early_gate) ---
        if len(q) < MIN_CHARS or len(q.split()) < MIN_WORDS:
            return GateVerdict(False, " ⚠️ Вопрос слишком короткий. Уточните, пожалуйста, что именно из документации нужно.", [], self._base_settings, parse_question(q))

        if _looks_like_smalltalk(q):
            return GateVerdict(False, "ℹ️ Я отвечаю только по документации Kartoteka.", [], self._base_settings, parse_question(q))

        if not _domain_hint(q) and not _looks_like_question(q):
            return GateVerdict(False, "🤔 Похоже, это не вопрос по документации.", [], self._base_settings, parse_question(q))

        # --- 1) поиск/хиты ---
        hits = list(prefetched_hits or [])
        if not hits:
            fn = search_fn or _rag_search
            if fn is not None:
                try:
                    hits = fn(q) or []
                except Exception:
                    logger.exception("Ошибка поиска RAG")
                    hits = []

        if not hits:
            return GateVerdict(False, "🔍 Не нашёл ничего похожего в документации. Проверь формулировку и попробуй уточнить запрос.", [], self._base_settings, parse_question(q))

        # оффтоп по лучшему скору
        best = hits[0]
        best_score = float(best.get("score", 0.0))
        if best_score < OFFTOP_THRESHOLD:
            return GateVerdict(False, "🔎 Не нашёл ничего похожего в документации. Проверь формулировку и попробуй уточнить запрос.", hits, self._base_settings, parse_question(q))

        # --- 2) вычисляем overlap и пороги ---
        settings = self._settings_for(q)
        overlap = self._lex_overlap(q, (best.get("text") or ""))

        # быстрая проверка твёрдых порогов
        if best_score >= settings.min_score and overlap >= settings.min_overlap:
            return GateVerdict(True, None, hits, settings, parse_question(q))

        # бордерлайн: немного мягче, без LLM
        borderline_score = settings.min_score * 0.85 if settings.min_score > 0 else settings.min_score
        borderline_overlap = settings.min_overlap * 0.6 if settings.min_overlap > 0 else settings.min_overlap
        if best_score >= borderline_score and overlap >= borderline_overlap:
            return GateVerdict(True, None, hits, settings, parse_question(q))

        # --- 3) LLM-фолбэк (опциональный) ---
        if not settings.fallback_enabled:
            return GateVerdict(False, "🙂 Недостаточно информации в базе, чтобы дать точный ответ. "
                                  "Пожалуйста, уточните вопрос конкретнее.", hits, settings, parse_question(q))

        if best_score < settings.fallback_score_threshold:
            return GateVerdict(False, "🙂 Недостаточно информации в базе, чтобы дать точный ответ. "
                                  "Пожалуйста, уточните вопрос конкретнее.", hits, settings, parse_question(q))

        if overlap >= settings.fallback_overlap_trigger:
            return GateVerdict(False, "🙂 Недостаточно информации в базе, чтобы дать точный ответ. "
                                  "Пожалуйста, уточните вопрос конкретнее.", hits, settings, parse_question(q))

        # спросим быстрый «да/нет» у LLM (или подставного судьи)
        if self._ask_relevance_llm:
            ok = bool(self._ask_relevance_llm(query=q, hit_text=best.get("text") or ""))
        else:
            # молча разрешаем, если score достаточно высок
            ok = best_score >= settings.fallback_score_threshold

        if ok:
            return GateVerdict(True, None, hits, settings, parse_question(q))
        return GateVerdict(False, "🙂 Недостаточно информации в базе, чтобы дать точный ответ. "
                                  "Пожалуйста, уточните вопрос конкретнее.", hits, settings, parse_question(q))
