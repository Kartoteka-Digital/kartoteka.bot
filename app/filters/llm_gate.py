"""Utilities for judging retrieval relevance with an LLM fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol

from app.config import LLMRelevanceSettings, llm_gate_settings


class RelevanceJudge(Protocol):
    """Callable capable of returning a boolean relevance verdict."""

    def __call__(self, *, query: str, hit_text: str) -> bool:
        ...


@dataclass
class RetrievalHit:
    """Simple container describing a retrieved chunk."""
    text: str
    score: float
    overlap: float


def llm_relevance_gate(
    query: str,
    hit: RetrievalHit,
    *,
    settings: LLMRelevanceSettings | None = None,
    ask_relevance_llm: Callable[..., bool] | None = None,
) -> bool:
    """Return ``True`` when the hit passes the relevance filter."""

    cfg = settings or llm_gate_settings()
    judge = ask_relevance_llm or _ask_relevance_llm

    if hit.score >= cfg.min_score and hit.overlap >= cfg.min_overlap:
        return True

    if cfg.min_score > 0:
        borderline_score = cfg.min_score * 0.85
    else:
        borderline_score = cfg.min_score
    if cfg.min_overlap > 0:
        borderline_overlap = cfg.min_overlap * 0.6
    else:
        borderline_overlap = cfg.min_overlap
    if hit.score >= borderline_score and hit.overlap >= borderline_overlap:
        return True

    if not cfg.fallback_enabled:
        return False

    if hit.score < cfg.fallback_score_threshold:
        return False

    if hit.overlap >= cfg.fallback_overlap_trigger:
        return False

    return bool(
        judge(
            query=query,
            hit_text=hit.text,
            lexical_score=hit.score,
            lexical_overlap=hit.overlap,
            settings=cfg,
        )
    )


def _ask_relevance_llm(
    *,
    query: str,
    hit_text: str,
    lexical_score: float,
    lexical_overlap: float,
    settings: LLMRelevanceSettings,
    llm_client: RelevanceJudge | None = None,
) -> bool:
    """Ask the LLM (or a heuristic stand-in) whether the hit is relevant."""

    judge = llm_client
    if judge is None and os.getenv("LLM_GATE_USE_LLM", "0") == "1":
        try:
            from app.rag.llm_client import get_client, get_model_name
        except Exception:
            judge = None
        else:
            client = get_client()
            model = get_model_name()

            def _client_call(*, query: str, hit_text: str) -> bool:
                prompt = (
                    "Ты помощник, который решает, подходит ли найденный фрагмент "
                    "вопросу пользователя. Ответь 'yes' или 'no'.\n"
                    f"Вопрос: {query}\nФрагмент: {hit_text}"
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1,
                )
                text = (response.choices[0].message.content or "").strip().lower()
                return text.startswith("y")

            judge = _client_call

    if judge is not None:
        return bool(judge(query=query, hit_text=hit_text))

    if lexical_score >= settings.fallback_score_threshold:
        return True

    return lexical_overlap >= settings.min_overlap