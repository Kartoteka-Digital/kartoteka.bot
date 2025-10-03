from __future__ import annotations

from dataclasses import dataclass
import os
from functools import lru_cache


@dataclass(frozen=True)
class LLMRelevanceSettings:
    """Runtime settings for the LLM relevance gate."""

    min_score: float = float(os.getenv("LLM_GATE_MIN_SCORE", "0.55"))
    min_overlap: float = float(os.getenv("LLM_GATE_MIN_OVERLAP", "0.35"))
    fallback_enabled: bool = os.getenv("LLM_GATE_ENABLE_FALLBACK", "1") == "1"
    fallback_score_threshold: float = float(
        os.getenv("LLM_GATE_FALLBACK_MIN_SCORE", "0.75")
    )
    fallback_overlap_trigger: float = float(
        os.getenv("LLM_GATE_FALLBACK_OVERLAP_TRIGGER", "0.15")
    )


@lru_cache(maxsize=1)
def llm_gate_settings() -> LLMRelevanceSettings:
    """Return the memoised relevance gate settings."""

    return LLMRelevanceSettings()