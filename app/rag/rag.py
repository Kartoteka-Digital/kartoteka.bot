
"""Фасад: наружу отдаём только функцию answer()."""

from typing import Optional, Dict, List
from .answer_engine import AnswerEngine

_engine = AnswerEngine(temperature=0.2)

def answer(
    question: str,
    chat_tail: Optional[str] = None,
    prefetched_hits: Optional[List[Dict]] = None,
) -> Dict:

    return _engine.answer(question, chat_tail, prefetched_hits = prefetched_hits)

__all__ = ["answer"]

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Что умеет бот?"
    print(answer(q)["answer"])
