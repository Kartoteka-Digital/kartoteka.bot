# rag/rag.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Фасад: наружу отдаём только функцию answer()."""

from typing import Optional, Dict, List
from .answer_engine import AnswerEngine

# один общий экземпляр на процесс
_engine = AnswerEngine(temperature=0.2)

def answer(
    question: str,
    chat_tail: Optional[str] = None,
    prefetched_hits: Optional[List[Dict]] = None,
) -> Dict:
    """Сформировать ответ через RAG + LLM.
    :param question: текст вопроса пользователя
    :param chat_tail: (опционально) хвост диалога для контекста
    :param prefetched_hits: (опционально) результаты поиска, полученные заранее
    :return: {"answer": "..."}
    """
    return _engine.answer(question, chat_tail, prefetched_hits = prefetched_hits)

__all__ = ["answer"]

if __name__ == "__main__":
    # мини-CLI для локальной проверки
    import sys
    q = " ".join(sys.argv[1:]) or "Что умеет бот?"
    print(answer(q)["answer"])
