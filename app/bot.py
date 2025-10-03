"""
Telegram-бот на aiogram.
Для прод — перевести на webhook и повесить под systemd.
"""
import os
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from collections import defaultdict, deque
from app.rag.rag import answer
# from app.filters.heuristics import early_gate
from app.rag.vector_store import search as vec_search
from app.rag.vector_store import warmup_vector_store
from app.filters.relevance_gate import RelevanceGate
from app.rag.retrieval_pipeline import retrieve_with_rerank

import logging
from openai import APIError, APIConnectionError, RateLimitError

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_TOKEN")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

gate = RelevanceGate()

USE_RETRIEVAL_PIPELINE = os.getenv("USE_RETRIEVAL_PIPELINE", "0") == "1"
SMART_PIPELINE_THRESHOLD = float(os.getenv("SMART_PIPELINE_THRESHOLD", "0.80"))

def smart_search(q: str):
    # 1) быстрый FAISS
    hits = vec_search(q)
    if not USE_RETRIEVAL_PIPELINE:
        return hits
    if not hits:
        # нет совпадений — пробуем «умный» пайплайн
        return retrieve_with_rerank(q)[1]

    # 2) если уверенность низкая — подключаем пайплайн
    best = float(hits[0].get("score", 0.0))
    if best < SMART_PIPELINE_THRESHOLD:
        return retrieve_with_rerank(q)[1]

    # 3) иначе остаёмся на быстром поиске
    return hits

CHAT_TAIL: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=4)) # 4 последние реплики

@dp.message(CommandStart())
async def start(m: Message):
    await m.answer(
        "👋 Привет! Я Kartoteka.Support. Задавай вопрос по документации.\n\n"
        "⏳ Пожалуйста, имей в виду: иногда ответ может занять чуть больше времени."
    )

# bot.py — обработчик ask_cmd: заменяем early_gate(...) на gate.check(...)
@dp.message(Command("ask"))
async def ask_cmd(m: Message):
    parts = m.text.split(maxsplit=1)
    user_q = parts[1].strip() if len(parts) > 1 else ""

    verdict = gate.check(
        question=user_q,
        search_fn=smart_search,  # единая точка поиска
    )
    if not verdict.allow:
        await m.answer(verdict.reason or "Пожалуйста, уточни вопрос по документации.")
        return

    await m.chat.do("typing")

    try:
        out = answer(
            user_q,
            chat_tail="\n".join(CHAT_TAIL[m.chat.id]),
            prefetched_hits=verdict.hits,  # уже проверенные и релевантные
        )
    except (APIError, APIConnectionError, RateLimitError):
        logging.exception("Ошибка при ответе")
        await m.answer("Упс! Что-то пошло не так. Попробуй ещё раз или уточни вопрос.")
        return

    CHAT_TAIL[m.chat.id].append(f"Пользователь: {user_q}\nБот: {out['answer']}")
    await m.answer(out["answer"])

    if out.get("links"):
        seen, lines = set(), []
        i = 1
        for item in out["links"]:
            url = str(item.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = (item.get("title") or "Источник").strip()
            if title.lower().endswith(".md"):
                from pathlib import Path
                title = Path(title).stem
            lines.append(f"{i}) {title} - {url}")
            i += 1
        if lines:
            await m.answer("🔗 Возможные источники:\n" + "\n".join(lines), disable_web_page_preview=True)


@dp.message(F.text)
async def any_text(m: Message):
    # любой текст — как вопрос
    await ask_cmd(m)

async def main():
    warmup_vector_store()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())