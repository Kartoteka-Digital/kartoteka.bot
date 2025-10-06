# app/rag/answer_engine.py
from __future__ import annotations

from typing import Optional, Dict, List
from pathlib import Path
import re
import logging

from .vector_store import search, norm_source
from .prompting import build_messages, build_context
from .llm_client import get_client, get_model_name

from app.filters.rules import NLUResult

logger = logging.getLogger(__name__)

# ===== Специфичные паттерны для проверки «следов» источника в ответе =====
SPECIFIC_PATTERNS = [
    re.compile(r"\b[A-Z][a-z]+/[A-Z][a-z]+\b", re.IGNORECASE),  # Oculus/Screenshots
    re.compile(r"\b\w+\.apk\b", re.IGNORECASE),  # file.apk
    re.compile(r"\badb\s+\w+\b", re.IGNORECASE),  # adb install
    re.compile(r"--\w+", re.IGNORECASE),  # --flag
    re.compile(
        r"\b(?:папк[аеи]|файл[ае]?|путь)\s+[\"']?[\w/.-]+[\"']?",
        re.IGNORECASE,
    ),
]

MD_STRIP_PATTERNS = (
    (re.compile(r"\*\*(.*?)\*\*", re.DOTALL), r"\1"),  # **жирный**
    (re.compile(r"__(.*?)__", re.DOTALL), r"\1"),      # __жирный__
    (re.compile(r"\*(.*?)\*", re.DOTALL), r"\1"),      # *курсив*
    (re.compile(r"_(.*?)_", re.DOTALL), r"\1"),        # _курсив_
    (re.compile(r"`{1,3}([^`]+)`{1,3}"), r"\1"),       # `код`
)


def _strip_md_inline(text: str) -> str:
    out = text or ""
    for pat, repl in MD_STRIP_PATTERNS:
        out = pat.sub(repl, out)
    return out

class AnswerEngine:
    def __init__(self, temperature: float = 0.2):
        self.temperature = temperature
        self.client = get_client()
        self.model = get_model_name()

    def _display_title(self, h: Dict) -> str:
        title = (h.get("title") or Path(h["file"]).stem).strip()
        return Path(title).stem if title.lower().endswith(".md") else title

    def _extract_keywords(self, text: str) -> set[str]:
        tokens = re.findall(r'\b[а-яёa-z0-9]+(?:[-/][а-яёa-z0-9]+)*\b', (text or "").lower())
        return {t for t in tokens if len(t) > 2}

    def _lexical_overlap(self, query: str, text: str) -> float:
        q = self._extract_keywords(query)
        t = self._extract_keywords(text or "")
        return (len(q & t) / len(q)) if q else 0.0

    # ---------- Оценка «использования» источника в ответе ----------
    def _answer_uses_source(
            self,
            hit: Dict,
            answer_lower: str,
            answer_kw: set[str],
            specific_matches: Dict[re.Pattern, List[str]],
    ) -> float:
        source_text = (hit.get("text") or "").lower()
        source_title = (hit.get("title") or "").lower()

        if not source_text or not answer_lower:
            return 0.0

        score = 0.0

        # 1) Точное совпадение 3-словных фраз
        words = answer_lower.split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i + 3])
            if len(phrase) > 10 and phrase in source_text:
                score += 0.2
                break

        # 2) Пересечение ключевых слов (основной вес)
        source_kw = self._extract_keywords(source_text + " " + source_title)
        if answer_kw:
            common = answer_kw & source_kw
            overlap_ratio = len(common) / len(answer_kw)
            score += overlap_ratio * 0.6

        # 3) Совпадения специальных паттернов (пути/файлы/флаги)
        for pattern in SPECIFIC_PATTERNS:
            matches = specific_matches.get(pattern)
            if not matches:
                continue
            for match in matches:
                if match in source_text:
                    score += 0.25
                    break

        return min(1.0, score)

    # ---------- Отбор ссылок ----------
    def _pick_links(self, hits: List[Dict], answer: str, nlu: Optional[NLUResult] = None) -> tuple[
        List[Dict], List[Dict]]:
        by_file: Dict[str, Dict] = {}
        for h in hits:
            key = norm_source(h["file"])
            cur = by_file.get(key)
            if not cur:
                by_file[key] = {
                    "title": self._display_title(h),
                    "url": h.get("url"),
                    "best_score": float(h["score"]),
                    "first_rank": int(h["rank"]),
                    "hit": h,  # сохраняем для анализа
                }
            else:
                if h["score"] > cur["best_score"]:
                    cur["best_score"] = float(h["score"])
                    cur["hit"] = h
                if h["rank"] < cur["first_rank"]:
                    cur["first_rank"] = int(h["rank"])

        # анализ использования источников ответом
        answer_lower = (answer or "").lower()
        answer_words = answer_lower.split()
        answer_kw = self._extract_keywords(answer or "")
        specific_matches: Dict[re.Pattern, List[str]] = {}
        for pattern in SPECIFIC_PATTERNS:
            found = [m.lower() for m in pattern.findall(answer or "")]
            if found:
                specific_matches[pattern] = found

        for item in by_file.values():
            usage_score = self._answer_uses_source(
                item["hit"],
                answer_lower=answer_lower,
                answer_kw=answer_kw,
                specific_matches=specific_matches,
            )
            item["usage_score"] = usage_score
            item["combined_score"] = item["best_score"] * 0.6 + usage_score * 0.4

        all_items = sorted(by_file.values(), key=lambda x: (-x["combined_score"], x["first_rank"]))
        if not all_items:
            return [], []

        # пороги
        MIN_USAGE = 0.15
        DELTA_KEEP = 0.05
        MAX_LINKS_DEFAULT = 2

        best = all_items[0]
        best_combined = best["combined_score"]
        best_usage = best["usage_score"]

        if nlu and nlu.get("qtype") in ("definition", "clarification"):
            if best_combined > 0.4 or best_usage > 0.2 or best["best_score"] > 0.45:
                selected_items = [best]
            else:
                selected_items = []
        else:
            selected_items = [best]
            for it in all_items[1:]:
                if it["usage_score"] >= MIN_USAGE and (best_combined - it["combined_score"]) <= DELTA_KEEP:
                    selected_items.append(it)
                if len(selected_items) >= MAX_LINKS_DEFAULT:
                    break
            if best_usage < MIN_USAGE and best["best_score"] > 0.5:
                selected_items = [best]

        if len(selected_items) > 1 and selected_items[1]["usage_score"] < MIN_USAGE:
            selected_items = selected_items[:1]

        links, seen_urls = [], set()
        for it in selected_items:
            url: str = str(it.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            links.append({
                "title": it["title"],
                "url": url,
                "debug": {
                    "faiss": round(it["best_score"], 3),
                    "usage": round(it["usage_score"], 3),
                    "combined": round(it["combined_score"], 3),
                }
            })
        return links, all_items

    def answer(
            self,
            question: str,
            chat_tail: Optional[str] = None,
            prefetched_hits: Optional[List[Dict]] = None,
    ) -> Dict:
        # 0) Базовый «нет ответа»
        no_answer_payload = {
            "answer": (
                "В загруженных материалах нет прямого ответа. "
                "Можешь переформулировать вопрос или уточнить, что именно ты ищешь?"
            ),
            "links": [],
        }

        # 1) Берём подготовленные хиты от RelevanceGate
        if prefetched_hits is not None:
            hits = list(prefetched_hits)
            logger.debug(
                "AnswerEngine.answer: используем префетченные результаты (n=%d)",
                len(hits),
            )
        else:
            logger.debug("AnswerEngine.answer: нет префетченных результатов — запускаем поиск для %r", question)
            hits = search(question)

        if not hits:
            return no_answer_payload

        # 2) Контекст из хитов
        context = build_context(hits)
        if not context.strip():
            return no_answer_payload

        # 3) Промпт
        messages = build_messages(
            context=context,
            question=question,
            chat_tail=chat_tail,
            nlu=None,
        )

        # 4) Вызов модели
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_tokens=256,
        )
        content = resp.choices[0].message.content
        content = _strip_md_inline(content)

        # 5) Анализ источников/ссылки
        links, all_items = self._pick_links(hits, answer=content, nlu=None)

        # 6) Диагностика (без падений)
        try:
            print(f"\n[QUESTION] {question}")
            print(f"[ANSWER] {content[:100]}...")
            print("\n[TOP HITS]")
            for h in hits[:3]:
                print(f"  {h['rank']}. {h['title'][:40]} | FAISS={h['score']:.3f}")
            print("\n[SOURCE ANALYSIS]")
            for item in all_items[:3]:
                print(f"  📄 {item['title'][:40]}")
                print(
                    f"     FAISS={item['best_score']:.3f} | Usage={item.get('usage_score', 0):.3f} | Combined={item.get('combined_score', 0):.3f}"
                )
            print(f"\n[FINAL LINKS] {len(links)} источников")
            for l in links:
                dbg = l.get("debug") or {}
                print(f"  ✓ {l['title']} | {dbg}")
            print("-" * 80)
        except Exception as e:
            print(f"[DEBUG ERROR] {e}")

        return {"answer": content, "links": links}

