# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple, Optional
from .rules import parse_question, NLUResult

def _extract_tail_keywords(chat_tail: Optional[str]) -> List[str]:
    if not chat_tail:
        return []

    keywords: List[str] = []
    # Разбираем только пользовательские реплики («Пользователь: …»)
    tail_lines = [
        line.split(":", 1)[1].strip()
        for line in chat_tail.splitlines()
        if line.strip().lower().startswith("пользователь:")
    ]

    # Берём несколько последних вопросов пользователя — они ближе всего к анафоре.
    for utterance in tail_lines[-3:]:
        if not utterance:
            continue
        tail_nlu = parse_question(utterance)

        for term in tail_nlu.get("focus_terms", []):
            if term and term not in keywords:
                keywords.append(term)

        for term in sorted(tail_nlu.get("require_terms", set())):
            if len(term) <= 3:
                continue
            if term not in keywords:
                keywords.append(term)

        if len(keywords) >= 8:
            break

    return keywords[:8]


def rewrite_query(original: str, chat_tail: Optional[str] = None) -> Tuple[str, NLUResult]:
    nlu = parse_question(original)
    q2 = original
    if nlu["focus_terms"]:
        q2 += " " + " ".join(nlu["focus_terms"][:4])  # мягкий буст
    for ph in nlu["forbid_phrases"]:
        q2 += " " + f"без {ph}"
    tail_keywords = _extract_tail_keywords(chat_tail)
    if tail_keywords:
        q2 += " " + " ".join(tail_keywords)
        # универсальная подсказка
    return q2, nlu

def _contains_all(text_low: str, phrase: str) -> bool:
    return all(t in text_low for t in phrase.lower().split())

def penalize_hits(hits: List[Dict], nlu: NLUResult) -> List[Dict]:
    if not hits: return hits
    req, forb = nlu.get("require_terms", set()), nlu.get("forbid_phrases", [])
    out: List[Dict] = []
    for h in hits:
        txt = ((h.get("title") or "") + " " + (h.get("text") or "")).lower()
        score = float(h["score"])
        for ph in forb:
            if _contains_all(txt, ph):
                score *= 0.45              # сильный штраф нарушителям запрета
        if req and any(t in txt for t in req): score *= 1.10   # небольшой буст
        elif req: score *= 0.90
        hh = dict(h); hh["score"] = score; out.append(hh)
    out.sort(key=lambda x: x["score"], reverse=True)
    for i,h in enumerate(out,1): h["rank"]=i
    return out
