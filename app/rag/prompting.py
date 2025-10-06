from __future__ import annotations
from typing import Optional, Dict, List
import re
from app.rag.prompt_loader import load_base_prompt, load_static_policy

MD_NAME_RE = re.compile(r"[«\"']?[\w\s\-/()]+\.md[»\"']?", re.IGNORECASE)
SECTION_REF_RE = re.compile(r"(?:в\s+разделе|см\.\s*раздел|смотрите\s+раздел)\s+[«\"']?[^»\"'\n]+[»\"']?",
                            re.IGNORECASE)

def _strip_md_refs(s: str) -> str:
    if not s:
        return s
    s = SECTION_REF_RE.sub("в документации", s)
    s = MD_NAME_RE.sub("документации", s)
    return s

def build_context(hits, max_chars: int = 8000, hide_titles: bool = True) -> str:
    parts: List[str] = []
    for h in hits:
        txt = _strip_md_refs((h.get("text") or "").strip())
        if not txt:
            continue
        if hide_titles:
            parts.append(txt + "\n")
        else:
            title = _strip_md_refs((h.get("title") or "").strip())
            parts.append(f"### {title}\n{txt}\n")
    return ("\n\n---\n\n".join(parts))[:max_chars]

def _build_dynamic_policy(nlu: Optional[Dict]) -> str:
    if not nlu:
        return ""

    bullets: List[str] = []

    forbid = (nlu.get("forbid_phrases") or [])
    if forbid:
        bullets.append("Запрещено упоминать/советовать: " + "; ".join(forbid))

    req = (nlu.get("require_terms") or [])
    if req:
        # не спамим — берём 4–6 штук максимум
        bullets.append("Обязательно упомяни: " + ", ".join(sorted(set(req))[:6]))

    if nlu.get("qtype") in ("definition", "clarification"):
        fterms = nlu.get("focus_terms") or []
        if fterms:
            bullets.append(f"Начни с краткого определения: «{fterms[0]}» (1–2 предложения).")

    bullets.append("Не упоминай названия файлов/разделов .md; при необходимости говори просто «в документации».")

    if not bullets:
        return ""

    return "- " + "\n- ".join(bullets)

def build_messages(
    *,
    context: str,
    question: str,
    chat_tail: Optional[str] = None,
    nlu: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """
    Простой и предсказуемый конструктор сообщений:
      - System: base.txt (+ статическая policy.md, если есть)
      - User: правила (только нужные), контекст, хвост диалога и текущий вопрос
    """
    system = load_base_prompt()
    static_policy = load_static_policy()
    dynamic_policy = _build_dynamic_policy(nlu)

    # Собираем единый policy-блок (если есть что добавлять)
    policy_parts = []
    if static_policy:
        policy_parts.append(static_policy)
    if dynamic_policy:
        policy_parts.append(dynamic_policy)
    policy_block = "\n".join(policy_parts).strip()

    # Формируем user-сообщение
    parts: List[str] = []
    if policy_block:
        parts.append("Правила ответа:\n" + policy_block)

    parts.append("Контекст знаний:\n" + (context or "").strip())

    if chat_tail:
        parts.append("Предыдущий контекст диалога:\n" + chat_tail.strip())

    parts.append("Текущий вопрос:\n" + question.strip())
    parts.append("Ответь кратко и по делу. Если в контексте нет ответа — так и скажи и уточни, что нужно.")

    user_content = "\n\n".join(parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]