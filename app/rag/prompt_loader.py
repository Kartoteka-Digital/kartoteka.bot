from __future__ import annotations
from pathlib import Path
from functools import lru_cache

PROMPTS_DIR = Path(__file__).parent / "prompts"

@lru_cache(maxsize=1)
def load_base_prompt() -> str:
    path = PROMPTS_DIR / "base.md"
    return path.read_text(encoding="utf-8").strip()

@lru_cache(maxsize=1)
def load_static_policy() -> str:
    path = PROMPTS_DIR / "policy.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""
