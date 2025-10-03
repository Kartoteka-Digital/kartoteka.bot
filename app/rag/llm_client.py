import os
from typing import Any, TYPE_CHECKING

try:
    from openai import OpenAI as _OpenAI
except ModuleNotFoundError:
    _OpenAI = None

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIType
else:
    OpenAIType = Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        return None

load_dotenv()

OPENAI_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

if not OPENAI_API_KEY:
    raise RuntimeError("Не задан OPENAI_API_KEY/GROQ_API_KEY")

_client: OpenAIType | None = None

def get_client() -> OpenAIType:
    if _OpenAI is None:
        raise RuntimeError("Библиотека openai не установлена")
    global _client
    if _client is None:
        _client = _OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client

def get_model_name() -> str:
    return LLM_MODEL
