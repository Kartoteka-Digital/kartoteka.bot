from pathlib import Path
import os

# /app  ->  корень проекта: /<repo>
APP_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = APP_ROOT.parent

def _env(name: str, default=None):
    return os.environ.get(name, default)

# Пути данных (по умолчанию /<repo>/data)
DATA_DIR  = Path(_env("DATA_DIR", str(PROJ_ROOT / "data")))
INDEX_DIR = Path(_env("INDEX_DIR", str(DATA_DIR / "index")))
MD_DIR    = Path(_env("MD_DIR",    str(DATA_DIR / "md")))
OUT_DIR   = Path(_env("OUT_DIR",   str(INDEX_DIR)))

# Токены / API
TELEGRAM_TOKEN  = _env("TELEGRAM_TOKEN")
OPENAI_API_KEY  = _env("OPENAI_API_KEY") or _env("GROQ_API_KEY")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL       = _env("LLM_MODEL", "llama-3.1-8b-instant")

LOG_LEVEL       = _env("LOG_LEVEL", "INFO")
