import re
from typing import TypedDict, Literal, List, Set

QType = Literal["definition","clarification","howto","binary","other"]
class NLUResult(TypedDict):
    forbid_phrases: List[str]
    require_terms: Set[str]
    focus_terms: List[str]
    qtype: QType
    normalized: str

TOKEN = re.compile(r"[a-zA-Zа-яА-Я0-9]+(?:[-\s][a-zA-Zа-яА-Я0-9]+)*")
NEG = re.compile(r"\b(?:нет|без|не(?:\s+нужен|(?:\s+)?требуется))\s+([^.,;:!?()\[\]{}]+)", re.I)
DEF = re.compile(r"\b(?:что такое|что за)\s+(.+?)\??$", re.I)
CLAR = re.compile(r"\b(?:какой|какая|какие|каков)\s+(.+?)\??$", re.I)
HOW = re.compile(r"\b(?:как|каким образом)\b", re.I)
BIN = re.compile(r"\b(?:можно ли|нужно ли|это ли)\b", re.I)
STOP = {"как","что","за","ли","и","или","у","меня","на","если","нет","без",
        "не","нужен","нужна","нужно","требуется","вообще","тогда","просто"}

def _tokens(s: str) -> List[str]:
    return [t.lower().replace("  ", " ").strip(" -") for t in TOKEN.findall(s.lower())]

def parse_question(q: str) -> NLUResult:
    low = (q or "").strip().lower()
    forbid_phrases: List[str] = []
    for m in NEG.finditer(low):
        toks = [t for t in _tokens(m.group(1)) if t not in STOP]
        if toks:
            forbid_phrases.append(" ".join(toks[:3]))

    qtype: QType = "other"
    focus_terms: List[str] = []
    mm = DEF.search(low)
    if mm:
        qtype = "definition"; focus_terms = [t for t in _tokens(mm.group(1)) if t not in STOP]
    else:
        mm = CLAR.search(low)
        if mm:
            qtype = "clarification"; focus_terms = [t for t in _tokens(mm.group(1)) if t not in STOP]
        elif HOW.search(low): qtype = "howto"
        elif BIN.search(low): qtype = "binary"

    require_terms: Set[str] = {t for t in _tokens(low) if t not in STOP}
    for ph in forbid_phrases:
        for t in _tokens(ph):
            require_terms.discard(t)

    return {
        "forbid_phrases": forbid_phrases,
        "require_terms": require_terms,
        "focus_terms": [t for t in focus_terms if t not in STOP],
        "qtype": qtype,
        "normalized": low,
    }
