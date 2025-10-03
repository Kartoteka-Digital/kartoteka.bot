#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сбор *.md → чанкинг (≈700 токенов, overlap 100) → эмбеддинги BAAI/bge-m3 → FAISS (cosine).
На выходе: index.faiss и meta.json в каталоге OUT_DIR.
"""
import os
import json
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

# эмбеддинги
from sentence_transformers import SentenceTransformer
# токенизатор для подсчёта токенов в bge-m3
from transformers import AutoTokenizer

# FAISS
import faiss

from dotenv import load_dotenv
load_dotenv()

MD_DIR = os.environ.get("MD_DIR", "app/data/md")              # где лежат .md
OUT_DIR = os.environ.get("OUT_DIR", "app/data/index")            # куда сохранить индексы
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", 700))   # целевой размер чанка (токены)
OVERLAP = int(os.environ.get("OVERLAP", 100))              # перехлёст (токены)
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-m3")

os.makedirs(OUT_DIR, exist_ok=True)

@dataclass
class MetaRow:
    id: int
    file: str
    title: str
    start_tok: int
    end_tok: int
    text: str

def read_markdown_files(md_dir: str) -> Dict[str, str]:
    files = {}
    for path in glob.glob(os.path.join(md_dir, "**", "*.md"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            files[path] = f.read()
    return files

def chunk_by_tokens(
    text: str, tokenizer, chunk_tokens: int, overlap: int
) -> List[Tuple[int, int, str]]:
    # Токенизируем весь текст (BPE ids)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        start_idx = i
        j = min(i + chunk_tokens, len(tokens))
        piece_ids = tokens[i:j]
        piece = tokenizer.decode(piece_ids)
        # лёгкая очистка
        piece = piece.strip().replace("\r", "")
        if piece:
            end_idx = start_idx + len(piece_ids)
            chunks.append((start_idx, end_idx, piece))
        if j == len(tokens):
            break
        i = j - overlap  # шаг с перехлёстом
        if i < 0:
            i = 0
    return chunks


def build_index():
    print("\n[1/4] Загрузка модели эмбеддингов:", EMB_MODEL)
    model = SentenceTransformer(EMB_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)

    print("[2/4] Чтение Markdown из:", MD_DIR)
    md_files = read_markdown_files(MD_DIR)
    if not md_files:
        raise SystemExit(f"В {MD_DIR} не найдено *.md")

    print("[3/4] Чанкинг → эмбеддинги")
    texts: List[str] = []
    metas: List[MetaRow] = []

    idx = 0
    for path, content in tqdm(md_files.items()):
        title = os.path.basename(path)
        chunks = chunk_by_tokens(content, tokenizer, CHUNK_TOKENS, OVERLAP)
        for start_idx, end_idx, chunk in chunks:
            texts.append(chunk)
            metas.append(
                MetaRow(
                    id=idx,
                    file=path,
                    title=title,
                    start_tok=start_idx,
                    end_tok=end_idx,
                    text=chunk,
                )
            )
            idx += 1

    print(f"Всего чанков: {len(texts)}")

    # Эмбеддинги батчами
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # [4/4] FAISS: cosine via inner product on normalized vectors
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in metas], f, ensure_ascii=False, indent=2)

    print("Готово →", os.path.join(OUT_DIR, "index.faiss"), ",", os.path.join(OUT_DIR, "meta.json"))


if __name__ == "__main__":
    build_index()
