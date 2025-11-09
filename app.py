# app.py
import os
import asyncio
import json
import hashlib
import shutil
from io import BytesIO, StringIO
from typing import List, Tuple

import gradio as gr
import numpy as np
import faiss
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import docx
from pptx import Presentation
from crawl4ai import AsyncWebCrawler

# ---------------- Config ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = "./cache"
SYSTEM_PROMPT = "You are a helpful assistant."
os.makedirs(CACHE_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

DOCS: List[str] = []
FILENAMES: List[str] = []
EMBEDDINGS: np.ndarray = None
FAISS_INDEX = None
CURRENT_CACHE_KEY: str = ""


# ---------------- Periodic cache cleanup ----------------
async def clear_cache_every_5min():
    while True:
        await asyncio.sleep(300)  # 5 minutes
        try:
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            print("🧹 Cache cleared successfully.")
        except Exception as e:
            print(f"[Cache cleanup error] {e}")

# Launch the cleaner in background
asyncio.get_event_loop().create_task(clear_cache_every_5min())

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        return f"[PDF extraction error] {e}"

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        f = BytesIO(file_bytes)
        doc = docx.Document(f)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[DOCX extraction error] {e}"

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[TXT extraction error] {e}"

def extract_text_from_excel(file_bytes: bytes) -> str:
    try:
        f = BytesIO(file_bytes)
        df = pd.read_excel(f, dtype=str)
        return "\n".join("\n".join(df[col].fillna("").astype(str).tolist()) for col in df.columns)
    except Exception as e:
        return f"[EXCEL extraction error] {e}"

def extract_text_from_pptx(file_bytes: bytes) -> str:
    try:
        f = BytesIO(file_bytes)
        prs = Presentation(f)
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    texts.append(shape.text)
        return "\n".join(texts)
    except Exception as e:
        return f"[PPTX extraction error] {e}"

def extract_text_from_csv(file_bytes: bytes) -> str:
    try:
        f = StringIO(file_bytes.decode("utf-8", errors="ignore"))
        df = pd.read_csv(f, dtype=str)
        return df.to_string(index=False)
    except Exception as e:
        return f"[CSV extraction error] {e}"

def extract_text_from_file_tuple(file_tuple) -> Tuple[str, bytes]:
    try:
        if hasattr(file_tuple, "name") and hasattr(file_tuple, "read"):
            return os.path.basename(file_tuple.name), file_tuple.read()
    except Exception:
        pass
    if isinstance(file_tuple, tuple) and len(file_tuple) == 2 and isinstance(file_tuple[1], (bytes, bytearray)):
        return file_tuple[0], bytes(file_tuple[1])
    if isinstance(file_tuple, str) and os.path.exists(file_tuple):
        with open(file_tuple, "rb") as fh:
            return os.path.basename(file_tuple), fh.read()
    raise ValueError("Unsupported file object passed by Gradio.")

def extract_text_by_ext(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"): return extract_text_from_pdf(file_bytes)
    if name.endswith(".docx"): return extract_text_from_docx(file_bytes)
    if name.endswith(".txt"): return extract_text_from_txt(file_bytes)
    if name.endswith((".xlsx", ".xls")): return extract_text_from_excel(file_bytes)
    if name.endswith(".pptx"): return extract_text_from_pptx(file_bytes)
    if name.endswith(".csv"): return extract_text_from_csv(file_bytes)
    return extract_text_from_txt(file_bytes)

