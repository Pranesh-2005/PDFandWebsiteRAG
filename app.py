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

