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
