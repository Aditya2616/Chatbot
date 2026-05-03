import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "gemini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "256"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
SNIPPET_LENGTH = int(os.getenv("SNIPPET_LENGTH", "200"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
