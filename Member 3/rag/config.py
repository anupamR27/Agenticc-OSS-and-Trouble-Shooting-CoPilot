"""Shared configuration for the RAG indexing and retrieval pipeline."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3
DEFAULT_SCORE_THRESHOLD = 0.35
SUPPORTED_EXTENSIONS = [".pdf"]