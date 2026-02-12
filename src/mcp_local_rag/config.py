import os
from pathlib import Path


CHUNK_OVERLAP = int(os.environ.get("MCP_LOCAL_RAG_CHUNK_OVERLAP", "50"))
CHUNK_SIZE = int(os.environ.get("MCP_LOCAL_RAG_CHUNK_SIZE", "512"))
DATA_DIR = (
    Path(os.environ.get("MCP_LOCAL_RAG_DATA_DIR", os.environ.get("LOCALAPPDATA", ".")))
    / "mcp-local-rag"
)
EMBEDDING_MODEL = os.environ.get(
    "MCP_LOCAL_RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
MAX_CHUNKS_PER_DOC = int(os.environ.get("MCP_LOCAL_RAG_MAX_CHUNKS_PER_DOC", "10000"))
MAX_CONCURRENT_FILES = int(os.environ.get("MCP_LOCAL_RAG_MAX_CONCURRENT_FILES", "32"))
MAX_CONCURRENT_GEMINI = int(
    os.environ.get("MCP_LOCAL_RAG_MAX_CONCURRENT_GEMINI", "128")
)

QDRANT_PATH = DATA_DIR / "qdrant"
SQLITE_PATH = DATA_DIR / "metadata.db"
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "plaintext",
    ".md": "plaintext",
    ".markdown": "plaintext",
    ".rst": "plaintext",
    ".text": "plaintext",
}


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
