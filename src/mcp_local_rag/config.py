import os
import sys
from pathlib import Path


def _default_data_base() -> str:
    if sys.platform == "win32":
        return os.environ.get("LOCALAPPDATA", ".")
    if sys.platform == "darwin":
        return str(Path.home() / "Library" / "Application Support")
    return os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))


CHUNK_OVERLAP = int(os.environ.get("MCP_LOCAL_RAG_CHUNK_OVERLAP", "50"))
CHUNK_SIZE = int(os.environ.get("MCP_LOCAL_RAG_CHUNK_SIZE", "512"))
DATA_DIR = (
    Path(os.environ.get("MCP_LOCAL_RAG_DATA_DIR", _default_data_base()))
    / "mcp-local-rag"
)
EMBEDDING_MODEL = os.environ.get(
    "MCP_LOCAL_RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
GEMINI_MODEL = os.environ.get("MCP_LOCAL_RAG_GEMINI_MODEL", "gemini-3-pro-preview")
MAX_CONCURRENT_FILES = int(os.environ.get("MCP_LOCAL_RAG_MAX_CONCURRENT_FILES", "32"))
MAX_CONCURRENT_GEMINI = int(
    os.environ.get("MCP_LOCAL_RAG_MAX_CONCURRENT_GEMINI", "128")
)

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.environ.get(
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
)
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

MARKDOWN_DIR = DATA_DIR / "markdown"
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
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
}

AZURE_DI_SUPPORTED_IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
}

IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
}


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
