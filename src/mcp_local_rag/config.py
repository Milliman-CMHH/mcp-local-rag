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

QDRANT_URL: str | None = os.environ.get("MCP_LOCAL_RAG_QDRANT_URL")

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.environ.get(
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
)
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

MARKDOWN_DIR = DATA_DIR / "markdown"
QDRANT_PATH = DATA_DIR / "qdrant"
SQLITE_PATH = DATA_DIR / "metadata.db"
SUPPORTED_EXTENSIONS = {
    ".bmp": "image",
    ".docx": "docx",
    ".heic": "image",
    ".heif": "image",
    ".html": "html",
    ".jfi": "image",
    ".jfif": "image",
    ".jif": "image",
    ".jpe": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".markdown": "plaintext",
    ".md": "plaintext",
    ".pdf": "pdf",
    ".png": "image",
    ".pptx": "pptx",
    ".rst": "plaintext",
    ".text": "plaintext",
    ".tif": "image",
    ".tiff": "image",
    ".txt": "plaintext",
    ".webp": "image",
    ".xlsx": "xlsx",
}

AZURE_DI_SUPPORTED_EXTENSIONS: set[str] = {
    ".bmp",
    ".html",
    ".jfi",
    ".jfif",
    ".jif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pptx",
    ".tif",
    ".tiff",
    ".xlsx",
}

GEMINI_SUPPORTED_EXTENSIONS: set[str] = {
    ".heic",
    ".heif",
    ".jfi",
    ".jfif",
    ".jif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".webp",
}

PYMUPDF_SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
}

IMAGE_MIME_TYPES: dict[str, str] = {
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jfi": "image/jpeg",
    ".jfif": "image/jpeg",
    ".jif": "image/jpeg",
    ".jpe": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
