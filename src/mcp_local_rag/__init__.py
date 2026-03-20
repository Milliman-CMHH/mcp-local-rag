from importlib.metadata import PackageNotFoundError, version

from mcp_local_rag.server import main, mcp

try:
    __version__ = version("mcp-local-rag")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["main", "mcp"]
