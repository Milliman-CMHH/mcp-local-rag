from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcp-local-rag")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


def main() -> None:
    # Deferred import: keeps package-level import cost near-zero.
    # The entry point (mcp-local-rag = "mcp_local_rag:main") calls this directly.
    from mcp_local_rag.server import main as _main  # noqa: PLC0415

    _main()
