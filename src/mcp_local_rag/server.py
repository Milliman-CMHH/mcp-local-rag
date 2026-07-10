from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any, AsyncIterator

from mcp.server.fastmcp import FastMCP

from mcp_local_rag.config import (
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
    AZURE_DOCUMENT_INTELLIGENCE_KEY,
    MAX_CONCURRENT_GEMINI,
    QDRANT_URL,
)
from mcp_local_rag.context import AppContext
from mcp_local_rag.storage import MetadataStore, VectorStore
from mcp_local_rag.telemetry import configure_azure_monitor_async, configure_logging
from mcp_local_rag.tools import register_tools

if TYPE_CHECKING:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    from azure.identity.aio import DefaultAzureCredential

logger = logging.getLogger("mcp_local_rag.server")

# Retry policy for background init stages.
_INIT_MAX_ATTEMPTS = 3
_INIT_BACKOFF_BASE = 2.0  # seconds; doubled on each retry (2 → 4 → 8), though only
                           # two sleeps ever happen: after attempt 1 (2s) and attempt 2 (4s)


async def _retry(label: str, coro_fn: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run *coro_fn* up to _INIT_MAX_ATTEMPTS times with exponential backoff.

    Raises the last exception if all attempts fail.
    """
    delay = _INIT_BACKOFF_BASE
    last_exc: BaseException | None = None
    for attempt in range(1, _INIT_MAX_ATTEMPTS + 1):
        try:
            await coro_fn()
            return
        except Exception as exc:
            last_exc = exc
            if attempt < _INIT_MAX_ATTEMPTS:
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.0fs",
                    label,
                    attempt,
                    _INIT_MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    label,
                    _INIT_MAX_ATTEMPTS,
                    exc,
                    exc_info=True,
                )
    assert last_exc is not None
    raise last_exc


async def _init_db_stage(app: AppContext) -> None:
    """Stage: initialize SQLite schema and API clients."""
    # Gemini client (cheap object creation)
    if api_key := os.environ.get("GEMINI_API_KEY"):
        from google import genai  # noqa: PLC0415
        app.gemini_client = genai.Client(api_key=api_key)
    else:
        logger.warning("GEMINI_API_KEY not set — Gemini OCR functionality disabled")

    # Azure DI client
    if AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT:
        try:
            from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            if AZURE_DOCUMENT_INTELLIGENCE_KEY:
                _credential: AzureKeyCredential | DefaultAzureCredential = (
                    AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
                )
            else:
                from azure.identity.aio import DefaultAzureCredential

                _credential = DefaultAzureCredential()

            app.azure_di_client = DocumentIntelligenceClient(
                AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, _credential
            )
            logger.info("Azure Document Intelligence client initialized")
        except ImportError:
            logger.warning(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is set but "
                "azure-ai-documentintelligence is not installed — "
                "install with: uv add mcp-local-rag[azure]"
            )

    # SQLite schema (I/O, can transiently fail on locked file etc.)
    await asyncio.to_thread(app.metadata_store._init_db)  # noqa: SLF001
    logger.info("DB stage complete")


async def _init_vector_stage(app: AppContext) -> None:
    """Stage: verify Qdrant is reachable.

    Only does a lightweight connectivity check (GET /). Collection creation
    remains lazy — it happens on the first add_chunks() call so that read-only
    operations (search, list, get_info) work even before any documents are
    indexed, and so a transient startup failure here doesn't permanently break
    all vector-touching tools for the life of the process.
    """
    await asyncio.to_thread(app.vector_store._check_connection)  # noqa: SLF001
    logger.info("Vector store connected")


async def _init_model_stage() -> None:
    """Stage: load the embedding model into the LRU cache."""
    from mcp_local_rag.processing.embeddings import get_embedding_model

    await asyncio.to_thread(get_embedding_model)
    logger.info("Embedding model loaded")


async def _run_stage(
    label: str,
    coro_fn: Callable[[], Coroutine[Any, Any, None]],
    mark_done: Callable[[BaseException | None], None],
) -> None:
    """Run a single init stage with retry, then mark its readiness gate."""
    try:
        await _retry(label, coro_fn)
        mark_done(None)
    except Exception as exc:
        mark_done(exc)
        logger.error("%s permanently failed — related tools will error", label)


async def _telemetry_wrapper() -> None:
    """Run Azure Monitor telemetry setup; log failures but never propagate."""
    try:
        await configure_azure_monitor_async()
    except Exception:
        logger.warning("Azure Monitor telemetry setup failed", exc_info=True)


async def _background_init(app: AppContext) -> None:
    """Concurrent background init.  Runs as an asyncio Task.

    All stages run in parallel via asyncio.gather:
    - DB (SQLite + API clients) → app.mark_db_ready()
    - Vector (Qdrant connect + collection) → app.mark_vector_ready()
    - Model (embedding warmup) → app.mark_model_ready()
    - Telemetry (Azure Monitor) → fire-and-forget

    Tool functions gate on the appropriate await_*_ready() method.
    """
    await asyncio.gather(
        _run_stage("DB init", lambda: _init_db_stage(app), app.mark_db_ready),
        _run_stage("Vector store", lambda: _init_vector_stage(app), app.mark_vector_ready),
        _run_stage("Model warmup", _init_model_stage, app.mark_model_ready),
        _telemetry_wrapper(),
    )


@asynccontextmanager
async def app_lifespan(server: FastMCP[AppContext]) -> AsyncIterator[AppContext]:
    _ = server

    # Build a shell AppContext and yield it *immediately* so VS Code sees the
    # MCP server as started. Heavy work runs in the background task.
    app = AppContext(
        azure_di_client=None,
        gemini_client=None,
        gemini_semaphore=asyncio.Semaphore(MAX_CONCURRENT_GEMINI),
        metadata_store=MetadataStore.create_uninitialized(),  # no I/O yet
        vector_store=VectorStore(url=QDRANT_URL),             # lazy — no I/O yet
    )

    init_task = asyncio.create_task(_background_init(app), name="mcp-local-rag-init")

    try:
        yield app
    finally:
        # Wait for the background task before tearing down so we never close
        # resources mid-init. Suppress CancelledError so that a shutdown/reload
        # (which cancels this coroutine) still runs the cleanup below.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(init_task)
        app.vector_store.close()
        if app.azure_di_client is not None:
            await app.azure_di_client.close()


mcp = FastMCP[AppContext]("local-rag", lifespan=app_lifespan)
register_tools(mcp)


def main() -> None:
    configure_logging()
    mcp.run()


if __name__ == "__main__":
    main()
