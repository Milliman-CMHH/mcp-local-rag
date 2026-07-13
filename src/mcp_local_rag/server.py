from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import contextlib
import logging
import os
from typing import TYPE_CHECKING, AsyncIterator

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


async def _init_app(app: AppContext) -> None:
    """Initialize API clients and SQLite schema.

    Runs synchronously before the lifespan yields so the MCP initialize
    handshake only completes once the metadata store is ready.  A failure
    here aborts startup cleanly rather than letting the server start in a
    broken state.
    """
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

    # SQLite schema — fast local I/O, must succeed before any tool runs
    await asyncio.to_thread(app.metadata_store._init_db)  # noqa: SLF001
    app.mark_db_ready()
    logger.info("DB initialized")


async def _background_warmup(app: AppContext) -> None:
    """Best-effort background warmup after the MCP handshake completes.

    Loads the embedding model into the lru_cache and fires Azure Monitor
    telemetry setup so they are ready before the first tool call arrives.
    Failures are logged but never propagate — tools retry these lazily.
    """
    async def _warmup_model() -> None:
        from mcp_local_rag.processing.embeddings import get_embedding_model  # noqa: PLC0415
        await asyncio.to_thread(get_embedding_model)
        logger.info("Embedding model warmed up")

    async def _warmup_model_safe() -> None:
        try:
            await _warmup_model()
        except Exception:
            logger.warning(
                "Background model warmup failed — will retry on first embedding call",
                exc_info=True,
            )

    async def _telemetry_safe() -> None:
        try:
            await configure_azure_monitor_async()
        except Exception:
            logger.warning("Azure Monitor telemetry setup failed", exc_info=True)

    await asyncio.gather(_warmup_model_safe(), _telemetry_safe())


@asynccontextmanager
async def app_lifespan(server: FastMCP[AppContext]) -> AsyncIterator[AppContext]:
    _ = server

    app = AppContext(
        azure_di_client=None,
        gemini_client=None,
        gemini_semaphore=asyncio.Semaphore(MAX_CONCURRENT_GEMINI),
        metadata_store=MetadataStore.create_uninitialized(),
        vector_store=VectorStore(url=QDRANT_URL),  # lazy — no I/O yet
    )

    # DB init runs synchronously: if it fails, initialize never completes
    # and VS Code surfaces a clean startup error rather than a broken server.
    await _init_app(app)

    # Kick off background warmup (model + telemetry) after yield so the
    # MCP handshake is not delayed.
    warmup_task = asyncio.create_task(
        _background_warmup(app), name="mcp-local-rag-warmup"
    )

    try:
        yield app
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(warmup_task)
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
