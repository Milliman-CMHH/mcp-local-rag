from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp.types import CallToolRequest

from mcp_local_rag.storage.metadata import MetadataStore
from mcp_local_rag.storage.vectors import VectorStore

if TYPE_CHECKING:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
    from google import genai as _genai


@dataclass
class AppContext:
    azure_di_client: DocumentIntelligenceClient | None
    gemini_client: _genai.Client | None
    gemini_semaphore: asyncio.Semaphore
    metadata_store: MetadataStore
    vector_store: VectorStore

    # ── DB readiness gate ────────────────────────────────────────────────
    # SQLite schema init runs synchronously before the lifespan yields, so
    # this event is set before any tool is ever called.  It exists only to
    # let tools defensively await it in case the ordering ever changes.
    _db_ready: asyncio.Event = field(default_factory=asyncio.Event)

    def mark_db_ready(self) -> None:
        """Signal that SQLite schema init completed successfully."""
        self._db_ready.set()

    async def await_db_ready(self) -> None:
        """Wait until the metadata store is ready.  Fast-path: already set."""
        await self._db_ready.wait()

    async def await_model_ready(self) -> None:
        """Ensure the embedding model is loaded before embedding-dependent tools run.

        The model is pre-warmed in a background task for fast first-call response.
        If the background warmup hasn't finished (or previously failed), this call
        loads the model synchronously in a thread.  lru_cache ensures the load
        only happens once on success; failed attempts are not cached so they retry.
        """
        await self.await_db_ready()
        from mcp_local_rag.processing.embeddings import get_embedding_model  # noqa: PLC0415
        await asyncio.to_thread(get_embedding_model)


Ctx = Context[ServerSession, AppContext, CallToolRequest]


def get_app(ctx: Ctx) -> AppContext:
    return ctx.request_context.lifespan_context
