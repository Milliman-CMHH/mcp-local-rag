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

    # ── Readiness gates ─────────────────────────────────────────────────
    # Stage 1: SQLite schema + API clients ready.
    # Lightweight tools (list/create/delete collections, list docs) gate here.
    _db_ready: asyncio.Event = field(default_factory=asyncio.Event)
    _db_error: BaseException | None = field(default=None)

    # Stage 2: embedding model loaded.
    # Search and indexing tools gate here.
    # Note: Qdrant has no startup gate — it is a remote service and should be
    # retried at each tool call rather than permanently failed at startup.
    _model_ready: asyncio.Event = field(default_factory=asyncio.Event)
    _model_error: BaseException | None = field(default=None)

    # ── Mark methods (called by _background_init only) ──────────────────

    def mark_db_ready(self, error: BaseException | None = None) -> None:
        """Signal that the DB stage finished (successfully or not)."""
        self._db_error = error
        self._db_ready.set()

    def mark_model_ready(self, error: BaseException | None = None) -> None:
        """Signal that the embedding model stage finished."""
        self._model_error = error
        self._model_ready.set()

    # ── Await methods (called by tool functions) ────────────────────────

    async def await_db_ready(self) -> None:
        """Wait until SQLite and API clients are initialized.

        Used by all tools — ensures metadata store is ready before any
        tool touches it.
        """
        await self._db_ready.wait()
        if self._db_error is not None:
            raise RuntimeError(
                "Server DB initialization failed — see server logs for details"
            ) from self._db_error

    async def await_model_ready(self) -> None:
        """Wait until the embedding model is loaded (implies DB is also ready).

        Used by tools that embed text: search and indexing.
        """
        await self.await_db_ready()
        await self._model_ready.wait()
        if self._model_error is not None:
            raise RuntimeError(
                "Embedding model failed to load — see server logs for details"
            ) from self._model_error


Ctx = Context[ServerSession, AppContext, CallToolRequest]


def get_app(ctx: Ctx) -> AppContext:
    return ctx.request_context.lifespan_context
