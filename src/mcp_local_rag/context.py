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
    # Each stage sets its event when done. Tool functions await the
    # appropriate gate before touching the corresponding resource.

    _db_ready: asyncio.Event = field(default_factory=asyncio.Event)
    _db_error: BaseException | None = field(default=None)

    _vector_ready: asyncio.Event = field(default_factory=asyncio.Event)
    _vector_error: BaseException | None = field(default=None)

    _model_ready: asyncio.Event = field(default_factory=asyncio.Event)
    _model_error: BaseException | None = field(default=None)

    # ── Mark methods (called by _background_init only) ──────────────────

    def mark_db_ready(self, error: BaseException | None = None) -> None:
        """Signal that the DB stage finished (successfully or not)."""
        self._db_error = error
        self._db_ready.set()

    def mark_vector_ready(self, error: BaseException | None = None) -> None:
        """Signal that the Qdrant connection stage finished."""
        self._vector_error = error
        self._vector_ready.set()

    def mark_model_ready(self, error: BaseException | None = None) -> None:
        """Signal that the embedding model stage finished."""
        self._model_error = error
        self._model_ready.set()

    # ── Await methods (called by tool functions) ────────────────────────

    async def await_db_ready(self) -> None:
        """Wait until SQLite and API clients are initialized.

        Used by tools that only touch the metadata store
        (collections, document listing, etc.).
        """
        await self._db_ready.wait()
        if self._db_error is not None:
            raise RuntimeError(
                "Server DB initialization failed — see server logs for details"
            ) from self._db_error

    async def await_vector_ready(self) -> None:
        """Wait until the Qdrant vector store is connected and collection exists.

        Implies DB is also ready (await_db_ready is called first).
        Used by tools that touch the vector store: delete, stats, search, index.
        """
        await self.await_db_ready()
        await self._vector_ready.wait()
        if self._vector_error is not None:
            raise RuntimeError(
                "Qdrant vector store connection failed — see server logs for details"
            ) from self._vector_error

    async def await_model_ready(self) -> None:
        """Wait until the embedding model is loaded.

        Implies vector store is also ready (which implies DB ready).
        Used by tools that embed text: search and indexing.
        """
        await self.await_vector_ready()
        await self._model_ready.wait()
        if self._model_error is not None:
            raise RuntimeError(
                "Embedding model failed to load — see server logs for details"
            ) from self._model_error


Ctx = Context[ServerSession, AppContext, CallToolRequest]


def get_app(ctx: Ctx) -> AppContext:
    return ctx.request_context.lifespan_context
