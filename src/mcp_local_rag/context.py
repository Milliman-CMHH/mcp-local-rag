from __future__ import annotations

import asyncio
from dataclasses import dataclass
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

    async def await_model_ready(self) -> None:
        """Ensure the embedding model is loaded.

        Pre-warmed in a background task; if the warmup hasn't finished or
        previously failed, loads synchronously here.  lru_cache means the
        load only runs once on success; exceptions are not cached so transient
        failures are retried on the next call.
        """
        from mcp_local_rag.processing.embeddings import get_embedding_model  # noqa: PLC0415

        await asyncio.to_thread(get_embedding_model)


Ctx = Context[ServerSession, AppContext, CallToolRequest]


def get_app(ctx: Ctx) -> AppContext:
    return ctx.request_context.lifespan_context
