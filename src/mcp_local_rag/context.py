from dataclasses import dataclass

from google import genai

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp.types import CallToolRequest

from mcp_local_rag.storage.metadata import MetadataStore
from mcp_local_rag.storage.vectors import VectorStore


@dataclass
class AppContext:
    gemini_client: genai.Client | None
    metadata_store: MetadataStore
    vector_store: VectorStore


Ctx = Context[ServerSession, AppContext, CallToolRequest]


def get_app(ctx: Ctx) -> AppContext:
    return ctx.request_context.lifespan_context
