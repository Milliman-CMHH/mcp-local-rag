import asyncio
from contextlib import asynccontextmanager
import logging
import os
from typing import AsyncIterator

from google import genai
from mcp.server.fastmcp import FastMCP

from mcp_local_rag.config import MAX_CONCURRENT_GEMINI
from mcp_local_rag.context import AppContext
from mcp_local_rag.storage import MetadataStore, VectorStore
from mcp_local_rag.telemetry import configure_telemetry
from mcp_local_rag.tools import register_tools

logger = logging.getLogger("mcp_local_rag.server")


@asynccontextmanager
async def app_lifespan(server: FastMCP[AppContext]) -> AsyncIterator[AppContext]:
    _ = server

    if api_key := os.environ.get("GEMINI_API_KEY"):
        gemini_client = genai.Client(api_key=api_key)
    else:
        logger.warning("GEMINI_API_KEY not set — OCR functionality disabled")
        gemini_client = None

    app = AppContext(
        gemini_client=gemini_client,
        gemini_semaphore=asyncio.Semaphore(MAX_CONCURRENT_GEMINI),
        metadata_store=MetadataStore(),
        vector_store=VectorStore(),
    )
    try:
        yield app
    finally:
        app.vector_store.close()


mcp = FastMCP[AppContext]("local-rag", lifespan=app_lifespan)
register_tools(mcp)


def main() -> None:
    configure_telemetry()
    mcp.run()


if __name__ == "__main__":
    main()
