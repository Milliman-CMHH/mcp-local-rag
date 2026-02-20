from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from typing import TYPE_CHECKING, AsyncIterator

from google import genai
from mcp.server.fastmcp import FastMCP

from mcp_local_rag.config import (
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
    AZURE_DOCUMENT_INTELLIGENCE_KEY,
    MAX_CONCURRENT_GEMINI,
)
from mcp_local_rag.context import AppContext
from mcp_local_rag.storage import MetadataStore, VectorStore
from mcp_local_rag.telemetry import configure_telemetry
from mcp_local_rag.tools import register_tools

if TYPE_CHECKING:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    from azure.identity.aio import DefaultAzureCredential

logger = logging.getLogger("mcp_local_rag.server")


@asynccontextmanager
async def app_lifespan(server: FastMCP[AppContext]) -> AsyncIterator[AppContext]:
    _ = server

    if api_key := os.environ.get("GEMINI_API_KEY"):
        gemini_client = genai.Client(api_key=api_key)
    else:
        logger.warning("GEMINI_API_KEY not set — Gemini OCR functionality disabled")
        gemini_client = None

    azure_di_client: DocumentIntelligenceClient | None = None
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

            azure_di_client = DocumentIntelligenceClient(
                AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, _credential
            )
            logger.info("Azure Document Intelligence client initialized")
        except ImportError:
            logger.warning(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is set but "
                "azure-ai-documentintelligence is not installed — "
                "install with: uv add mcp-local-rag[azure]"
            )

    app = AppContext(
        azure_di_client=azure_di_client,
        gemini_client=gemini_client,
        gemini_semaphore=asyncio.Semaphore(MAX_CONCURRENT_GEMINI),
        metadata_store=MetadataStore(),
        vector_store=VectorStore(),
    )
    try:
        yield app
    finally:
        app.vector_store.close()
        if azure_di_client is not None:
            await azure_di_client.close()


mcp = FastMCP[AppContext]("local-rag", lifespan=app_lifespan)
register_tools(mcp)


def main() -> None:
    configure_telemetry()
    mcp.run()


if __name__ == "__main__":
    main()
