from contextlib import asynccontextmanager
import os
import sys
from typing import AsyncIterator

from google import genai
from mcp.server.fastmcp import FastMCP

from mcp_local_rag.context import AppContext
from mcp_local_rag.storage import MetadataStore, VectorStore
from mcp_local_rag.tools import register_tools


@asynccontextmanager
async def app_lifespan(server: FastMCP[AppContext]) -> AsyncIterator[AppContext]:
    _ = server

    if api_key := os.environ.get("GEMINI_API_KEY"):
        gemini_client = genai.Client(api_key=api_key)
    else:
        print(
            "Warning: GEMINI_API_KEY not set. OCR functionality will be disabled.",
            file=sys.stderr,
        )
        gemini_client = None

    yield AppContext(
        gemini_client=gemini_client,
        metadata_store=MetadataStore(),
        vector_store=VectorStore(),
    )


mcp = FastMCP[AppContext]("local-rag", lifespan=app_lifespan)
register_tools(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
