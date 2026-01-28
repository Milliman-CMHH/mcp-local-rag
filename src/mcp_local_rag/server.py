from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from mcp_local_rag.context import AppContext
from mcp_local_rag.storage import MetadataStore, VectorStore
from mcp_local_rag.tools import register_tools


@asynccontextmanager
async def app_lifespan(server: FastMCP[AppContext]) -> AsyncIterator[AppContext]:
    _ = server
    yield AppContext(
        metadata_store=MetadataStore(),
        vector_store=VectorStore(),
    )


mcp = FastMCP[AppContext]("local-rag", lifespan=app_lifespan)
register_tools(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
