from pydantic import BaseModel

from mcp_local_rag.context import Ctx, get_app
from mcp_local_rag.processing import embed_texts
from mcp_local_rag.tools.collections import CollectionNotFoundError


class SearchResult(BaseModel):
    text: str
    file_path: str
    collection: str
    score: float


class SearchResults(BaseModel):
    results: list[SearchResult]


async def search(query: str, top_k: int, ctx: Ctx) -> SearchResults:
    app = get_app(ctx)
    query_embedding = embed_texts([query])[0]
    results = app.vector_store.search(query_embedding, top_k=top_k)

    return SearchResults(
        results=[
            SearchResult(
                text=r.text,
                file_path=r.file_path,
                collection=r.collection,
                score=r.score,
            )
            for r in results
        ]
    )


async def search_collection(
    query: str, collection: str, top_k: int, ctx: Ctx
) -> SearchResults:
    app = get_app(ctx)

    if not app.metadata_store.collection_exists(collection):
        raise CollectionNotFoundError(collection)

    query_embedding = embed_texts([query])[0]
    results = app.vector_store.search(
        query_embedding, collection=collection, top_k=top_k
    )

    return SearchResults(
        results=[
            SearchResult(
                text=r.text,
                file_path=r.file_path,
                collection=r.collection,
                score=r.score,
            )
            for r in results
        ]
    )
