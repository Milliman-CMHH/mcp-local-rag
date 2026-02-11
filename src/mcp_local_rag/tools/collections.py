from datetime import datetime

from pydantic import BaseModel

from mcp_local_rag.context import Ctx, get_app


class CollectionNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Collection '{name}' not found")


class CollectionAlreadyExistsError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Collection '{name}' already exists")


class InvalidCollectionNameError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__("Collection name cannot be empty")


class DocumentSummary(BaseModel):
    file_path: str
    chunk_count: int


class CollectionInfo(BaseModel):
    name: str
    created_at: datetime
    document_count: int
    chunk_count: int
    documents: list[DocumentSummary]


async def create_collection(name: str, ctx: Ctx) -> None:
    app = get_app(ctx)

    if not name.strip():
        raise InvalidCollectionNameError(name)

    created = app.metadata_store.create_collection(name)
    if not created:
        raise CollectionAlreadyExistsError(name)


async def delete_collection(name: str, ctx: Ctx) -> None:
    app = get_app(ctx)

    if not app.metadata_store.collection_exists(name):
        raise CollectionNotFoundError(name)

    app.vector_store.delete_collection_chunks(name)
    app.metadata_store.clear_page_cache_for_collection(name)
    app.metadata_store.delete_collection(name)


async def list_collections(ctx: Ctx) -> list[str]:
    app = get_app(ctx)
    collections = app.metadata_store.list_collections()
    return [c.name for c in collections]


async def get_collection_info(name: str, ctx: Ctx) -> CollectionInfo:
    app = get_app(ctx)
    coll = app.metadata_store.get_collection(name)
    if coll is None:
        raise CollectionNotFoundError(name)

    stats = app.vector_store.get_collection_stats(name)
    docs = app.metadata_store.list_documents(name)

    return CollectionInfo(
        name=coll.name,
        created_at=coll.created_at,
        document_count=coll.document_count,
        chunk_count=stats.chunk_count,
        documents=[
            DocumentSummary(file_path=doc.file_path, chunk_count=doc.chunk_count)
            for doc in docs
        ],
    )
