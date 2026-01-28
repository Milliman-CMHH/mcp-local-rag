from pathlib import Path

from pydantic import BaseModel

from mcp_local_rag.context import Ctx, get_app
from mcp_local_rag.tools.collections import CollectionNotFoundError


class DocumentNotFoundError(Exception):
    def __init__(self, file_path: str, collection: str) -> None:
        self.file_path = file_path
        self.collection = collection
        super().__init__(
            f"Document '{file_path}' not found in collection '{collection}'"
        )


class DocumentSummary(BaseModel):
    file_path: str
    file_type: str
    chunk_count: int


async def list_documents(collection: str, ctx: Ctx) -> list[DocumentSummary]:
    app = get_app(ctx)

    if not app.metadata_store.collection_exists(collection):
        raise CollectionNotFoundError(collection)

    docs = app.metadata_store.list_documents(collection)

    return [
        DocumentSummary(
            file_path=doc.file_path,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
        )
        for doc in docs
    ]


async def get_document_content(file_path: str, collection: str, ctx: Ctx) -> list[str]:
    app = get_app(ctx)
    abs_path = str(Path(file_path).expanduser().resolve())
    doc = app.metadata_store.get_document_by_path(abs_path, collection)

    if doc is None:
        raise DocumentNotFoundError(file_path, collection)

    chunks = app.vector_store.get_document_chunks(doc.doc_id)

    return [chunk.text for chunk in chunks]
