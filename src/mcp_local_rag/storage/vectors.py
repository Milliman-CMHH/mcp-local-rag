from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import uuid

from mcp_local_rag.config import QDRANT_PATH, ensure_data_dir
from mcp_local_rag.processing.embeddings import get_embedding_dimension

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Condition,
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )

logger = logging.getLogger("mcp_local_rag.storage.vectors")


def _qdrant_client_cls() -> type[QdrantClient]:
    from qdrant_client import QdrantClient as _cls  # noqa: PLC0415
    return _cls


def _qdrant_models():  # type: ignore[return]
    from qdrant_client import models as _m  # noqa: PLC0415
    return _m


@dataclass
class CollectionStats:
    chunk_count: int


@dataclass
class SearchResult:
    text: str
    doc_id: str
    file_path: str
    collection: str
    chunk_index: int
    score: float


class VectorStore:
    COLLECTION_NAME = "chunks"

    def __init__(self, db_path: Path | None = None, url: str | None = None) -> None:
        ensure_data_dir()
        self._url = url
        self._db_path_arg = db_path
        self._mode: Literal["embedded", "client"]
        self.db_path: Path | None
        self._client: QdrantClient | None = None
        self._collection_ready = False

        if url:
            self._mode = "client"
            self.db_path = None
        else:
            self._mode = "embedded"
            self.db_path = db_path or QDRANT_PATH

    @property
    def client(self) -> QdrantClient:
        """Lazily connect to Qdrant on first use."""
        if self._client is None:
            _QC = _qdrant_client_cls()
            if self._url:
                try:
                    self._client = _QC(url=self._url)
                except Exception as exc:
                    raise ConnectionError(
                        f"Cannot reach Qdrant server at {self._url}.\n\n"
                        "Ensure Qdrant is running and accessible at the configured URL."
                    ) from exc
                logger.info("Qdrant client mode: connected to %s", self._url)
            else:
                try:
                    self._client = _QC(path=str(self.db_path))
                except RuntimeError as exc:
                    if "lock" in str(exc).lower():
                        raise RuntimeError(
                            "Qdrant storage is locked by another process.\n\n"
                            "To support multiple mcp-local-rag instances, run a "
                            "standalone Qdrant server\n"
                            "and set MCP_LOCAL_RAG_QDRANT_URL="
                            "http://127.0.0.1:6333"
                        ) from exc
                    raise
                logger.info("Qdrant embedded mode: %s", self.db_path)
        return self._client

    def _ensure_collection_once(self) -> None:
        """Ensure the Qdrant collection exists. Called lazily on first operation."""
        if self._collection_ready:
            return
        self._ensure_collection()
        self._collection_ready = True

    def _ensure_collection(self) -> None:
        m = _qdrant_models()
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        if self.COLLECTION_NAME not in collection_names:
            dim = get_embedding_dimension()
            if dim is None:
                raise RuntimeError("Failed to determine embedding dimension")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=m.VectorParams(size=dim, distance=m.Distance.COSINE),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        m = _qdrant_models()
        info = self.client.get_collection(self.COLLECTION_NAME)
        existing_indexes: set[str] = (
            set(info.payload_schema.keys()) if info.payload_schema else set()
        )

        if "collection" not in existing_indexes:
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="collection",
                field_schema=m.PayloadSchemaType.KEYWORD,
            )
        if "doc_id" not in existing_indexes:
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="doc_id",
                field_schema=m.PayloadSchemaType.KEYWORD,
            )

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: NDArray[np.float32],
        doc_id: str,
        file_path: str,
        collection: str,
    ) -> int:
        self._ensure_collection_once()
        if len(chunks) == 0:
            return 0

        m = _qdrant_models()
        points = [
            m.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "text": chunk,
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "collection": collection,
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
        return len(points)

    def delete_document_chunks(self, doc_id: str) -> int:
        # If the collection hasn't been created yet there's nothing to delete.
        if not self._collection_ready:
            return 0
        m = _qdrant_models()
        # Count before deletion
        count_result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=m.Filter(
                must=[m.FieldCondition(key="doc_id", match=m.MatchValue(value=doc_id))]
            ),
        )
        count_before = count_result.count

        if count_before > 0:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=m.Filter(
                    must=[m.FieldCondition(key="doc_id", match=m.MatchValue(value=doc_id))]
                ),
            )
        return count_before

    def delete_collection_chunks(self, collection: str) -> int:
        # If the collection hasn't been created yet there's nothing to delete.
        if not self._collection_ready:
            return 0
        m = _qdrant_models()
        # Count before deletion
        count_result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=m.Filter(
                must=[
                    m.FieldCondition(key="collection", match=m.MatchValue(value=collection))
                ]
            ),
        )
        count_before = count_result.count

        if count_before > 0:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=m.Filter(
                    must=[
                        m.FieldCondition(
                            key="collection", match=m.MatchValue(value=collection)
                        )
                    ]
                ),
            )
        return count_before

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 10,
        collection: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        self._ensure_collection_once()
        m = _qdrant_models()
        # Build filter conditions
        conditions: list[Condition] = []
        if collection:
            conditions.append(
                m.FieldCondition(key="collection", match=m.MatchValue(value=collection))
            )
        if doc_ids:
            doc_conditions: list[Condition] = [
                m.FieldCondition(key="doc_id", match=m.MatchValue(value=d)) for d in doc_ids
            ]
            doc_filter: Condition = m.Filter(should=doc_conditions)
            conditions.append(doc_filter)

        query_filter: Filter | None = None
        if conditions:
            query_filter = m.Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                text=str(point.payload["text"]) if point.payload else "",
                doc_id=str(point.payload["doc_id"]) if point.payload else "",
                file_path=str(point.payload["file_path"]) if point.payload else "",
                collection=str(point.payload["collection"]) if point.payload else "",
                chunk_index=int(point.payload["chunk_index"]) if point.payload else 0,
                score=point.score,
            )
            for point in results.points
        ]

    def get_collection_stats(self, collection: str) -> CollectionStats:
        # If the collection hasn't been created yet it has no chunks.
        if not self._collection_ready:
            return CollectionStats(chunk_count=0)
        m = _qdrant_models()
        count_result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=m.Filter(
                must=[
                    m.FieldCondition(key="collection", match=m.MatchValue(value=collection))
                ]
            ),
        )
        return CollectionStats(chunk_count=count_result.count)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
