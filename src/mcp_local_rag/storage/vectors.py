from dataclasses import dataclass
from pathlib import Path
import uuid

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

from mcp_local_rag.config import MAX_CHUNKS_PER_DOC, QDRANT_PATH, ensure_data_dir
from mcp_local_rag.processing.embeddings import get_embedding_dimension


@dataclass
class DocumentChunk:
    text: str
    chunk_index: int


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

    def __init__(self, db_path: Path | None = None) -> None:
        ensure_data_dir()
        self.db_path = db_path or QDRANT_PATH
        self.client = QdrantClient(path=str(self.db_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        if self.COLLECTION_NAME not in collection_names:
            dim = get_embedding_dimension()
            if dim is None:
                raise RuntimeError("Failed to determine embedding dimension")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        info = self.client.get_collection(self.COLLECTION_NAME)
        existing_indexes: set[str] = (
            set(info.payload_schema.keys()) if info.payload_schema else set()
        )

        if "collection" not in existing_indexes:
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="collection",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        if "doc_id" not in existing_indexes:
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: NDArray[np.float32],
        doc_id: str,
        file_path: str,
        collection: str,
    ) -> int:
        if len(chunks) == 0:
            return 0

        points = [
            PointStruct(
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
        # Count before deletion
        count_result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        count_before = count_result.count

        if count_before > 0:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
        return count_before

    def delete_collection_chunks(self, collection: str) -> int:
        # Count before deletion
        count_result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=Filter(
                must=[
                    FieldCondition(key="collection", match=MatchValue(value=collection))
                ]
            ),
        )
        count_before = count_result.count

        if count_before > 0:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="collection", match=MatchValue(value=collection)
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
        # Build filter conditions
        conditions: list[Condition] = []
        if collection:
            conditions.append(
                FieldCondition(key="collection", match=MatchValue(value=collection))
            )
        if doc_ids:
            # For multiple doc_ids, we need to use should (OR) conditions
            doc_conditions: list[Condition] = [
                FieldCondition(key="doc_id", match=MatchValue(value=d)) for d in doc_ids
            ]
            # Wrap in a Filter with should for OR logic
            doc_filter: Condition = Filter(should=doc_conditions)
            conditions.append(doc_filter)

        # Construct the final filter
        query_filter: Filter | None = None
        if conditions:
            query_filter = Filter(must=conditions)

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
        count_result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=Filter(
                must=[
                    FieldCondition(key="collection", match=MatchValue(value=collection))
                ]
            ),
        )
        return CollectionStats(chunk_count=count_result.count)

    def get_document_chunks(self, doc_id: str) -> list[DocumentChunk]:
        # Scroll through all points matching doc_id
        results, _ = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=MAX_CHUNKS_PER_DOC,
            with_payload=True,
        )

        chunks = [
            DocumentChunk(
                text=str(point.payload["text"]) if point.payload else "",
                chunk_index=int(point.payload["chunk_index"]) if point.payload else 0,
            )
            for point in results
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)
