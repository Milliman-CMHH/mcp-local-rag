import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from mcp_local_rag.config import SQLITE_PATH, ensure_data_dir


@dataclass
class CollectionInfo:
    name: str
    created_at: datetime
    document_count: int


@dataclass
class DocumentInfo:
    doc_id: str
    file_path: str
    file_hash: str
    file_mtime: float
    file_type: str
    collection: str
    chunk_count: int
    markdown_path: str
    indexed_at: datetime


class MetadataStore:
    def __init__(self, db_path: Path | None = None) -> None:
        ensure_data_dir()
        self.db_path = db_path or SQLITE_PATH
        self._init_db()
        self._vacuum()

    def _vacuum(self) -> None:
        with self._get_connection() as conn:
            conn.execute("VACUUM")

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_mtime REAL NOT NULL DEFAULT 0,
                    file_type TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    markdown_path TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (collection) REFERENCES collections(name) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
                CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);
                CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

                CREATE TABLE IF NOT EXISTS page_cache (
                    file_hash TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (file_hash, page_index)
                );

                CREATE INDEX IF NOT EXISTS idx_page_cache_file_hash ON page_cache(file_hash);
                """
            )

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_collection(self, name: str) -> bool:
        with self._get_connection() as conn:
            try:
                conn.execute("INSERT INTO collections (name) VALUES (?)", (name,))
                return True
            except sqlite3.IntegrityError:
                return False

    def delete_collection(self, name: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def get_collection(self, name: str) -> CollectionInfo | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT c.name, c.created_at, COUNT(d.doc_id) as document_count
                FROM collections c
                LEFT JOIN documents d ON c.name = d.collection
                WHERE c.name = ?
                GROUP BY c.name
                """,
                (name,),
            ).fetchone()

            if row is None:
                return None

            return CollectionInfo(
                name=row["name"],
                created_at=row["created_at"],
                document_count=row["document_count"],
            )

    def list_collections(self) -> list[CollectionInfo]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.name, c.created_at, COUNT(d.doc_id) as document_count
                FROM collections c
                LEFT JOIN documents d ON c.name = d.collection
                GROUP BY c.name
                ORDER BY c.name
                """
            ).fetchall()

            return [
                CollectionInfo(
                    name=row["name"],
                    created_at=row["created_at"],
                    document_count=row["document_count"],
                )
                for row in rows
            ]

    def collection_exists(self, name: str) -> bool:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM collections WHERE name = ?", (name,)
            ).fetchone()
            return row is not None

    def add_document(
        self,
        doc_id: str,
        file_path: str,
        file_hash: str,
        file_mtime: float,
        file_type: str,
        collection: str,
        chunk_count: int,
        markdown_path: str,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, file_path, file_hash, file_mtime, file_type, collection, chunk_count, markdown_path, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    doc_id,
                    file_path,
                    file_hash,
                    file_mtime,
                    file_type,
                    collection,
                    chunk_count,
                    markdown_path,
                ),
            )

    def remove_document(self, doc_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()

            if row is None:
                return None

            return DocumentInfo(
                doc_id=row["doc_id"],
                file_path=row["file_path"],
                file_hash=row["file_hash"],
                file_mtime=row["file_mtime"] or 0.0,
                file_type=row["file_type"],
                collection=row["collection"],
                chunk_count=row["chunk_count"],
                markdown_path=row["markdown_path"],
                indexed_at=row["indexed_at"],
            )

    def get_document_by_path(
        self, file_path: str, collection: str
    ) -> DocumentInfo | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_path = ? AND collection = ?",
                (file_path, collection),
            ).fetchone()

            if row is None:
                return None

            return DocumentInfo(
                doc_id=row["doc_id"],
                file_path=row["file_path"],
                file_hash=row["file_hash"],
                file_mtime=row["file_mtime"] or 0.0,
                file_type=row["file_type"],
                collection=row["collection"],
                chunk_count=row["chunk_count"],
                markdown_path=row["markdown_path"],
                indexed_at=row["indexed_at"],
            )

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        with self._get_connection() as conn:
            if collection:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE collection = ? ORDER BY file_path",
                    (collection,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY collection, file_path"
                ).fetchall()

            return [
                DocumentInfo(
                    doc_id=row["doc_id"],
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    file_mtime=row["file_mtime"] or 0.0,
                    file_type=row["file_type"],
                    collection=row["collection"],
                    chunk_count=row["chunk_count"],
                    markdown_path=row["markdown_path"],
                    indexed_at=row["indexed_at"],
                )
                for row in rows
            ]

    def get_file_hash(self, file_path: str, collection: str) -> str | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT file_hash FROM documents WHERE file_path = ? AND collection = ?",
                (file_path, collection),
            ).fetchone()
            return row["file_hash"] if row else None

    def update_document_mtime(self, doc_id: str, file_mtime: float) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE documents SET file_mtime = ? WHERE doc_id = ?",
                (file_mtime, doc_id),
            )

    # ── Page cache ──────────────────────────────────────────────────────

    def get_cached_page(self, file_hash: str, page_index: int) -> str | None:
        """Load a cached page conversion result, or return None."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT content FROM page_cache WHERE file_hash = ? AND page_index = ?",
                (file_hash, page_index),
            ).fetchone()
            return row["content"] if row else None

    def cache_page(self, file_hash: str, page_index: int, content: str) -> None:
        """Store a page conversion result in the cache."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO page_cache (file_hash, page_index, content) VALUES (?, ?, ?)",
                (file_hash, page_index, content),
            )

    def clear_page_cache(self, file_hash: str) -> int:
        """Remove all cached pages for a given file hash. Returns count deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM page_cache WHERE file_hash = ?", (file_hash,)
            )
            return cursor.rowcount

    def clear_page_cache_for_collection(self, collection: str) -> int:
        """Remove cached pages for all documents in a collection."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM page_cache WHERE file_hash IN (
                    SELECT file_hash FROM documents WHERE collection = ?
                )
                """,
                (collection,),
            )
            return cursor.rowcount
