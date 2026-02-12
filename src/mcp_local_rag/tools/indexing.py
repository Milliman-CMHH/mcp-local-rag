import asyncio
import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel

from mcp_local_rag.config import MAX_CONCURRENT_FILES, SUPPORTED_EXTENSIONS
from mcp_local_rag.context import AppContext, Ctx, get_app
from mcp_local_rag.processing import (
    chunk_text,
    compute_file_hash,
    embed_texts,
    extract_document,
    ExtractionMethod,
    get_file_mtime,
    is_supported_file,
)

logger = logging.getLogger("mcp_local_rag.tools.indexing")


def make_doc_id(file_path: str, collection: str) -> str:
    """Generate deterministic doc_id from file path and collection.

    This ensures that re-indexing the same file overwrites orphaned chunks
    if a previous indexing was interrupted.
    """
    key = f"{collection}\x00{file_path}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class DirectoryNotFoundError(Exception):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Directory not found: {path}")


class NotADirectoryError(Exception):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Not a directory: {path}")


class NoSupportedFilesError(Exception):
    def __init__(self, directory: str, extensions: list[str]) -> None:
        self.directory = directory
        self.extensions = extensions
        super().__init__(
            f"No supported files found. Supported extensions: {', '.join(extensions)}"
        )


class FileIndexResult(BaseModel):
    file_path: str
    success: bool
    message: str | None = None


async def _index_single_file(
    app: AppContext,
    file_path: Path,
    collection: str,
    force: bool,
    semaphore: asyncio.Semaphore,
    extraction_method: ExtractionMethod = "auto",
) -> FileIndexResult:
    async with semaphore:
        if not file_path.exists():
            return FileIndexResult(
                file_path=str(file_path), success=False, message="File not found"
            )

        if not is_supported_file(file_path):
            return FileIndexResult(
                file_path=str(file_path),
                success=False,
                message=f"Unsupported file type: {file_path.suffix}",
            )

        abs_path = str(file_path.resolve())

        doc_id = make_doc_id(abs_path, collection)
        current_mtime = get_file_mtime(file_path)

        if not force:
            existing = app.metadata_store.get_document_by_path(abs_path, collection)
            if existing:
                if existing.file_mtime == current_mtime:
                    logger.info("[%s] Skipped (unchanged)", file_path.name)
                    return FileIndexResult(file_path=str(file_path), success=True)
                # mtime changed, verify with hash
                current_hash = compute_file_hash(file_path)
                if current_hash == existing.file_hash:
                    # Content unchanged, just update mtime
                    app.metadata_store.update_document_mtime(
                        existing.doc_id, current_mtime
                    )
                    logger.info("[%s] Skipped (unchanged)", file_path.name)
                    return FileIndexResult(file_path=str(file_path), success=True)

        logger.info("[%s] Indexing into collection '%s'", file_path.name, collection)

        try:
            doc = await extract_document(
                file_path,
                metadata_store=app.metadata_store,
                gemini_client=app.gemini_client,
                gemini_semaphore=app.gemini_semaphore,
                force=force,
                extraction_method=extraction_method,
            )
        except Exception as e:
            logger.error("[%s] Extraction failed: %s", file_path.name, e)
            return FileIndexResult(
                file_path=str(file_path),
                success=False,
                message=f"Extraction failed for {file_path.name}: {e}",
            )

        try:
            chunks = chunk_text(doc.content)
            if not chunks:
                logger.warning("[%s] No content extracted", file_path.name)
                return FileIndexResult(
                    file_path=str(file_path),
                    success=False,
                    message=f"No content extracted from: {file_path.name}",
                )

            logger.info(
                "[%s] Chunked into %d chunks, embedding...",
                file_path.name,
                len(chunks),
            )

            # Remove existing chunks (if previous indexing was interrupted and retried)
            app.vector_store.delete_document_chunks(doc_id)

            embeddings = embed_texts(chunks)

            app.vector_store.add_chunks(
                chunks, embeddings, doc_id, abs_path, collection
            )

            app.metadata_store.add_document(
                doc_id=doc_id,
                file_path=abs_path,
                file_hash=doc.file_hash,
                file_mtime=current_mtime,
                file_type=doc.file_type,
                collection=collection,
                chunk_count=len(chunks),
            )
        except Exception as e:
            logger.error("[%s] Indexing failed: %s", file_path.name, e)
            return FileIndexResult(
                file_path=str(file_path),
                success=False,
                message=f"Indexing failed for {file_path.name}: {e}",
            )

        # Conversion + indexing succeeded — clear the page cache for this file
        app.metadata_store.clear_page_cache(doc.file_hash)

        logger.info("[%s] Indexed: %d chunks stored", file_path.name, len(chunks))
        return FileIndexResult(file_path=str(file_path), success=True)


async def index_files(
    file_paths: list[str],
    collection: str,
    ctx: Ctx,
    force: bool = False,
    extraction_method: ExtractionMethod = "auto",
) -> list[FileIndexResult]:
    app = get_app(ctx)

    if not app.metadata_store.collection_exists(collection):
        app.metadata_store.create_collection(collection)

    logger.info("Indexing %d file(s) into collection '%s'", len(file_paths), collection)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
    paths = [Path(p).expanduser() for p in file_paths]
    results = list(
        await asyncio.gather(
            *[
                _index_single_file(
                    app, p, collection, force, semaphore, extraction_method
                )
                for p in paths
            ]
        )
    )

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    logger.info("Indexing complete: %d succeeded, %d failed", succeeded, failed)

    return results


async def index_directory(
    directory_path: str,
    collection: str,
    ctx: Ctx,
    glob_pattern: str = "*",
    recursive: bool = False,
    force: bool = False,
    extraction_method: ExtractionMethod = "auto",
) -> list[FileIndexResult]:
    app = get_app(ctx)

    directory = Path(directory_path).expanduser()

    if not directory.exists():
        raise DirectoryNotFoundError(str(directory))

    if not directory.is_dir():
        raise NotADirectoryError(str(directory))

    if not app.metadata_store.collection_exists(collection):
        app.metadata_store.create_collection(collection)

    if recursive:
        files = list(directory.rglob(glob_pattern))
    else:
        files = list(directory.glob(glob_pattern))

    supported_files = [f for f in files if f.is_file() and is_supported_file(f)]

    if not supported_files:
        raise NoSupportedFilesError(str(directory), list(SUPPORTED_EXTENSIONS.keys()))

    logger.info(
        "Indexing directory %s: %d supported file(s) into collection '%s'",
        directory,
        len(supported_files),
        collection,
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
    results = list(
        await asyncio.gather(
            *[
                _index_single_file(
                    app, p, collection, force, semaphore, extraction_method
                )
                for p in supported_files
            ]
        )
    )

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    logger.info(
        "Directory indexing complete: %d succeeded, %d failed", succeeded, failed
    )

    return results


async def remove_documents(
    file_paths: list[str], collection: str, ctx: Ctx
) -> list[FileIndexResult]:
    app = get_app(ctx)
    results: list[FileIndexResult] = []

    for file_path in file_paths:
        abs_path = str(Path(file_path).expanduser().resolve())
        doc = app.metadata_store.get_document_by_path(abs_path, collection)

        if doc is None:
            results.append(
                FileIndexResult(
                    file_path=file_path, success=False, message="Document not found"
                )
            )
        else:
            app.vector_store.delete_document_chunks(doc.doc_id)
            app.metadata_store.clear_page_cache(doc.file_hash)
            app.metadata_store.remove_document(doc.doc_id)
            results.append(FileIndexResult(file_path=file_path, success=True))

    return results
