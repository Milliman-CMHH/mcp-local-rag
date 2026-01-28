import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from mcp_local_rag.config import SUPPORTED_EXTENSIONS
from mcp_local_rag.context import AppContext, Ctx, get_app
from mcp_local_rag.processing import (
    chunk_text,
    compute_file_hash,
    embed_texts,
    extract_document,
    is_supported_file,
)


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


@dataclass
class IndexResult:
    success: bool
    message: str
    chunk_count: int | None = None


async def _index_single_file(
    app: AppContext,
    file_path: Path,
    collection: str,
    force: bool,
) -> IndexResult:
    if not file_path.exists():
        return IndexResult(False, "File not found")

    if not is_supported_file(file_path):
        return IndexResult(False, f"Unsupported file type: {file_path.suffix}")

    abs_path = str(file_path.resolve())

    if not force:
        existing = app.metadata_store.get_document_by_path(abs_path, collection)
        if existing:
            current_hash = compute_file_hash(file_path)
            if current_hash == existing.file_hash:
                return IndexResult(
                    True, f"Already indexed (unchanged): {file_path.name}"
                )

    try:
        doc = await extract_document(file_path)
    except Exception as e:
        return IndexResult(False, f"Extraction failed for {file_path.name}: {e}")

    chunks = chunk_text(doc.content)
    if not chunks:
        return IndexResult(False, f"No content extracted from: {file_path.name}")

    doc_id = f"{collection}_{uuid.uuid4().hex[:12]}"

    existing = app.metadata_store.get_document_by_path(abs_path, collection)
    if existing:
        app.vector_store.delete_document_chunks(existing.doc_id)
        app.metadata_store.remove_document(existing.doc_id)

    embeddings = embed_texts(chunks)

    app.vector_store.add_chunks(chunks, embeddings, doc_id, abs_path, collection)

    app.metadata_store.add_document(
        doc_id=doc_id,
        file_path=abs_path,
        file_hash=doc.file_hash,
        file_type=doc.file_type,
        collection=collection,
        chunk_count=len(chunks),
    )

    return IndexResult(True, f"Indexed: {file_path.name}", len(chunks))


async def index_files(
    file_paths: list[str],
    collection: str,
    ctx: Ctx,
    force: bool = False,
) -> list[FileIndexResult]:
    app = get_app(ctx)

    if not app.metadata_store.collection_exists(collection):
        app.metadata_store.create_collection(collection)

    results: list[FileIndexResult] = []

    for path_str in file_paths:
        path = Path(path_str).expanduser()
        result = await _index_single_file(app, path, collection, force)
        results.append(
            FileIndexResult(
                file_path=str(path),
                success=result.success,
                message=None if result.success else result.message,
            )
        )

    return results


async def index_directory(
    directory_path: str,
    collection: str,
    ctx: Ctx,
    glob_pattern: str = "*",
    recursive: bool = False,
    force: bool = False,
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

    results: list[FileIndexResult] = []

    for path in supported_files:
        result = await _index_single_file(app, path, collection, force)
        results.append(
            FileIndexResult(
                file_path=str(path),
                success=result.success,
                message=None if result.success else result.message,
            )
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
            app.metadata_store.remove_document(doc.doc_id)
            results.append(FileIndexResult(file_path=file_path, success=True))

    return results
