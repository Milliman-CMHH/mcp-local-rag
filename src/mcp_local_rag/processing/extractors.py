from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import aiofiles
from google import genai
from google.genai import errors
from google.genai import types
from google.genai.types import MediaResolution
from markitdown import MarkItDown
from pydantic import BaseModel
import pymupdf  # type: ignore[import-untyped]
from pymupdf import Document  # pyright: ignore[reportMissingTypeStubs]
import pymupdf.layout  # pyright: ignore[reportUnusedImport]
import pymupdf4llm  # type: ignore[import-untyped]
from pymupdf4llm.helpers.check_ocr import should_ocr_page  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from mcp_local_rag.config import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeResult,
        DocumentTable,
        DocumentTableCell,
    )

    from mcp_local_rag.storage.metadata import MetadataStore

logger = logging.getLogger("mcp_local_rag.processing.extractors")

ExtractionMethod = Literal["auto", "azure", "gemini", "pymupdf"]


@dataclass
class ExtractedDocument:
    file_path: str
    file_hash: str
    content: str
    file_type: str
    page_count: int | None = None


class GeminiMarkdownResponse(BaseModel):
    markdown: str


def compute_file_hash(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_file_mtime(file_path: Path) -> float:
    return file_path.stat().st_mtime


def _convert_pdf_page_to_bytes(
    file_path: Path,
    page_index: int,
) -> bytes:
    doc = pymupdf.open(str(file_path))
    single_page_doc: Document = pymupdf.open()
    try:
        single_page_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)  # pyright: ignore[reportUnknownMemberType]
        pdf_bytes = single_page_doc.tobytes()  # pyright: ignore[reportUnknownMemberType]
    finally:
        single_page_doc.close()
        doc.close()

    return pdf_bytes


async def _gemini_ocr_pdf_page(
    file_path: Path,
    page_index: int,
    gemini_client: genai.Client,
    media_resolution: MediaResolution = MediaResolution.MEDIA_RESOLUTION_MEDIUM,
) -> str:
    pdf_bytes = _convert_pdf_page_to_bytes(
        file_path=file_path,
        page_index=page_index,
    )

    response = await gemini_client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
        model="gemini-3-pro-preview",
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(
                text=(
                    "Convert this PDF page to Markdown. "
                    "Preserve headings, lists, tables, and formatting. "
                    "Return only the Markdown content."
                )
            ),
        ],
        config=types.GenerateContentConfig(media_resolution=media_resolution),
    )

    response_text = response.text
    if response_text is None:
        raise RuntimeError("Gemini OCR response contained no text.")

    return response_text


def _get_retry_after_seconds(error: errors.ClientError) -> float:
    response = getattr(error, "response", None)
    if response is None:
        raise error

    headers = getattr(response, "headers", None)
    if headers is None:
        raise error

    retry_after = headers.get("Retry-After")
    if retry_after is None:
        raise error

    retry_after = retry_after.strip()
    if retry_after.isdigit():
        return float(retry_after)

    retry_at = cast(datetime, parsedate_to_datetime(retry_after))
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


async def _gemini_ocr_pdf_page_with_retry(
    file_path: Path,
    page_index: int,
    gemini_client: genai.Client,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                return await _gemini_ocr_pdf_page(
                    file_path=file_path,
                    page_index=page_index,
                    gemini_client=gemini_client,
                )
        except errors.ClientError as err:
            if err.code != 429 or attempt >= max_retries:
                raise

            retry_after = _get_retry_after_seconds(err)
            logger.warning(
                "[%s] page %d: Gemini 429 rate-limited, Retry-After=%.1fs — sleeping (attempt %d/%d)",
                file_path.name,
                page_index + 1,
                retry_after,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(retry_after)

    raise RuntimeError("Gemini OCR retries exhausted.")


async def _convert_and_cache_gemini_page(
    file_path: Path,
    page_index: int,
    file_hash: str,
    gemini_client: genai.Client,
    semaphore: asyncio.Semaphore,
    metadata_store: MetadataStore,
) -> str:
    """Run Gemini OCR for a single page and cache the result on success."""
    result = await _gemini_ocr_pdf_page_with_retry(
        file_path=file_path,
        page_index=page_index,
        gemini_client=gemini_client,
        semaphore=semaphore,
    )
    metadata_store.cache_page(file_hash, page_index, result)
    return result


def _needs_html_table(table: DocumentTable) -> bool:
    """Return True if any cell spans multiple rows or columns."""
    for cell in table.cells:
        if (cell.row_span or 1) > 1 or (cell.column_span or 1) > 1:
            return True
    return False


def _build_html_table(table: DocumentTable) -> str:
    """Render the table as an HTML table to support colspan/rowspan."""
    # Track which (row, col) positions are already occupied by a spanning cell.
    occupied: set[tuple[int, int]] = set()

    rows_html: list[str] = []
    header_rows: set[int] = set()
    for cell in table.cells:
        kind = cell.kind or "content"
        if kind in ("columnHeader", "rowHeader", "stubHead"):
            header_rows.add(cell.row_index)

    # Group cells by row
    by_row: dict[int, list[DocumentTableCell]] = defaultdict(list)
    for cell in table.cells:
        by_row[cell.row_index].append(cell)

    for r_idx in range(table.row_count):
        cells_in_row = sorted(by_row.get(r_idx, []), key=lambda c: c.column_index)
        row_parts: list[str] = []
        for cell in cells_in_row:
            pos = (cell.row_index, cell.column_index)
            if pos in occupied:
                continue
            rs = cell.row_span or 1
            cs = cell.column_span or 1
            # Mark all spanned positions as occupied
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((cell.row_index + dr, cell.column_index + dc))
            tag = "th" if r_idx in header_rows else "td"
            attrs = ""
            if rs > 1:
                attrs += f' rowspan="{rs}"'
            if cs > 1:
                attrs += f' colspan="{cs}"'
            content = cell.content.replace("\n", " ").strip()
            row_parts.append(f"<{tag}{attrs}>{content}</{tag}>")
        rows_html.append("  <tr>" + "".join(row_parts) + "</tr>")

    lines: list[str] = []
    if table.caption:
        lines.append(f"**{table.caption.content.strip()}**")
        lines.append("")
    lines.append("<table>")
    lines.extend(rows_html)
    lines.append("</table>")
    if table.footnotes:
        for fn in table.footnotes:
            lines.append("")
            lines.append(f"_{fn.content.strip()}_")
    return "\n".join(lines)


def _build_markdown_table(table: DocumentTable) -> str:
    """Reconstruct a table from structured Azure DI cell data.

    Uses HTML when the table contains spanning cells (colspan/rowspan), since
    standard Markdown tables don't support them. Falls back to a plain
    Markdown table otherwise.
    """
    if _needs_html_table(table):
        return _build_html_table(table)

    row_count: int = table.row_count
    col_count: int = table.column_count

    grid: list[list[str]] = [["" for _ in range(col_count)] for _ in range(row_count)]
    header_rows: set[int] = set()

    for cell in table.cells:
        r, c = cell.row_index, cell.column_index
        grid[r][c] = cell.content.replace("\n", " ").strip()
        kind = cell.kind or "content"
        if kind in ("columnHeader", "rowHeader", "stubHead"):
            header_rows.add(r)

    separator_after = max(header_rows) if header_rows else 0

    lines: list[str] = []
    if table.caption:
        lines.append(f"**{table.caption.content.strip()}**")
        lines.append("")
    for r_idx, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if r_idx == separator_after:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
    if table.footnotes:
        for fn in table.footnotes:
            lines.append("")
            lines.append(f"_{fn.content.strip()}_")
    return "\n".join(lines)


def _rebuild_content_tables(result: AnalyzeResult) -> str:
    """Replace each table block in ``result.content`` with a reconstructed
    Markdown table built from the structured ``result.tables`` cell data.

    Replacements are applied right-to-left so earlier span offsets remain
    valid throughout the substitution loop.
    """
    content: str = result.content
    if not result.tables:
        return content

    replacements: list[tuple[int, int, str]] = []
    for table in result.tables:
        if not table.spans:
            continue
        span = table.spans[0]
        replacements.append(
            (span.offset, span.offset + span.length, _build_markdown_table(table))
        )

    for start, end, md in sorted(replacements, key=lambda x: x[0], reverse=True):
        content = content[:start] + md + content[end:]

    return content


async def _azure_extract_pdf(
    file_path: Path,
    azure_di_client: DocumentIntelligenceClient,
) -> str:
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        DocumentContentFormat,
    )

    async with aiofiles.open(file_path, "rb") as f:
        pdf_bytes = await f.read()

    poller = await azure_di_client.begin_analyze_document(
        "prebuilt-layout",
        body=AnalyzeDocumentRequest(
            bytes_source=pdf_bytes,
        ),
        output_content_format=DocumentContentFormat.MARKDOWN,
    )
    await poller.wait()
    result = await poller.result()

    return _rebuild_content_tables(result)


async def extract_pdf(
    file_path: Path,
    metadata_store: MetadataStore,
    gemini_client: genai.Client | None = None,
    gemini_semaphore: asyncio.Semaphore | None = None,
    azure_di_client: DocumentIntelligenceClient | None = None,
    force: bool = False,
    extraction_method: ExtractionMethod = "auto",
) -> ExtractedDocument:
    file_hash = compute_file_hash(file_path)

    if force:
        metadata_store.clear_page_cache(file_hash)

    doc = pymupdf.open(str(file_path))
    page_count: int = len(doc)
    doc.close()

    logger.info(
        "[%s] Extracting PDF (%d pages, method=%s)",
        file_path.name,
        page_count,
        extraction_method,
    )

    # Fast path: Azure Document Intelligence processes the whole document at once.
    # Also used by 'auto' when an Azure client is available (preferred over Gemini).
    if extraction_method == "azure" or (
        extraction_method == "auto" and azure_di_client is not None
    ):
        if azure_di_client is None:
            raise RuntimeError(
                "extraction_method='azure' requires AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "to be configured and azure-ai-documentintelligence to be installed."
            )
        logger.info("[%s] Extracting with Azure Document Intelligence", file_path.name)
        content: str = await _azure_extract_pdf(file_path, azure_di_client)
        logger.info(
            "[%s] Extraction complete: %d pages (Azure Document Intelligence)",
            file_path.name,
            page_count,
        )
        return ExtractedDocument(
            file_path=str(file_path.resolve()),
            file_hash=file_hash,
            content=content,
            file_type="pdf",
            page_count=page_count,
        )

    # Fast path: pymupdf-only needs no per-page iteration
    if extraction_method == "pymupdf":
        logger.info("[%s] Converting entire document with pymupdf", file_path.name)
        content: str = pymupdf4llm.to_markdown(  # type: ignore[assignment]
            doc=str(file_path),
            use_ocr=False,
        )
        logger.info(
            "[%s] Extraction complete: %d pages (all pymupdf)",
            file_path.name,
            page_count,
        )
        return ExtractedDocument(
            file_path=str(file_path.resolve()),
            file_hash=file_hash,
            content=content,
            file_type="pdf",
            page_count=page_count,
        )

    semaphore = gemini_semaphore or asyncio.Semaphore(16)

    gemini_page_count = 0
    pymupdf_page_count = 0
    cached_page_count = 0
    md_pages: list[str | asyncio.Task[str]] = []

    doc = pymupdf.open(str(file_path))
    for page_index, page in enumerate(doc.pages()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
        cached = metadata_store.get_cached_page(file_hash, page_index)
        if cached is not None:
            logger.info(
                "[%s]   [%d/%d] cached",
                file_path.name,
                page_index + 1,
                page_count,
            )
            md_pages.append(cached)
            cached_page_count += 1
            continue

        use_gemini = False
        if extraction_method == "gemini":
            use_gemini = gemini_client is not None
        elif extraction_method == "auto":
            use_gemini = (
                gemini_client is not None and should_ocr_page(page)["should_ocr"]  # pyright: ignore[reportUnknownArgumentType]
            )
        # extraction_method == "pymupdf" → use_gemini stays False

        if use_gemini:
            assert gemini_client is not None
            logger.info(
                "[%s]   [%d/%d] Gemini OCR",
                file_path.name,
                page_index + 1,
                page_count,
            )
            task = asyncio.create_task(
                _convert_and_cache_gemini_page(
                    file_path=file_path,
                    page_index=page_index,
                    file_hash=file_hash,
                    gemini_client=gemini_client,
                    semaphore=semaphore,
                    metadata_store=metadata_store,
                )
            )
            md_pages.append(task)  # pyright: ignore[reportArgumentType]
            gemini_page_count += 1
        else:
            logger.info(
                "[%s]   [%d/%d] pymupdf",
                file_path.name,
                page_index + 1,
                page_count,
            )
            page_md: str = pymupdf4llm.to_markdown(  # type: ignore[assignment]
                doc=str(file_path),
                pages=[page_index],
                use_ocr=False,
            )
            metadata_store.cache_page(file_hash, page_index, page_md)
            md_pages.append(page_md)
            pymupdf_page_count += 1

    doc.close()

    tasks = [item for item in md_pages if isinstance(item, asyncio.Task)]
    md_text: list[str] = []
    if tasks:
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        task_iter = iter(task_results)
        failed_pages: list[int] = []
        for page_idx, item in enumerate(md_pages):
            if isinstance(item, asyncio.Task):
                result = next(task_iter)
                if isinstance(result, BaseException):
                    failed_pages.append(page_idx + 1)
                    md_text.append("")
                else:
                    md_text.append(result)
            else:
                md_text.append(item)
        if failed_pages:
            page_list = ", ".join(str(p) for p in failed_pages)
            logger.error(
                "[%s] Gemini OCR failed for %d page(s): %s",
                file_path.name,
                len(failed_pages),
                page_list,
            )
            raise RuntimeError(
                f"Gemini OCR failed for {len(failed_pages)} page(s) of "
                f"{file_path.name}: pages {page_list}"
            )
    else:
        for item in md_pages:
            if isinstance(item, asyncio.Task):
                raise RuntimeError("Unexpected task while building markdown output.")
            md_text.append(item)

    logger.info(
        "[%s] Extraction complete: %d pages (%d cached, %d Gemini OCR, %d pymupdf)",
        file_path.name,
        page_count,
        cached_page_count,
        gemini_page_count,
        pymupdf_page_count,
    )

    return ExtractedDocument(
        file_path=str(file_path.resolve()),
        file_hash=file_hash,
        content="\n\n".join(md_text),
        file_type="pdf",
        page_count=page_count,
    )


async def extract_docx(file_path: Path) -> ExtractedDocument:
    logger.info("[%s] Extracting DOCX (markitdown)", file_path.name)
    file_hash = compute_file_hash(file_path)
    converter = MarkItDown()
    result = converter.convert(source=file_path)

    return ExtractedDocument(
        file_path=str(file_path.resolve()),
        file_hash=file_hash,
        content=result.text_content,
        file_type="docx",
    )


async def extract_plaintext(file_path: Path) -> ExtractedDocument:
    logger.info("[%s] Extracting plaintext", file_path.name)
    file_hash = compute_file_hash(file_path)

    async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = await f.read()

    return ExtractedDocument(
        file_path=str(file_path.resolve()),
        file_hash=file_hash,
        content=content,
        file_type="plaintext",
    )


async def extract_document(
    file_path: Path,
    metadata_store: MetadataStore,
    gemini_client: genai.Client | None = None,
    gemini_semaphore: asyncio.Semaphore | None = None,
    azure_di_client: DocumentIntelligenceClient | None = None,
    force: bool = False,
    extraction_method: ExtractionMethod = "auto",
) -> ExtractedDocument:
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    file_type = SUPPORTED_EXTENSIONS[suffix]

    match file_type:
        case "pdf":
            return await extract_pdf(
                file_path,
                metadata_store=metadata_store,
                gemini_client=gemini_client,
                gemini_semaphore=gemini_semaphore,
                azure_di_client=azure_di_client,
                force=force,
                extraction_method=extraction_method,
            )
        case "docx":
            return await extract_docx(file_path)
        case _:
            return await extract_plaintext(file_path)


def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
