import asyncio
import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import aiofiles
from google import genai
from google.genai import errors
from google.genai import types
from google.genai.types import MediaResolution
from markitdown import MarkItDown
from pydantic import BaseModel
import pymupdf  # type: ignore[import-untyped]
from pymupdf import Document  # pyright: ignore[reportMissingTypeStubs]
import pymupdf4llm  # type: ignore[import-untyped]
from pymupdf4llm.helpers.check_ocr import should_ocr_page  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from mcp_local_rag.config import SUPPORTED_EXTENSIONS


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
            await asyncio.sleep(retry_after)

    raise RuntimeError("Gemini OCR retries exhausted.")


async def extract_pdf(
    file_path: Path,
    gemini_client: genai.Client | None = None,
) -> ExtractedDocument:
    file_hash = compute_file_hash(file_path)

    doc = pymupdf.open(str(file_path))
    page_count: int = len(doc)
    md_pages: list[str | asyncio.Task[str]] = []
    semaphore = asyncio.Semaphore(16)

    for page_index, page in enumerate(doc.pages()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
        if gemini_client is not None and should_ocr_page(page):  # pyright: ignore[reportUnknownArgumentType]
            md_page = asyncio.create_task(
                _gemini_ocr_pdf_page_with_retry(
                    file_path=file_path,
                    page_index=page_index,
                    gemini_client=gemini_client,
                    semaphore=semaphore,
                )
            )
        else:
            md_page = pymupdf4llm.to_markdown(  # type: ignore[assignment]
                doc=str(file_path),
                use_ocr=False,
            )
        md_pages.append(md_page)  # pyright: ignore[reportArgumentType]

    doc.close()

    tasks = [item for item in md_pages if isinstance(item, asyncio.Task)]
    md_text: list[str] = []
    if tasks:
        task_results = await asyncio.gather(*tasks)
        task_iter = iter(task_results)
        for item in md_pages:
            if isinstance(item, asyncio.Task):
                md_text.append(next(task_iter))
            else:
                md_text.append(item)
    else:
        for item in md_pages:
            if isinstance(item, asyncio.Task):
                raise RuntimeError("Unexpected task while building markdown output.")
            md_text.append(item)

    return ExtractedDocument(
        file_path=str(file_path.resolve()),
        file_hash=file_hash,
        content="\n\n".join(md_text),
        file_type="pdf",
        page_count=page_count,
    )


async def extract_docx(file_path: Path) -> ExtractedDocument:
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
    gemini_client: genai.Client | None = None,
) -> ExtractedDocument:
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    file_type = SUPPORTED_EXTENSIONS[suffix]

    match file_type:
        case "pdf":
            return await extract_pdf(file_path, gemini_client=gemini_client)
        case "docx":
            return await extract_docx(file_path)
        case _:
            return await extract_plaintext(file_path)


def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
