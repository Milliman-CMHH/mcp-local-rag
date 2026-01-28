import pymupdf  # type: ignore[import-untyped]

# Minimum text length threshold to consider a PDF as having extractable text
MIN_TEXT_LENGTH_THRESHOLD = 50

# Minimum ratio of text chars to page count to consider PDF as having text
MIN_TEXT_PER_PAGE_RATIO = 10


def ocr_pdf_pages(doc: pymupdf.Document, dpi: int = 300, language: str = "eng") -> str:
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page: pymupdf.Page = doc[page_num]
        tp = page.get_textpage_ocr(dpi=dpi, language=language, full=True)  # pyright: ignore[reportUnknownMemberType]
        page_text = str(page.get_text(textpage=tp))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        if page_text.strip():
            pages_text.append(f"## Page {page_num + 1}\n\n{page_text}")

    return "\n\n".join(pages_text)


def needs_ocr(extracted_text: str, page_count: int) -> bool:
    text_length = len(extracted_text.strip())

    if text_length < MIN_TEXT_LENGTH_THRESHOLD:
        return True

    if page_count > 0 and text_length / page_count < MIN_TEXT_PER_PAGE_RATIO:
        return True

    return False
