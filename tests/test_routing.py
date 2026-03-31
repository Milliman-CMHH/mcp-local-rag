from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_local_rag.config import SUPPORTED_EXTENSIONS
from mcp_local_rag.processing.extractors import (
    ExtractedDocument,
    extract_azure_di_document,
    extract_document,
    extract_image,
    extract_pdf,
    is_supported_file,
    provider_supports_file,
)

MODULE = "mcp_local_rag.processing.extractors"
DUMMY_HASH = "abc123"
DUMMY_CONTENT = "extracted content"


def _dummy_doc(suffix: str) -> ExtractedDocument:
    return ExtractedDocument(
        file_path=f"/fake/file{suffix}",
        file_hash=DUMMY_HASH,
        content=DUMMY_CONTENT,
        file_type=SUPPORTED_EXTENSIONS.get(suffix, "unknown"),
    )


class TestProviderSupportsFile:
    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".bmp", True),
            (".heic", False),
            (".heif", False),
            (".html", True),
            (".jfif", True),
            (".jpe", True),
            (".jpeg", True),
            (".jpg", True),
            (".pdf", True),
            (".png", True),
            (".pptx", True),
            (".tiff", True),
            (".webp", False),
            (".xlsx", True),
        ],
    )
    def test_azure(self, suffix: str, expected: bool) -> None:
        assert provider_supports_file("azure", suffix) is expected

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".bmp", False),
            (".heic", True),
            (".heif", True),
            (".html", False),
            (".jfif", True),
            (".jpe", True),
            (".jpeg", True),
            (".jpg", True),
            (".pdf", True),
            (".png", True),
            (".pptx", False),
            (".tiff", False),
            (".webp", True),
            (".xlsx", False),
        ],
    )
    def test_gemini(self, suffix: str, expected: bool) -> None:
        assert provider_supports_file("gemini", suffix) is expected

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".html", False),
            (".jpg", False),
            (".pdf", True),
            (".png", False),
            (".pptx", False),
            (".xlsx", False),
        ],
    )
    def test_pymupdf(self, suffix: str, expected: bool) -> None:
        assert provider_supports_file("pymupdf", suffix) is expected

    def test_unknown_provider(self) -> None:
        assert provider_supports_file("unknown", ".pdf") is False


class TestIsSupportedFile:
    @pytest.mark.parametrize(
        "suffix",
        [
            ".bmp",
            ".docx",
            ".heic",
            ".heif",
            ".html",
            ".jfi",
            ".jfif",
            ".jif",
            ".jpe",
            ".jpeg",
            ".jpg",
            ".markdown",
            ".md",
            ".pdf",
            ".png",
            ".pptx",
            ".rst",
            ".text",
            ".tif",
            ".tiff",
            ".txt",
            ".webp",
            ".xlsx",
        ],
    )
    def test_supported(self, suffix: str) -> None:
        assert is_supported_file(Path(f"test{suffix}")) is True

    @pytest.mark.parametrize(
        "suffix", [".gif", ".mp4", ".avi", ".unknown", ".doc", ".csv"]
    )
    def test_unsupported(self, suffix: str) -> None:
        assert is_supported_file(Path(f"test{suffix}")) is False


class TestExtractDocumentDispatch:
    @pytest.fixture()
    def metadata_store(self) -> MagicMock:
        store = MagicMock()
        store.get_file_metadata = MagicMock(return_value=None)
        return store

    @pytest.mark.parametrize(
        "suffix,extractor_name",
        [
            (".heic", "extract_image"),
            (".heif", "extract_image"),
            (".html", "extract_azure_di_document"),
            (".jfif", "extract_image"),
            (".jpe", "extract_image"),
            (".jpeg", "extract_image"),
            (".jpg", "extract_image"),
            (".pdf", "extract_pdf"),
            (".pptx", "extract_azure_di_document"),
            (".xlsx", "extract_azure_di_document"),
        ],
    )
    async def test_dispatches_to_correct_extractor(
        self,
        suffix: str,
        extractor_name: str,
        metadata_store: MagicMock,
    ) -> None:
        doc = _dummy_doc(suffix)
        with patch(
            f"{MODULE}.{extractor_name}",
            new_callable=AsyncMock,
            return_value=doc,
        ) as mock_fn:
            await extract_document(
                Path(f"/fake/file{suffix}"),
                metadata_store=metadata_store,
            )
            mock_fn.assert_called_once()

    async def test_docx_dispatch(self, metadata_store: MagicMock) -> None:
        with patch(
            f"{MODULE}.extract_docx",
            new_callable=AsyncMock,
            return_value=_dummy_doc(".docx"),
        ) as mock_fn:
            await extract_document(
                Path("/fake/file.docx"), metadata_store=metadata_store
            )
            mock_fn.assert_called_once()

    async def test_plaintext_dispatch(self, metadata_store: MagicMock) -> None:
        with patch(
            f"{MODULE}.extract_plaintext",
            new_callable=AsyncMock,
            return_value=_dummy_doc(".txt"),
        ) as mock_fn:
            await extract_document(
                Path("/fake/file.txt"), metadata_store=metadata_store
            )
            mock_fn.assert_called_once()

    async def test_unsupported_extension_raises(
        self, metadata_store: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported file type"):
            await extract_document(
                Path("/fake/file.xyz"), metadata_store=metadata_store
            )


class TestImageFallback:
    @pytest.fixture(autouse=True)
    def _mock_hash(self) -> Generator[None]:
        with patch(f"{MODULE}.compute_file_hash", return_value=DUMMY_HASH):
            yield

    @pytest.fixture()
    def gemini(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def azure(self) -> MagicMock:
        return MagicMock()

    async def test_auto_jpg_prefers_gemini(
        self, gemini: MagicMock, azure: MagicMock
    ) -> None:
        with patch(
            f"{MODULE}._gemini_extract_image",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ) as mock_gemini:
            result = await extract_image(
                Path("/fake/photo.jpg"),
                azure_di_client=azure,
                gemini_client=gemini,
            )
            mock_gemini.assert_called_once()
            assert result.file_type == "image"

    async def test_auto_jpg_falls_back_to_azure(self, azure: MagicMock) -> None:
        with patch(
            f"{MODULE}._azure_extract_document",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ) as mock_azure:
            result = await extract_image(
                Path("/fake/photo.jpg"),
                azure_di_client=azure,
                gemini_client=None,
            )
            mock_azure.assert_called_once()
            assert result.file_type == "image"

    async def test_auto_jpg_no_providers_raises(self) -> None:
        with pytest.raises(RuntimeError):
            await extract_image(
                Path("/fake/photo.jpg"),
                azure_di_client=None,
                gemini_client=None,
            )

    async def test_auto_heic_uses_gemini(self, gemini: MagicMock) -> None:
        with patch(
            f"{MODULE}._gemini_extract_image",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ):
            result = await extract_image(
                Path("/fake/photo.heic"),
                gemini_client=gemini,
            )
            assert result.file_type == "image"

    async def test_auto_heic_without_gemini_raises(self, azure: MagicMock) -> None:
        with pytest.raises(RuntimeError):
            await extract_image(
                Path("/fake/photo.heic"),
                azure_di_client=azure,
                gemini_client=None,
            )

    async def test_auto_webp_without_gemini_raises(self, azure: MagicMock) -> None:
        with pytest.raises(RuntimeError):
            await extract_image(
                Path("/fake/photo.webp"),
                azure_di_client=azure,
                gemini_client=None,
            )

    async def test_auto_bmp_skips_gemini_uses_azure(
        self, gemini: MagicMock, azure: MagicMock
    ) -> None:
        with patch(
            f"{MODULE}._azure_extract_document",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ) as mock_azure:
            result = await extract_image(
                Path("/fake/photo.bmp"),
                azure_di_client=azure,
                gemini_client=gemini,
            )
            mock_azure.assert_called_once()
            assert result.file_type == "image"

    async def test_auto_tiff_skips_gemini_uses_azure(
        self, gemini: MagicMock, azure: MagicMock
    ) -> None:
        with patch(
            f"{MODULE}._azure_extract_document",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ) as mock_azure:
            result = await extract_image(
                Path("/fake/photo.tiff"),
                azure_di_client=azure,
                gemini_client=gemini,
            )
            mock_azure.assert_called_once()
            assert result.file_type == "image"

    async def test_explicit_gemini(self, gemini: MagicMock) -> None:
        with patch(
            f"{MODULE}._gemini_extract_image",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ):
            result = await extract_image(
                Path("/fake/photo.jpg"),
                gemini_client=gemini,
                extraction_method="gemini",
            )
            assert result.file_type == "image"

    async def test_explicit_gemini_not_configured_raises(self) -> None:
        with pytest.raises(RuntimeError):
            await extract_image(
                Path("/fake/photo.jpg"),
                gemini_client=None,
                extraction_method="gemini",
            )

    async def test_explicit_azure_for_supported_ext(self, azure: MagicMock) -> None:
        with patch(
            f"{MODULE}._azure_extract_document",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ):
            result = await extract_image(
                Path("/fake/photo.jpg"),
                azure_di_client=azure,
                extraction_method="azure",
            )
            assert result.file_type == "image"

    async def test_explicit_azure_not_configured_raises(self) -> None:
        with pytest.raises(RuntimeError):
            await extract_image(
                Path("/fake/photo.jpg"),
                azure_di_client=None,
                extraction_method="azure",
            )

    async def test_explicit_azure_unsupported_ext_raises(
        self, azure: MagicMock
    ) -> None:
        with pytest.raises(RuntimeError):
            await extract_image(
                Path("/fake/photo.heic"),
                azure_di_client=azure,
                extraction_method="azure",
            )

    async def test_explicit_pymupdf_raises(self) -> None:
        with pytest.raises(RuntimeError):
            await extract_image(
                Path("/fake/photo.jpg"),
                extraction_method="pymupdf",
            )


class TestAzureDiDocumentFallback:
    @pytest.fixture(autouse=True)
    def _mock_hash(self) -> Generator[None]:
        with patch(f"{MODULE}.compute_file_hash", return_value=DUMMY_HASH):
            yield

    @pytest.fixture()
    def azure(self) -> MagicMock:
        return MagicMock()

    @pytest.mark.parametrize("suffix", [".html", ".pptx", ".xlsx"])
    async def test_auto_uses_azure_di(self, suffix: str, azure: MagicMock) -> None:
        with patch(
            f"{MODULE}._azure_extract_document",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ):
            result = await extract_azure_di_document(
                Path(f"/fake/file{suffix}"),
                azure_di_client=azure,
            )
            assert result.file_type == SUPPORTED_EXTENSIONS[suffix]

    @pytest.mark.parametrize("suffix", [".html", ".pptx", ".xlsx"])
    async def test_auto_without_azure_di_raises(self, suffix: str) -> None:
        with pytest.raises(RuntimeError):
            await extract_azure_di_document(
                Path(f"/fake/file{suffix}"),
                azure_di_client=None,
            )

    @pytest.mark.parametrize("suffix", [".html", ".pptx", ".xlsx"])
    async def test_explicit_gemini_raises(self, suffix: str) -> None:
        with pytest.raises(RuntimeError):
            await extract_azure_di_document(
                Path(f"/fake/file{suffix}"),
                extraction_method="gemini",
            )

    @pytest.mark.parametrize("suffix", [".html", ".pptx", ".xlsx"])
    async def test_explicit_pymupdf_raises(self, suffix: str) -> None:
        with pytest.raises(RuntimeError):
            await extract_azure_di_document(
                Path(f"/fake/file{suffix}"),
                extraction_method="pymupdf",
            )

    @pytest.mark.parametrize("suffix", [".html", ".pptx", ".xlsx"])
    async def test_explicit_azure_not_configured_raises(self, suffix: str) -> None:
        with pytest.raises(RuntimeError):
            await extract_azure_di_document(
                Path(f"/fake/file{suffix}"),
                azure_di_client=None,
                extraction_method="azure",
            )

    @pytest.mark.parametrize("suffix", [".html", ".pptx", ".xlsx"])
    async def test_explicit_azure_works(self, suffix: str, azure: MagicMock) -> None:
        with patch(
            f"{MODULE}._azure_extract_document",
            new_callable=AsyncMock,
            return_value=DUMMY_CONTENT,
        ):
            result = await extract_azure_di_document(
                Path(f"/fake/file{suffix}"),
                azure_di_client=azure,
                extraction_method="azure",
            )
            assert result.file_type == SUPPORTED_EXTENSIONS[suffix]


class TestPdfFallback:
    @pytest.fixture(autouse=True)
    def _mock_hash(self) -> Generator[None]:
        with patch(f"{MODULE}.compute_file_hash", return_value=DUMMY_HASH):
            yield

    @pytest.fixture()
    def metadata_store(self) -> MagicMock:
        store = MagicMock()
        store.get_file_metadata = MagicMock(return_value=None)
        store.get_cached_page = MagicMock(return_value=None)
        store.cache_page = MagicMock()
        store.clear_page_cache = MagicMock()
        return store

    @pytest.fixture()
    def gemini(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def azure(self) -> MagicMock:
        return MagicMock()

    async def test_auto_prefers_azure_di(
        self,
        metadata_store: MagicMock,
        azure: MagicMock,
        gemini: MagicMock,
    ) -> None:
        with (
            patch(
                f"{MODULE}._azure_extract_document",
                new_callable=AsyncMock,
                return_value=DUMMY_CONTENT,
            ) as mock_azure,
            patch(f"{MODULE}._get_pdf_page_count", return_value=1),
        ):
            result = await extract_pdf(
                Path("/fake/doc.pdf"),
                metadata_store=metadata_store,
                azure_di_client=azure,
                gemini_client=gemini,
            )
            mock_azure.assert_called_once()
            assert result.file_type == "pdf"

    async def test_auto_falls_back_to_pymupdf(self, metadata_store: MagicMock) -> None:
        with (
            patch(f"{MODULE}._get_pdf_page_count", return_value=1),
            patch(f"{MODULE}._pymupdf_to_markdown", return_value=DUMMY_CONTENT),
            patch(f"{MODULE}.pymupdf") as mock_pymupdf,
        ):
            mock_doc = MagicMock()
            mock_doc.pages.return_value = [MagicMock()]
            mock_pymupdf.open.return_value = mock_doc
            result = await extract_pdf(
                Path("/fake/doc.pdf"),
                metadata_store=metadata_store,
                azure_di_client=None,
                gemini_client=None,
            )
            assert result.file_type == "pdf"

    async def test_explicit_azure_not_configured_raises(
        self, metadata_store: MagicMock
    ) -> None:
        with patch(f"{MODULE}._get_pdf_page_count", return_value=1):
            with pytest.raises(RuntimeError):
                await extract_pdf(
                    Path("/fake/doc.pdf"),
                    metadata_store=metadata_store,
                    azure_di_client=None,
                    extraction_method="azure",
                )

    async def test_explicit_pymupdf(self, metadata_store: MagicMock) -> None:
        with (
            patch(f"{MODULE}._get_pdf_page_count", return_value=1),
            patch(
                f"{MODULE}._pymupdf_to_markdown", return_value=DUMMY_CONTENT
            ) as mock_pymupdf,
        ):
            result = await extract_pdf(
                Path("/fake/doc.pdf"),
                metadata_store=metadata_store,
                extraction_method="pymupdf",
            )
            mock_pymupdf.assert_called_once()
            assert result.file_type == "pdf"

    async def test_explicit_gemini_not_configured_raises(
        self, metadata_store: MagicMock
    ) -> None:
        with patch(f"{MODULE}._get_pdf_page_count", return_value=1):
            with pytest.raises(RuntimeError):
                await extract_pdf(
                    Path("/fake/doc.pdf"),
                    metadata_store=metadata_store,
                    gemini_client=None,
                    extraction_method="gemini",
                )
