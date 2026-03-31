from mcp.server.fastmcp import FastMCP

from mcp_local_rag.context import AppContext
from mcp_local_rag.tools.collections import (
    create_collection,
    delete_collection,
    get_collection_info,
    list_collections,
)
from mcp_local_rag.tools.documents import get_document_content, list_documents
from mcp_local_rag.tools.indexing import index_directory, index_files, remove_documents
from mcp_local_rag.tools.search import search, search_collection


def register_tools(mcp: FastMCP[AppContext]) -> None:
    mcp.add_tool(
        fn=create_collection,
        description="Create a new collection for organizing documents. "
        "Collections allow you to scope searches to specific sets of documents.",
    )
    mcp.add_tool(
        fn=delete_collection,
        description="Delete a collection and all its indexed documents. This action cannot be undone.",
    )
    mcp.add_tool(
        fn=list_collections,
        description="List all collections by name.",
    )
    mcp.add_tool(
        fn=get_collection_info,
        description="Get detailed information about a specific collection including its documents.",
    )
    mcp.add_tool(
        fn=index_files,
        description="Index one or more files into a collection. "
        "Supports PDF, DOCX, HTML, PPTX, XLSX, plaintext, and image files (JPG, PNG, BMP, TIFF, WebP, HEIC/HEIF). "
        "Files are automatically chunked, embedded, and stored for semantic search. "
        "Use force=True to re-index files even if unchanged. "
        "extraction_method controls conversion quality: "
        "'auto' (default) uses Azure Document Intelligence if configured (best quality, private), "
        "otherwise Gemini AI for scanned/OCR pages and PyMuPDF for text-based pages; "
        "'azure' explicitly uses Azure AI Document Intelligence for the whole document "
        "(processes within your Azure tenant, handles scanned pages and complex tables, requires AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT); "
        "'gemini' uses Gemini AI for ALL pages (high quality but uses shared API infrastructure); "
        "'pymupdf' uses only local PyMuPDF (fastest, no API calls, but may miss scanned content or lose table formatting). "
        "Note: image files require Gemini or Azure DI — 'pymupdf' does not support images. "
        "Prefer 'auto' unless the user has a specific reason to override.",
    )
    mcp.add_tool(
        fn=index_directory,
        description="Index all supported files in a directory into a collection. "
        "Supports PDF, DOCX, HTML, PPTX, XLSX, plaintext, and image files (JPG, PNG, BMP, TIFF, WebP, HEIC/HEIF). "
        "Use glob_pattern to filter files (e.g., '*.pdf' for only PDFs, '*.png' for only PNGs). "
        "Set recursive=True to include subdirectories. "
        "extraction_method controls conversion quality: "
        "'auto' (default) uses Azure Document Intelligence if configured (best quality, private), "
        "otherwise Gemini AI for scanned/OCR pages and PyMuPDF for text-based pages; "
        "'azure' explicitly uses Azure AI Document Intelligence for the whole document "
        "(processes within your Azure tenant, handles scanned pages and complex tables, requires AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT); "
        "'gemini' uses Gemini AI for ALL pages (high quality but uses shared API infrastructure); "
        "'pymupdf' uses only local PyMuPDF (fastest, no API calls, but may miss scanned content or lose table formatting). "
        "Note: image files require Gemini or Azure DI — 'pymupdf' does not support images. "
        "Prefer 'auto' unless the user has a specific reason to override.",
    )
    mcp.add_tool(
        fn=remove_documents,
        description="Remove one or more documents from a collection by their file paths.",
    )
    mcp.add_tool(
        fn=list_documents,
        description="List all indexed documents in a specific collection.",
    )
    mcp.add_tool(
        fn=search,
        description="Search for relevant document chunks using semantic similarity across all collections. "
        "Returns the most relevant text passages matching your query.",
    )
    mcp.add_tool(
        fn=search_collection,
        description="Search within a specific collection. Shorthand for search() with collection parameter.",
    )
    mcp.add_tool(
        fn=get_document_content,
        description="Saves the content of an indexed document to a Markdown file.",
    )
