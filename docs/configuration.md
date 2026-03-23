# Configuration

Additional configuration is possible via environment variables:

| Variable | Description |
|---|---|
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Azure Application Insights connection string. When set, logs and traces are exported to Azure Monitor in addition to stderr. When unset, there is no overhead. |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure AI Document Intelligence endpoint URL. When set, `auto` extraction uses Azure DI for the entire document (best quality, processes within your Azure tenant). Requires `mcp-local-rag[azure]`. |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | API key for Azure Document Intelligence. When omitted, `DefaultAzureCredential` is used instead. |
| `GEMINI_API_KEY` | Google Gemini API key. Used for OCR on scanned PDF pages and for image extraction. Falls back from Azure DI when not configured. Without it, scanned pages are skipped and image files cannot be indexed (text-based PDFs still work fine). |
| `MCP_LOCAL_RAG_DATA_DIR` | Base directory for server data (defaults to `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS, or `$XDG_DATA_HOME` on Linux). A `mcp-local-rag/` subfolder is created inside it. |
| `MCP_LOCAL_RAG_GEMINI_MODEL` | Gemini model to use for OCR and image extraction (default: `gemini-3-pro-preview`). |

## Extraction method

The `index_files` and `index_directory` tools accept an `extraction_method` parameter that controls how files are converted to Markdown:

| Value | Behaviour | Best for |
|---|---|---|
| `auto` (default) | Uses Azure Document Intelligence if configured (best quality, private); otherwise Gemini AI for scanned/OCR pages and PyMuPDF for text-based pages. See [Image files](#image-files) for image-specific behaviour. | Most deployments — picks the best available backend automatically |
| `azure` | Uses Azure AI Document Intelligence for the whole document | Complex tables, scanned content; processes within your Azure tenant. Requires `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` |
| `gemini` | Sends **every** page through Gemini AI | High quality OCR when Azure DI is not available |
| `pymupdf` | Uses only local PyMuPDF — no API calls (PDFs only, not supported for images) | Speed-sensitive workloads or when offline; may miss scanned content or lose table formatting |

Prefer `auto` unless there is a specific reason to override.

### Image files

Image files require Gemini or Azure Document Intelligence for extraction — there is no local fallback. In `auto` mode, Gemini is preferred when configured (higher resolution support); otherwise Azure DI is used as a fallback. Note that Azure DI only supports JPEG, PNG, BMP, and TIFF images — formats like GIF and WebP require Gemini. The `pymupdf` method does not support images and will return an error.

## Tuning

| Variable | Default | Description |
|---|---|---|
| `MCP_LOCAL_RAG_CHUNK_SIZE` | `512` | Maximum tokens per chunk |
| `MCP_LOCAL_RAG_CHUNK_OVERLAP` | `50` | Overlap tokens between chunks |
| `MCP_LOCAL_RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers embedding model. Downloaded automatically on first use; changing it requires re-indexing all documents. |
| `MCP_LOCAL_RAG_MAX_CONCURRENT_FILES` | `32` | Maximum files indexed concurrently |
| `MCP_LOCAL_RAG_MAX_CONCURRENT_GEMINI` | `128` | Maximum concurrent Gemini API requests across all files |
