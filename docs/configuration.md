# Configuration

Additional configuration is possible via environment variables:

| Variable | Description |
|---|---|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure AI Document Intelligence endpoint URL. When set, `auto` extraction uses Azure DI for the entire document (best quality, processes within your Azure tenant). Requires `mcp-local-rag[azure]`. |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | API key for Azure Document Intelligence. When omitted, `DefaultAzureCredential` is used instead. |
| `GEMINI_API_KEY` | Google Gemini API key for OCR on scanned PDF pages. Used as a fallback when Azure DI is not configured. Without it, scanned pages are skipped (text-based PDFs still work fine). |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Azure Application Insights connection string. When set, logs and traces are exported to Azure Monitor in addition to stderr. When unset, there is no overhead. |
| `MCP_LOCAL_RAG_DATA_DIR` | Base directory for server data (defaults to `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS, or `$XDG_DATA_HOME` on Linux). A `mcp-local-rag/` subfolder is created inside it. |

## PDF extraction method

The `index_files` and `index_directory` tools accept an `extraction_method` parameter that controls how PDF pages are converted to Markdown:

| Value | Behaviour | Best for |
|---|---|---|
| `auto` (default) | Uses Azure Document Intelligence if configured (best quality, private); otherwise Gemini AI for scanned/OCR pages and PyMuPDF for text-based pages | Most deployments — picks the best available backend automatically |
| `azure` | Uses Azure AI Document Intelligence for the whole document | Complex tables, scanned content; processes within your Azure tenant. Requires `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` |
| `gemini` | Sends **every** page through Gemini AI | High quality OCR when Azure DI is not available |
| `pymupdf` | Uses only local PyMuPDF — no API calls | Speed-sensitive workloads or when offline; may miss scanned content or lose table formatting |

Prefer `auto` unless there is a specific reason to override.

## Tuning

Additional `MCP_LOCAL_RAG_*` environment variables control chunking parameters, the embedding model, and concurrency limits. These all have reasonable defaults and are defined in `config.py`. The embedding model (`BAAI/bge-small-en-v1.5` by default) is downloaded automatically on first use; changing it requires re-indexing all documents.
