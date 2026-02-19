# Configuration

All configuration is via environment variables. Every setting has a sensible default — most deployments only need `GEMINI_API_KEY`.

## Key variables

- **`GEMINI_API_KEY`** — Google Gemini API key for OCR on scanned PDF pages. Without it, scanned pages are skipped (text-based PDFs still work fine).
- **`APPLICATIONINSIGHTS_CONNECTION_STRING`** — Azure Application Insights connection string. When set, logs and traces are exported to Azure Monitor in addition to stderr. When unset, there is no overhead.
- **`MCP_LOCAL_RAG_DATA_DIR`** — Base directory for server data (defaults to `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS, or `$XDG_DATA_HOME` / `~/.local/share` on Linux). A `mcp-local-rag/` subfolder is created inside it.

## PDF extraction method

The `index_files` and `index_directory` tools accept an `extraction_method` parameter that controls how PDF pages are converted to Markdown:

| Value | Behaviour | Best for |
|---|---|---|
| `auto` (default) | Uses local PyMuPDF for text-based pages; falls back to Gemini AI for scanned / OCR pages | Mixed documents — fast where possible, accurate where needed |
| `gemini` | Sends **every** page through Gemini AI | Documents with complex tables, forms, or scanned content where formatting fidelity matters |
| `pymupdf` | Uses only local PyMuPDF — no API calls | Speed-sensitive workloads or when offline; may lose table formatting and cannot OCR scanned pages |

When the user hasn't indicated a preference, the bot should ask whether they prefer **speed** (`pymupdf`), **quality** (`gemini`), or to **let the tool decide** (`auto`).

## Tuning

Additional `MCP_LOCAL_RAG_*` environment variables control chunking parameters, the embedding model, and concurrency limits. These all have reasonable defaults and are defined in `config.py`. The embedding model (`BAAI/bge-small-en-v1.5` by default) is downloaded automatically on first use; changing it requires re-indexing all documents.
