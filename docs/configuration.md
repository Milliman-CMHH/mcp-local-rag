# Configuration

All configuration is via environment variables. Every setting has a sensible default — most deployments only need `GEMINI_API_KEY`.

## Key variables

- **`GEMINI_API_KEY`** — Google Gemini API key for OCR on scanned PDF pages. Without it, scanned pages are skipped (text-based PDFs still work fine).
- **`APPLICATIONINSIGHTS_CONNECTION_STRING`** — Azure Application Insights connection string. When set, logs and traces are exported to Azure Monitor in addition to stderr. When unset, there is no overhead.
- **`MCP_LOCAL_RAG_DATA_DIR`** — Base directory for server data (defaults to the current working directory). A `.mcp-local-rag/` subfolder is created inside it.

## Tuning

Additional `MCP_LOCAL_RAG_*` environment variables control chunking parameters, the embedding model, and concurrency limits. These all have reasonable defaults and are defined in `config.py`. The embedding model (`BAAI/bge-small-en-v1.5` by default) is downloaded automatically on first use; changing it requires re-indexing all documents.
