# Architecture

## Overview

```
MCP Client (VS Code, etc.)
    │
    ▼
┌────────────────────────────────────┐
│  FastMCP Server  (server.py)       │
├────────────────────────────────────┤
│  Tools         (tools/)            │
├────────────────────────────────────┤
│  Processing    (processing/)       │
│  extractors · chunking · embed     │
├────────────────────────────────────┤
│  Storage       (storage/)          │
│  SQLite metadata · Qdrant vectors  │
└────────────────────────────────────┘
```

The server starts up via an async lifespan that initializes shared resources (database connections, embedding model, Gemini client, concurrency semaphores) and tears them down on shutdown. All shared state lives in an `AppContext` dataclass passed through MCP tool calls.

## Indexing pipeline

When files are indexed, each goes through:

1. **Extraction** — convert the source file to Markdown (see [PDF extraction](#pdf-extraction) below; DOCX uses markitdown; plaintext is read directly)
2. **Chunking** — split the Markdown into token-aware overlapping chunks
3. **Embedding** — encode chunks into vectors via sentence-transformers
4. **Storage** — write vectors to Qdrant and metadata to SQLite

### Change detection

Files are skipped if their content hasn't changed since the last index (checked via mtime, then SHA-256 hash). Use `force=True` to bypass this.

## PDF extraction

PDF extraction works page-by-page. Each page is independently classified by pymupdf4llm's `should_ocr_page` heuristic, which considers things like how much of the page is covered by text vs. images vs. vector graphics, whether existing text characters are readable, and whether image-heavy pages look like photos or scanned documents.

- **Text-extractable pages** are converted locally with pymupdf4llm
- **Scanned/unreadable pages** are sent to Google Gemini for OCR (requires `GEMINI_API_KEY`)

Gemini OCR requests respect the `Retry-After` header on 429 responses and retry automatically.

### Page cache (resumability)

Each successfully converted page is cached in SQLite, keyed by file content hash and page index. If extraction is interrupted (e.g., Gemini rate limits partway through a large PDF), the next attempt picks up where it left off — cached pages are reused and only unconverted pages are retried. The cache is cleared automatically after a document is fully indexed or deleted.

## Concurrency

Files are indexed concurrently (configurable limit, default 16). Across all files, Gemini OCR requests share a separate global concurrency limit (also configurable, default 16) to avoid overwhelming the API.

## Storage

The server uses two local stores under the `.mcp-local-rag/` directory:

- **SQLite** — document and collection metadata, plus the temporary page cache for resumable PDF extraction. Uses WAL mode for concurrent access safety.
- **Qdrant** — file-backed vector database storing all chunk embeddings. Chunks carry payload fields for filtering by collection and document.

## Logging & telemetry

All logs go to **stderr** (visible in VS Code's MCP server output panel). Indexing operations log per-file and per-page progress, including which extraction method was used and summary counts.

When `APPLICATIONINSIGHTS_CONNECTION_STRING` is set, logs are additionally exported to Azure Monitor via OpenTelemetry. When unset, there is no overhead.
