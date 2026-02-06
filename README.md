# mcp-local-rag

Local MCP server for RAG over PDFs, DOCX, and plaintext files.

## Requirements

To process scanned PDFs, set the `GEMINI_API_KEY` environment variable for OCR support. Text-based PDFs, DOCX, and plaintext files work without it.

## Data Storage

By default, the server stores data in `%LOCALAPPDATA%\mcp-local-rag\` on Windows, or `./mcp-local-rag` in the current directory on other systems:

- `metadata.db` - SQLite database for document/collection metadata
- `qdrant/` - Vector database for embeddings

AI Models are cached in the default HuggingFace cache directory (`~/.cache/huggingface/`).

To customize the data directory, set the `MCP_LOCAL_RAG_DATA_DIR` environment variable.

## Usage

### VS Code

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "mcp-local-rag": {
      "command": "uvx",
      "args": [
        "mcp-local-rag@latest"
      ]
    }
  }
}
```

If you run into SSL errors (Zscaler), you can try:

```json
{
  "servers": {
    "mcp-local-rag": {
      "command": "uvx",
      "args": [
        "--native-tls",
        "--with",
        "pip-system-certs",
        "mcp-local-rag@latest"
      ]
    }
  }
}
```
