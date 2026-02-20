# mcp-local-rag

Local MCP server for RAG over PDFs, DOCX, and plaintext files.

## Requirements

For more complex PDFs, the following environment variables can be provided:

- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`; requires `mcp-local-rag[azure]`.
- `AZURE_DOCUMENT_INTELLIGENCE_KEY`; when omitted, `DefaultAzureCredential` is used. Requires `mcp-local-rag[azure]`.
- `GEMINI_API_KEY`

## Data Storage

By default, the server stores data in:

- **Windows**: `%LOCALAPPDATA%\mcp-local-rag\`
- **macOS**: `~/Library/Application Support/mcp-local-rag/`
- **Linux**: `$XDG_DATA_HOME/mcp-local-rag/`

The data directory contains:

- `markdown/` - Extracted Markdown content of indexed documents
- `metadata.db` - SQLite database for document/collection metadata
- `qdrant/` - Vector database for embeddings

AI Models are cached in the default HuggingFace cache directory (`~/.cache/huggingface/`).

To customize the data directory, set the `MCP_LOCAL_RAG_DATA_DIR` environment variable (a `mcp-local-rag/` subfolder is created automatically inside it).

## Usage

### VS Code

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "mcp-local-rag": {
      "command": "uvx",
      "args": [
        "--python",
        "3.13",  // Does not support Python 3.14 yet: https://github.com/microsoft/markitdown/issues/1470
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
        "--python",
        "3.13",  // Does not support Python 3.14 yet: https://github.com/microsoft/markitdown/issues/1470
        "--with",
        "pip-system-certs",
        "mcp-local-rag@latest"
      ]
    }
  }
}
```
