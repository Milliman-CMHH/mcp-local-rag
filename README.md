# mcp-local-rag

Fully local MCP server for RAG over PDFs, DOCX, and plaintext files.

## Requirements

To process scanned PDFs, install [Tesseract] for OCR support. Text-based PDFs, DOCX, and plaintext files work without it.

### Windows

1. Run:

    ```pwsh
    winget install -e --id UB-Mannheim.TesseractOCR
    ```

1. Add the tesseract folder to your PATH: `C:\Users\<User>\AppData\Local\Programs\Tesseract-OCR`

1. Restart VS Code to refresh environment variables

## Data Storage

By default, the server stores data in the current working directory under `.mcp-local-rag/`:

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

[Tesseract]: https://github.com/tesseract-ocr/tesseract
