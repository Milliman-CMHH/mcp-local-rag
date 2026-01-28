# mcp-local-rag

Fully local MCP server for RAG over PDFs, DOCX, and plaintext files.

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
