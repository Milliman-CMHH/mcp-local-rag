from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from mcp_local_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

if TYPE_CHECKING:
    from semantic_text_splitter import TextSplitter


@lru_cache(maxsize=1)
def _get_splitter() -> TextSplitter:
    # Deferred imports: tokenizers downloads a HuggingFace tokenizer on first call
    # (~3s). Keeping them here avoids blocking process startup.
    from semantic_text_splitter import TextSplitter as _TS  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    return _TS.from_huggingface_tokenizer(  # pyright: ignore[reportUnknownMemberType]
        tokenizer=Tokenizer.from_pretrained(EMBEDDING_MODEL),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        capacity=(CHUNK_SIZE - CHUNK_OVERLAP, CHUNK_SIZE),
        overlap=CHUNK_OVERLAP,
    )


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return _get_splitter().chunks(text)
