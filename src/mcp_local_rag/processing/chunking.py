from functools import lru_cache

from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

from mcp_local_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_splitter() -> TextSplitter:
    return TextSplitter.from_huggingface_tokenizer(  # pyright: ignore[reportUnknownMemberType]
        tokenizer=Tokenizer.from_pretrained(EMBEDDING_MODEL),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        capacity=(CHUNK_SIZE - CHUNK_OVERLAP, CHUNK_SIZE),
        overlap=CHUNK_OVERLAP,
    )


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return _get_splitter().chunks(text)
