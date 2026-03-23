from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

from mcp_local_rag.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return np.stack(embeddings).astype(np.float32)


def embed_query(query: str) -> NDArray[np.float32]:
    return embed_texts([query])[0]


def get_embedding_dimension() -> int | None:
    model = get_embedding_model()
    dim: int = model.embedding_size  # pyright: ignore[reportAttributeAccessIssue]
    return dim
