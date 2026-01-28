from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from mcp_local_rag.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    model = get_embedding_model()
    embeddings: NDArray[np.float32] = model.encode(  # pyright: ignore[reportUnknownMemberType]
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> NDArray[np.float32]:
    return embed_texts([query])[0]


def get_embedding_dimension() -> int | None:
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    return dim
