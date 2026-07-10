from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mcp_local_rag.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Deferred import: sentence_transformers pulls in torch (~20s on cold start).
    # Keeping it here means the process starts and registers tools instantly;
    # the model only loads when embed_texts/embed_query is first called.
    from sentence_transformers import SentenceTransformer as _ST  # noqa: PLC0415

    # Try loading from local cache first (fast, no network — avoids SSL failures
    # behind corporate proxies like Zscaler). Fall back to network download only
    # if the model isn't cached yet (first-ever run).
    try:
        return _ST(EMBEDDING_MODEL, local_files_only=True)
    except OSError:
        return _ST(EMBEDDING_MODEL)


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
