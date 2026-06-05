import os
from collections.abc import Iterable
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from env_loader import load_env_file

# No API key is required for local embeddings. The .env file is only used here
# so you can optionally override the local Hugging Face model name.
load_env_file()

# Recommended free local model. It creates 384-dimensional vectors and is good
# enough for learning semantic search/RAG.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load the local embedding model once and reuse it.

    The first run may download the model from Hugging Face. After that, it is
    cached on your machine and can run without calling an API.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(
    texts: Iterable[str],
    *,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed texts locally and return one vector per input text."""
    # Empty strings cannot produce useful embeddings, so skip them.
    text_list = [text for text in texts if text.strip()]
    if not text_list:
        return []

    model = _get_embedding_model()

    # normalize_embeddings=True makes cosine similarity equivalent to dot-product
    # ranking, while still working with the cosine helper in vector_store.py.
    vectors = model.encode(
        text_list,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed one search query.

    This is not used while building the vector store, but it is useful if you later
    add retrieval/search back on top of `vector_store.json`.
    """
    vectors = embed_texts([text], batch_size=1)
    return vectors[0] if vectors else []


def embed_documents(texts: Iterable[str]) -> list[list[float]]:
    """Embed document chunks for storage."""
    return embed_texts(texts)
