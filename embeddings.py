import os
from collections.abc import Iterable
from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from env_loader import load_env_file

# No API key is required for local embeddings. The .env file is only used here
# so you can optionally override the local Hugging Face model name.
load_env_file()

# Free local retrieval model. It creates 384-dimensional vectors and is stronger
# for retrieval than the previous MiniLM learning model.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load the local embedding model once and reuse it.

    The first run may download the model from Hugging Face. After that, it is
    cached on your machine and can run without calling an API.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_reranker_model() -> CrossEncoder:
    """Load the local reranker once and reuse it."""
    return CrossEncoder(RERANKER_MODEL)


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
    # BGE embedding models perform better for retrieval when the query receives
    # this instruction prefix. Do not add this prefix to document chunks.
    vectors = embed_texts([EMBEDDING_QUERY_PREFIX + text], batch_size=1)
    return vectors[0] if vectors else []


def embed_documents(texts: Iterable[str]) -> list[list[float]]:
    """Embed document chunks for storage."""
    return embed_texts(texts)


def rerank_results(question: str, results: list[dict], *, top_k: int) -> list[dict]:
    """Rerank candidate chunks with a cross-encoder.

    Embeddings are fast but approximate. A reranker reads `(question, chunk text)`
    pairs directly and is usually better at ordering the final top results.
    """
    if not results:
        return []

    reranker = _get_reranker_model()
    pairs = [(question, result.get("text", "")) for result in results]
    scores = reranker.predict(pairs)

    reranked = []
    for result, score in zip(results, scores, strict=True):
        item = dict(result)
        item["embedding_score"] = item.get("score", 0.0)
        item["rerank_score"] = float(score)
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]
