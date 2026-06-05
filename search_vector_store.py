import argparse

from embeddings import EMBEDDING_MODEL, embed_query
from vector_store import DEFAULT_STORE_PATH, cosine_similarity, load_vector_store


def search_vector_store(
    question: str,
    *,
    store_path: str = DEFAULT_STORE_PATH,
    top_k: int = 5,
) -> list[dict]:
    """Return the most similar chunks for a question.

    This tests the vector store without using a chat model:
    question -> local embedding -> cosine similarity -> top matching chunks.
    """
    store = load_vector_store(store_path)
    query_vector = embed_query(question)

    results: list[dict] = []
    for chunk in store.get("chunks", []):
        score = cosine_similarity(query_vector, chunk.get("embedding", []))
        results.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "source": chunk.get("source", ""),
                "page": chunk.get("page"),
                "score": score,
                "text": chunk.get("text", ""),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search vector_store.json.")
    parser.add_argument("question", help="Search question/query text.")
    parser.add_argument("--store", default=DEFAULT_STORE_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = search_vector_store(args.question, store_path=args.store, top_k=args.top_k)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Question: {args.question}")
    print()

    for index, result in enumerate(results, start=1):
        preview = " ".join(result["text"].split())
        if len(preview) > 500:
            preview = preview[:500] + "..."

        print(
            f"{index}. {result['chunk_id']} "
            f"page={result['page']} score={result['score']:.4f}"
        )
        print(preview)
        print()


if __name__ == "__main__":
    main()
