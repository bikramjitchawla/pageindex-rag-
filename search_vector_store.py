import argparse

from embeddings import EMBEDDING_MODEL, RERANKER_MODEL, embed_query, rerank_results
from pg_vector_store import search_pgvector
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
    parser = argparse.ArgumentParser(description="Search pgvector or vector_store.json.")
    parser.add_argument("question", help="Search question/query text.")
    parser.add_argument("--backend", choices=["pgvector", "json"], default="pgvector")
    parser.add_argument("--store", default=DEFAULT_STORE_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidates",
        type=int,
        default=25,
        help="Number of vector matches to fetch before reranking.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank candidates with the local cross-encoder. Better quality but slower.",
    )
    args = parser.parse_args()

    if args.backend == "pgvector":
        candidate_count = max(args.candidates, args.top_k)
        results = search_pgvector(embed_query(args.question), top_k=candidate_count)
    else:
        candidate_count = max(args.candidates, args.top_k)
        results = search_vector_store(
            args.question,
            store_path=args.store,
            top_k=candidate_count,
        )

    if args.rerank:
        results = rerank_results(args.question, results, top_k=args.top_k)
    else:
        results = results[: args.top_k]

    print(f"Embedding model: {EMBEDDING_MODEL}")
    if args.rerank:
        print(f"Reranker model: {RERANKER_MODEL}")
    print(f"Question: {args.question}")
    print()

    for index, result in enumerate(results, start=1):
        preview = " ".join(result["text"].split())
        if len(preview) > 500:
            preview = preview[:500] + "..."

        print(
            f"{index}. {result['chunk_id']} "
            f"page={result['page']} score={result['score']:.4f} source={result['source']}"
        )
        if "embedding_score" in result:
            print(f"   embedding_score={result['embedding_score']:.4f}")
        print(preview)
        print()


if __name__ == "__main__":
    main()
