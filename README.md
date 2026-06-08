# PDF Chunking and Embeddings

This repo only contains the code needed to:

1. Extract text from a PDF.
2. Split the text into overlapping chunks.
3. Create local sentence-transformers embeddings for those chunks.
4. Save the chunks and vectors into Postgres/pgvector.

There is no API server, PageIndex, Weave tracing, or final-answer LLM in this cleaned version.
Search can optionally rerank retrieved chunks with a local cross-encoder, but
plain pgvector search is the default because it is much faster.

## Files

- `build_vector_store.py` extracts PDF text, chunks it, embeds it, and saves to pgvector by default.
- `embeddings.py` loads the local `BAAI/bge-small-en-v1.5` embedding model.
- `pg_vector_store.py` creates the pgvector table, inserts chunks, and searches Postgres.
- `search_vector_store.py` searches pgvector and reranks candidates with `BAAI/bge-reranker-base`.
- `vector_store.py` loads `vector_store.json` and includes cosine similarity helper code for optional JSON search.
- `env_loader.py` loads optional settings from `.env`.
- `pdfs/sample_report.pdf` is the sample input PDF.

## Setup

No API key is required.

The default local embedding model is `BAAI/bge-small-en-v1.5`.
Configure `.env`:

```bash
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_QUERY_PREFIX=Represent this sentence for searching relevant passages: 
RERANKER_MODEL=BAAI/bge-reranker-base
PGVECTOR_DATABASE_URL=postgresql://appuser:test%40123@localhost:5432/vectordb
```

The password `test@123` must be encoded as `test%40123` inside a PostgreSQL URL.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Build Embeddings

```bash
python3 build_vector_store.py --pdf pdfs/openshift-guide-screen.pdf
```

Optional chunk settings:

```bash
python3 build_vector_store.py --pdf pdfs/openshift-guide-screen.pdf --chunk-words 350 --overlap-words 70
```

To also write a JSON copy:

```bash
python3 build_vector_store.py --pdf pdfs/openshift-guide-screen.pdf --store both
```

## Test Search

Search the vector store without an LLM:

```bash
python3 search_vector_store.py "What is this document about?"
```

Show more or fewer matches:

```bash
python3 search_vector_store.py "What is OpenShift?" --top-k 3
```

Use reranking when you want better final ordering and can accept slower search:

```bash
python3 search_vector_store.py "What is OpenShift?" --top-k 3 --rerank
```
