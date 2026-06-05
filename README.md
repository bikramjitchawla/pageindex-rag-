# PDF Chunking and Embeddings

This repo only contains the code needed to:

1. Extract text from a PDF.
2. Split the text into overlapping chunks.
3. Create local sentence-transformers embeddings for those chunks.
4. Save the chunks and vectors into `vector_store.json`.

There is no API server, PageIndex, Weave tracing, or final-answer LLM in this cleaned version.

## Files

- `build_vector_store.py` extracts PDF text, chunks it, embeds it, and saves `vector_store.json`.
- `embeddings.py` loads the local `sentence-transformers/all-MiniLM-L6-v2` embedding model.
- `vector_store.py` loads `vector_store.json` and includes cosine similarity helper code.
- `env_loader.py` loads optional settings from `.env`.
- `pdfs/sample_report.pdf` is the sample input PDF.

## Setup

No API key is required.

The default local embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
You can optionally override it in `.env`:

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Build Embeddings

```bash
python3 build_vector_store.py --pdf pdfs/sample_report.pdf
```

Optional chunk settings:

```bash
python3 build_vector_store.py --pdf pdfs/sample_report.pdf --chunk-words 700 --overlap-words 120
```

The output is:

```text
vector_store.json
```

## Test Search

Search the vector store without an LLM:

```bash
python3 search_vector_store.py "What is this document about?"
```

Show more or fewer matches:

```bash
python3 search_vector_store.py "What is Kubernetes?" --top-k 3
```
