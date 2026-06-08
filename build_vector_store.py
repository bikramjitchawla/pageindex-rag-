import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


# Default input/output paths used when no CLI arguments are passed.
DEFAULT_PDF_PATH = "pdfs/openshift-guide-screen.pdf"
DEFAULT_STORE_PATH = "vector_store.json"


@dataclass
class Chunk:
    """A single searchable piece of PDF text before it is embedded."""

    # Stable ID for citations/debugging. Example: p3-c2 means page 3, chunk 2.
    chunk_id: str
    # Original file name, kept so vector_store.json knows where each chunk came from.
    source: str
    # 1-based PDF page number.
    page: int
    # The actual text that will be sent to the embedding model.
    text: str


def _clean_text(text: str) -> str:
    """Normalize raw PDF text before chunking."""
    # Some PDFs can contain null bytes. They do not help search, so replace them.
    text = text.replace("\x00", " ")

    # Collapse repeated spaces/tabs while preserving paragraph newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Keep at most one blank line between paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract cleaned text from each page of the PDF.

    Returns a list of `(page_number, page_text)` tuples. Empty pages are skipped.
    """
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: install pypdf with `python3 -m pip install pypdf`."
        ) from exc

    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []

    # `enumerate(..., start=1)` keeps page numbers aligned with how humans count pages.
    for index, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append((index, text))
    return pages


def chunk_text(
    text: str,
    *,
    chunk_words: int = 350,
    overlap_words: int = 70,
) -> list[str]:
    """Split text into overlapping word chunks.

    Example with `chunk_words=3` and `overlap_words=1`:
    `one two three four five` becomes:
    - `one two three`
    - `three four five`

    The overlap preserves context near chunk boundaries. Without overlap, an answer
    split across two chunks could become harder to retrieve later.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must not be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    # This is simple word-based chunking. It does not use an LLM.
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []

    # Move forward by the non-overlapping part of the chunk.
    # For 350 words with 70 overlap, each next chunk starts 280 words later.
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))

        # Stop once the final chunk has included the end of the text.
        if start + chunk_words >= len(words):
            break
    return chunks


def build_chunks(
    pdf_path: str,
    *,
    chunk_words: int = 350,
    overlap_words: int = 70,
) -> list[Chunk]:
    """Extract PDF pages and convert them into Chunk objects."""
    source = os.path.basename(pdf_path)
    chunks: list[Chunk] = []

    # Chunk each page separately so every chunk can keep a useful page number.
    for page_number, page_text in extract_pdf_pages(pdf_path):
        page_chunks = chunk_text(
            page_text,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )

        # Give each chunk a stable ID based on page and page-local chunk index.
        for chunk_index, text in enumerate(page_chunks, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"p{page_number}-c{chunk_index}",
                    source=source,
                    page=page_number,
                    text=text,
                )
            )

    return chunks


def build_vector_store(
    pdf_path: str,
    *,
    output_path: str = DEFAULT_STORE_PATH,
    store: str = "pgvector",
    chunk_words: int = 350,
    overlap_words: int = 70,
) -> dict:
    """Extract, chunk, embed, and save chunks to the selected vector store.

    This is the main pipeline:
    PDF file -> page text -> chunks -> embeddings -> pgvector or vector_store.json
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 1: Convert the PDF into text chunks.
    chunks = build_chunks(
        str(pdf),
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )
    if not chunks:
        raise RuntimeError(f"No extractable text found in {pdf_path}")

    # Import the local embedding model only when embeddings are actually needed.
    # This lets you import and test chunking helpers before installing dependencies.
    from embeddings import EMBEDDING_MODEL, embed_documents

    # Step 2: Send only the text to the embedding model.
    # The metadata stays local and is reattached below.
    texts = [chunk.text for chunk in chunks]
    vectors = embed_documents(texts)

    # Every chunk must have exactly one embedding vector.
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks"
        )

    # Step 3: Store metadata, text, and embedding vectors.
    json_store = {
        "source": str(pdf),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "chunks": [
            {
                **asdict(chunk),
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    }

    saved_targets: list[str] = []

    if store in ("json", "both"):
        with open(output_path, "w") as f:
            json.dump(json_store, f)
        saved_targets.append(output_path)

    if store in ("pgvector", "both"):
        from pg_vector_store import save_chunks_to_pgvector

        row_count = save_chunks_to_pgvector(
            chunks=chunks,
            vectors=vectors,
            source=os.path.basename(str(pdf)),
            embedding_model=EMBEDDING_MODEL,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
        saved_targets.append(f"pgvector:{row_count} rows")

    # Return a short summary for CLI output or future programmatic use.
    return {
        "saved_targets": saved_targets,
        "source": str(pdf),
        "chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
    }


def main() -> None:
    """CLI entry point for building the vector store from a PDF."""
    parser = argparse.ArgumentParser(description="Build a local embedding vector store.")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to the source PDF.")
    parser.add_argument(
        "--out",
        default=DEFAULT_STORE_PATH,
        help="Path to write the JSON vector store when --store json or --store both.",
    )
    parser.add_argument(
        "--store",
        choices=["pgvector", "json", "both"],
        default="pgvector",
        help="Where to store embeddings.",
    )

    # These two flags control chunk size. Larger chunks carry more context but make
    # retrieval less precise. Smaller chunks are more precise but can lose context.
    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=70)
    args = parser.parse_args()

    result = build_vector_store(
        args.pdf,
        output_path=args.out,
        store=args.store,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    print(
        f"Saved {result['chunks']} embedded chunks from {result['source']} "
        f"to {', '.join(result['saved_targets'])} using {result['embedding_model']}"
    )


if __name__ == "__main__":
    main()
