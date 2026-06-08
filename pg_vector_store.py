import os
from urllib.parse import quote, urlsplit, urlunsplit

import psycopg

from env_loader import load_env_file


load_env_file()

# Your provided password contains "@", which must be URL-encoded as "%40" in a
# PostgreSQL URL. The default below is the encoded version of:
# postgresql://appuser:test@123@localhost:5432/vectordb
DEFAULT_DATABASE_URL = "postgresql://appuser:test%40123@localhost:5432/vectordb"
DATABASE_URL = os.getenv("PGVECTOR_DATABASE_URL", DEFAULT_DATABASE_URL)
TABLE_NAME = os.getenv("PGVECTOR_TABLE", "document_chunks")
EMBEDDING_MODEL_FILTER = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def _vector_literal(vector: list[float]) -> str:
    """Format a Python list as a pgvector literal."""
    return "[" + ",".join(str(value) for value in vector) + "]"


def _safe_table_name(name: str) -> str:
    """Keep table names simple because identifiers cannot be query parameters."""
    if not name.replace("_", "").isalnum():
        raise ValueError("PGVECTOR_TABLE must contain only letters, numbers, and underscores")
    return name


def normalize_database_url(raw_url: str) -> str:
    """Return a URL where the password is safe for PostgreSQL URL parsing.

    If a password contains special characters, especially "@", the URL parser can
    mistake part of the password for the host. Prefer putting this in `.env`:

    PGVECTOR_DATABASE_URL=postgresql://appuser:test%40123@localhost:5432/vectordb
    """
    parts = urlsplit(raw_url)
    if not parts.hostname or not parts.username:
        return raw_url
    if parts.password and "%" in parts.password:
        return raw_url

    username = quote(parts.username, safe="")
    password = quote(parts.password or "", safe="")
    auth = username if not password else f"{username}:{password}"
    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"{auth}@{host}", parts.path, parts.query, parts.fragment))


def init_pgvector_store(*, dimension: int, database_url: str = DATABASE_URL) -> None:
    """Create the pgvector extension and document chunk table if needed."""
    table = _safe_table_name(TABLE_NAME)

    with psycopg.connect(normalize_database_url(database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id BIGSERIAL PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunk_words INTEGER NOT NULL,
                    overlap_words INTEGER NOT NULL,
                    embedding vector({dimension}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (source, embedding_model, chunk_id)
                )
                """
            )

            # Exact search works without an index. This index helps once the table
            # gets large. If the installed pgvector version lacks HNSW support,
            # insertion/search still works without it.
            try:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw_idx
                    ON {table}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            except psycopg.Error:
                conn.rollback()
                with conn.cursor() as retry_cur:
                    retry_cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    retry_cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {table} (
                            id BIGSERIAL PRIMARY KEY,
                            chunk_id TEXT NOT NULL,
                            source TEXT NOT NULL,
                            page INTEGER NOT NULL,
                            text TEXT NOT NULL,
                            embedding_model TEXT NOT NULL,
                            chunk_words INTEGER NOT NULL,
                            overlap_words INTEGER NOT NULL,
                            embedding vector({dimension}) NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            UNIQUE (source, embedding_model, chunk_id)
                        )
                        """
                    )


def save_chunks_to_pgvector(
    *,
    chunks: list,
    vectors: list[list[float]],
    source: str,
    embedding_model: str,
    chunk_words: int,
    overlap_words: int,
    database_url: str = DATABASE_URL,
) -> int:
    """Store chunks and their embeddings in Postgres/pgvector."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    dimension = len(vectors[0])
    init_pgvector_store(dimension=dimension, database_url=database_url)
    table = _safe_table_name(TABLE_NAME)

    rows = [
        (
            chunk.chunk_id,
            chunk.source,
            chunk.page,
            chunk.text,
            embedding_model,
            chunk_words,
            overlap_words,
            _vector_literal(vector),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    with psycopg.connect(normalize_database_url(database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {table} WHERE source = %s AND embedding_model = %s",
                (source, embedding_model),
            )
            cur.executemany(
                f"""
                INSERT INTO {table} (
                    chunk_id,
                    source,
                    page,
                    text,
                    embedding_model,
                    chunk_words,
                    overlap_words,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (source, embedding_model, chunk_id)
                DO UPDATE SET
                    page = EXCLUDED.page,
                    text = EXCLUDED.text,
                    chunk_words = EXCLUDED.chunk_words,
                    overlap_words = EXCLUDED.overlap_words,
                    embedding = EXCLUDED.embedding,
                    created_at = now()
                """,
                rows,
            )
    return len(rows)


def search_pgvector(
    query_vector: list[float],
    *,
    top_k: int = 5,
    embedding_model: str = EMBEDDING_MODEL_FILTER,
    database_url: str = DATABASE_URL,
) -> list[dict]:
    """Search Postgres/pgvector using cosine distance."""
    table = _safe_table_name(TABLE_NAME)
    query = _vector_literal(query_vector)

    with psycopg.connect(normalize_database_url(database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    chunk_id,
                    source,
                    page,
                    text,
                    embedding_model,
                    1 - (embedding <=> %s::vector) AS score
                FROM {table}
                WHERE embedding_model = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query, embedding_model, query, top_k),
            )
            return [
                {
                    "chunk_id": row[0],
                    "source": row[1],
                    "page": row[2],
                    "text": row[3],
                    "embedding_model": row[4],
                    "score": float(row[5]),
                }
                for row in cur.fetchall()
            ]
