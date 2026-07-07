import os


def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into os.environ.

    Example .env:
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
    EMBEDDING_QUERY_PREFIX=Represent this sentence for searching relevant passages:
    RERANKER_MODEL=BAAI/bge-reranker-base
    PGVECTOR_DATABASE_URL=

    Existing environment variables are not overwritten. That means a value already
    exported in your terminal wins over the value in `.env`.
    """
    if not os.path.exists(path):
        return

    with open(path, "r") as f:
        for raw in f:
            # Remove spaces and the trailing newline from each file line.
            line = raw.strip()

            # Skip blank lines, comments, and malformed lines.
            if not line or line.startswith("#") or "=" not in line:
                continue

            # Split only on the first "=" so values can safely contain "=" later.
            key, value = line.split("=", 1)
            key = key.strip()

            # Support values written as KEY=value, KEY="value", or KEY='value'.
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
