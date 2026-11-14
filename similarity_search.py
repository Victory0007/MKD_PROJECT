"""Semantic product search backed by Sentence Transformers and Pinecone."""
import os
from functools import lru_cache

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mkd-ecommerce-ds-project")


def _required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def _get_model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_index():
    client = Pinecone(api_key=_required_env("PINECONE_API_KEY"))
    return client.Index(INDEX_NAME)


def similarity_search(query, top_k=5):
    """Return the closest products and their metadata for a text query."""
    query = (query or "").strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    vector = _get_model().encode(query, normalize_embeddings=True).tolist()
    response = _get_index().query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )

    matches = getattr(response, "matches", None) or []
    products = []
    for match in matches:
        metadata = getattr(match, "metadata", None) or {}
        products.append({
            "id": getattr(match, "id", None),
            "score": float(getattr(match, "score", 0.0) or 0.0),
            "description": metadata.get("description"),
            "stock_code": metadata.get("stock_code"),
            "unit_price": metadata.get("unit_price"),
        })
    return products
