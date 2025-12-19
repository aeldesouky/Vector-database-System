"""
Unit tests for the embedding module.
"""

import numpy as np
import pytest

from src.embeddings.embedder import TextEmbedder


def test_embed_texts_shape() -> None:
    embedder = TextEmbedder()
    texts = [
        "Databases use indexes to improve performance.",
        "Machine learning models learn from data.",
    ]

    embeddings = embedder.embed_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0


def test_embed_query_shape() -> None:
    embedder = TextEmbedder()
    query = "What is an index in a database?"

    embedding = embedder.embed_query(query)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0


def test_empty_text_list_raises_error() -> None:
    embedder = TextEmbedder()

    with pytest.raises(ValueError):
        embedder.embed_texts([])


def test_empty_query_raises_error() -> None:
    embedder = TextEmbedder()

    with pytest.raises(ValueError):
        embedder.embed_query("   ")
