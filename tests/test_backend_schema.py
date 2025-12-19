"""
Tests that NumPy and FAISS backends return different score schemas.
"""

import numpy as np

from src.vector_store.numpy_store import NumpyVectorStore
from src.vector_store.faiss_store import FaissVectorStore


def test_numpy_store_score_schema() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [{"id": "a"}, {"id": "b"}]

    store = NumpyVectorStore()
    store.add(vectors, metadata)

    result = store.search(np.array([1.0, 0.0]), top_k=1)[0][1]

    assert "cosine" in result
    assert "dot" in result
    assert "euclidean" in result
    assert "final" in result


def test_faiss_store_score_schema() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    metadata = [{"id": "a"}, {"id": "b"}]

    store = FaissVectorStore(dim=2)
    store.add(vectors, metadata)

    result = store.search(np.array([1.0, 0.0]), top_k=1)[0][1]

    assert "faiss_score" in result
    assert "final" in result
    assert "cosine" not in result
