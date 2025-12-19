"""
Unit tests for vector store implementations.
"""

import numpy as np
import pytest

from src.vector_store.numpy_store import NumpyVectorStore
from src.vector_store.faiss_store import FaissVectorStore


@pytest.fixture
def sample_vectors() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def sample_metadata() -> list:
    return [
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
    ]


def test_numpy_store_search(sample_vectors, sample_metadata) -> None:
    store = NumpyVectorStore()
    store.add(sample_vectors, sample_metadata)

    query = np.array([1.0, 0.0, 0.0])
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0][1]["metadata"]["id"] == "a"


def test_numpy_store_learning_effect(sample_vectors, sample_metadata) -> None:
    store = NumpyVectorStore()
    store.add(sample_vectors, sample_metadata)

    store.apply_feedback(index=1, positive=True)
    assert store.learning_weights[1] > 1.0


def test_faiss_store_search(sample_vectors, sample_metadata) -> None:
    store = FaissVectorStore(dim=3)
    store.add(sample_vectors, sample_metadata)

    query = np.array([1.0, 0.0, 0.0])
    results = store.search(query, top_k=1)

    assert results[0][1]["metadata"]["id"] == "a"
