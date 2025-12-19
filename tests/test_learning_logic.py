"""
Tests for learning weight updates and ranking behavior.
"""

import numpy as np

from src.vector_store.numpy_store import NumpyVectorStore


def test_learning_weight_increase() -> None:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1]])
    metadata = [{"id": "doc1"}, {"id": "doc2"}]

    store = NumpyVectorStore()
    store.add(vectors, metadata)

    initial_weight = store.learning_weights[0]

    store.learning_weights[0] *= 1.1

    assert store.learning_weights[0] > initial_weight


def test_learning_affects_ranking() -> None:
    vectors = np.array([[1.0, 0.0], [0.95, 0.05]])
    metadata = [{"id": "doc1"}, {"id": "doc2"}]

    store = NumpyVectorStore()
    store.add(vectors, metadata)

    query = np.array([1.0, 0.0])

    initial = store.search(query, top_k=2)
    assert initial[0][1]["metadata"]["id"] == "doc1"

    store.learning_weights[1] *= 2.0

    updated = store.search(query, top_k=2)
    assert updated[0][1]["metadata"]["id"] == "doc2"
