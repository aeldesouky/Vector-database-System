"""
Tests for learning behavior and weight mutation.
"""

import numpy as np

from src.vector_store.numpy_store import NumpyVectorStore


def test_learning_weights_mutate_in_place() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    metadata = [{"id": "a"}, {"id": "b"}]

    store = NumpyVectorStore()
    store.add(vectors, metadata)

    original_id = id(store.learning_weights)

    store.learning_weights[0] *= 1.1

    assert id(store.learning_weights) == original_id
    assert store.learning_weights[0] > 1.0
def test_learning_changes_ranking() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ]
    )
    metadata = [{"id": "doc1"}, {"id": "doc2"}]

    store = NumpyVectorStore()
    store.add(vectors, metadata)

    query = np.array([1.0, 0.0])

    initial_results = store.search(query, top_k=2)
    assert initial_results[0][1]["metadata"]["id"] == "doc1"

    # Boost second document
    store.learning_weights[1] *= 2.0

    new_results = store.search(query, top_k=2)
    assert new_results[0][1]["metadata"]["id"] == "doc2"
