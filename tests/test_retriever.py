"""
Unit tests for the retrieval layer.
"""

import numpy as np
import pytest

from src.embeddings.embedder import TextEmbedder
from src.retrieval.retriever import Retriever
from src.vector_store.numpy_store import NumpyVectorStore


@pytest.fixture
def setup_retriever() -> Retriever:
    embedder = TextEmbedder()
    store = NumpyVectorStore()

    documents = [
        "Database indexes speed up queries.",
        "University attendance policy details.",
        "Machine learning models learn patterns.",
    ]

    vectors = embedder.embed_texts(documents)
    metadata = [
        {"text": documents[0]},
        {"text": documents[1]},
        {"text": documents[2]},
    ]

    store.add(vectors, metadata)

    return Retriever(embedder, store, top_k=2)


def test_retrieve_returns_results(setup_retriever) -> None:
    results = setup_retriever.retrieve("What is a database index?")

    assert len(results) == 2
    assert "scores" in results[0]
    assert "cosine" in results[0]["scores"]


def test_feedback_changes_learning_weight(setup_retriever) -> None:
    retriever = setup_retriever

    initial_results = retriever.retrieve("attendance rules")
    index = initial_results[0]["index"]

    retriever.apply_feedback(index=index, positive=True)

    store = retriever.vector_store
    assert store.learning_weights[index] > 1.0
