"""
Unit tests for NumPy vector store persistence.
"""

from pathlib import Path
import shutil

import numpy as np
import pytest

from src.vector_store.numpy_store import NumpyVectorStore


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


def test_vectors_are_saved_and_loaded(temp_data_dir: Path) -> None:
    vector_path = temp_data_dir / "vectors.npy"
    metadata_path = temp_data_dir / "metadata.json"

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    metadata = [
        {"id": "doc1"},
        {"id": "doc2"},
    ]

    store = NumpyVectorStore(
        vector_path=vector_path,
        metadata_path=metadata_path,
    )
    store.add(vectors, metadata)

    assert vector_path.exists()
    assert metadata_path.exists()

    reloaded_store = NumpyVectorStore(
        vector_path=vector_path,
        metadata_path=metadata_path,
    )

    assert reloaded_store.vectors is not None
    assert reloaded_store.vectors.shape == vectors.shape
    assert reloaded_store.metadata == metadata


def test_search_after_reload(temp_data_dir: Path) -> None:
    vector_path = temp_data_dir / "vectors.npy"
    metadata_path = temp_data_dir / "metadata.json"

    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    metadata = [
        {"id": "a"},
        {"id": "b"},
    ]

    store = NumpyVectorStore(
        vector_path=vector_path,
        metadata_path=metadata_path,
    )
    store.add(vectors, metadata)

    reloaded_store = NumpyVectorStore(
        vector_path=vector_path,
        metadata_path=metadata_path,
    )

    query = np.array([1.0, 0.0])
    results = reloaded_store.search(query, top_k=1)

    assert results[0][1]["metadata"]["id"] == "a"
