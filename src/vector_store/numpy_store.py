"""
NumPy-based in-memory vector store with learning support.
"""

from typing import List, Tuple
from pathlib import Path
import json

import numpy as np

from src.vector_store.base import VectorStore


class NumpyVectorStore(VectorStore):
    """
    Simple in-memory vector store using NumPy with learning weights.
    """

    def __init__(
        self,
        vector_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.vectors: np.ndarray | None = None
        self.metadata: List[dict] = []
        self.learning_weights: np.ndarray | None = None

        self.vector_path = vector_path
        self.metadata_path = metadata_path

        if vector_path and metadata_path:
            self._load_from_disk()

    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        self.vectors = vectors
        self.metadata = metadata
        self.learning_weights = np.ones(vectors.shape[0])

        self._save_to_disk()

    def _save_to_disk(self) -> None:
        if self.vector_path:
            np.save(self.vector_path, self.vectors)

        if self.metadata_path:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

    def _load_from_disk(self) -> None:
        if self.vector_path and self.vector_path.exists():
            self.vectors = np.load(self.vector_path)

        if self.metadata_path and self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        if self.vectors is not None:
            self.learning_weights = np.ones(self.vectors.shape[0])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, dict]]:
        cosine = (
            self.vectors @ query_vector
            / (
                np.linalg.norm(self.vectors, axis=1)
                * np.linalg.norm(query_vector)
                + 1e-10
            )
        )
        dot = self.vectors @ query_vector
        euclidean = -np.linalg.norm(self.vectors - query_vector, axis=1)

        final = (
            0.5 * cosine
            + 0.3 * dot
            + 0.2 * euclidean
        ) * self.learning_weights

        indices = np.argsort(final)[::-1][:top_k]

        return [
            (
                idx,
                {
                    "cosine": float(cosine[idx]),
                    "dot": float(dot[idx]),
                    "euclidean": float(euclidean[idx]),
                    "final": float(final[idx]),
                    "metadata": self.metadata[idx],
                },
            )
            for idx in indices
        ]

    def apply_feedback(self, index: int, positive: bool) -> None:
        """
        Apply relevance feedback to learning weights.
        """
        if self.learning_weights is None:
            return

        self.learning_weights[index] *= 1.1 if positive else 0.9
