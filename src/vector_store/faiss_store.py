"""
FAISS-based vector store implementation.
"""

from typing import List, Tuple

import faiss
import numpy as np

from src.vector_store.base import VectorStore


class FaissVectorStore(VectorStore):
    """
    In-memory FAISS vector store.
    """

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.vectors: np.ndarray | None = None
        self.metadata: List[dict] = []
        self.learning_weights: np.ndarray | None = None

    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D.")

        self.vectors = vectors.astype("float32")
        self.index.add(self.vectors)
        self.metadata = metadata
        self.learning_weights = np.ones(vectors.shape[0])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, dict]]:
        if self.vectors is None or self.learning_weights is None:
            raise RuntimeError("Vector store is empty.")

        query = query_vector.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            adjusted_score = scores[0][rank] * self.learning_weights[idx]
            results.append(
                (
                    int(idx),
                    {
                        "faiss_score": float(scores[0][rank]),
                        "final": float(adjusted_score),
                        "metadata": self.metadata[idx],
                    },
                )
            )

        return results

    def apply_feedback(self, index: int, positive: bool) -> None:
        if self.learning_weights is None:
            return

        if positive:
            self.learning_weights[index] *= 1.1
        else:
            self.learning_weights[index] *= 0.9
