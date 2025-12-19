"""
Base interface for vector storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    """

    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        """
        Add vectors and associated metadata to the store.

        Args:
            vectors (np.ndarray): 2D array of embeddings.
            metadata (List[dict]): Metadata for each vector.
        """

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Search for nearest vectors.

        Args:
            query_vector (np.ndarray): Query embedding.
            top_k (int): Number of results.

        Returns:
            List[Tuple[int, float]]: Index and score pairs.
        """
