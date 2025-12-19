"""
Retrieval layer that connects embeddings with vector stores.
"""

from typing import List, Dict, Any

import numpy as np

from src.embeddings.embedder import TextEmbedder
from src.vector_store.base import VectorStore


class Retriever:
    """
    High-level retriever for similarity-based document search.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        top_k: int = 3,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            embedder (TextEmbedder): Embedding model wrapper.
            vector_store (VectorStore): Vector store backend.
            top_k (int): Number of documents to retrieve.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query (str): User query.

        Returns:
            List[Dict[str, Any]]: Retrieved documents with scores.
        """
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, self.top_k)

        retrieved = []
        for index, score_dict in results:
            retrieved.append(
                {
                    "index": index,
                    "scores": score_dict,
                }
            )

        return retrieved

    def apply_feedback(self, index: int, positive: bool) -> None:
        """
        Apply feedback to the vector store.

        Args:
            index (int): Document index.
            positive (bool): Feedback polarity.
        """
        if hasattr(self.vector_store, "apply_feedback"):
            self.vector_store.apply_feedback(index, positive)
