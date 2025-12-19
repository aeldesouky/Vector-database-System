"""
Embedding module for converting text documents into dense vector representations.

This module provides a clean abstraction over sentence-transformers
to support downstream vector storage and similarity search.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    A wrapper around Sentence-Transformers for text embedding.

    Attributes:
        model_name (str): Name of the sentence-transformer model.
        model (SentenceTransformer): Loaded embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.

        Args:
            model_name (str): Sentence-transformer model name.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): Input text documents.

        Returns:
            np.ndarray: 2D array of shape (n_texts, embedding_dim).
        """
        if not texts:
            raise ValueError("Input text list must not be empty.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single query string.

        Args:
            query (str): User query.

        Returns:
            np.ndarray: 1D embedding vector.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return embedding
