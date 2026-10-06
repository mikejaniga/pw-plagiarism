"""Module for generating embedding vectors and similarity matrices."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TextEmbedder:
    """Wrapper class around a SentenceTransformer model for computing feature vectors."""

    def __init__(self, model_name: str, model_label: str) -> None:
        self.model_name = model_name
        self.model_label = model_label
        print(
            f"-> Loading language model: {self.model_label} ({self.model_name})..."
        )
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Creates normalized embedding vectors for the given list of texts."""
        embedding_dim: int | None = self.model.get_embedding_dimension()
        if embedding_dim is None:
            raise RuntimeError(
                "Embedding dimension is not available for this model."
            )

        if not texts:
            return np.empty((0, embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    @staticmethod
    def compute_similarity_matrix(
        embeddings_a: np.ndarray, embeddings_b: np.ndarray
    ) -> np.ndarray:
        """Computes the cosine similarity matrix between two sets of vectors."""
        if embeddings_a.size == 0 or embeddings_b.size == 0:
            return np.zeros((len(embeddings_a), len(embeddings_b)))

        # Vectors are normalized, so the dot product equals the cosine of the angle
        sim = cosine_similarity(embeddings_a, embeddings_b)
        # Guard against numerical errors (outside the [-1, 1] range)
        return np.clip(sim, -1.0, 1.0)
