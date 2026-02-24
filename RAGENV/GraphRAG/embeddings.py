"""
Embedding generation module using Google Gemini.
Handles text-to-vector conversion with retry logic.
"""

import random
import time
from typing import List

from google import genai
from google.genai import types


class EmbeddingGenerator:
    """Class for generating embeddings using Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        embedding_dimensions: int = 768,
        max_retries: int = 3,
    ):
        """
        Initialize embedding generator.

        Args:
            api_key: Gemini API key
            model: Embedding model to use
            max_retries: Maximum number of retries for rate limiting
        """
        self.api_key = api_key
        self.model = model
        self.embedding_dimensions = int(embedding_dimensions)
        self.max_retries = max_retries
        self.client = genai.Client(api_key=self.api_key)

        print(f"Initialized Gemini Embedding Model: {self.model} ({self.embedding_dimensions} dims)")

    def generate(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding vector for text with retry logic.

        Args:
            text: Input text to embed
            task_type: Type of task (retrieval_document or retrieval_query)

        Returns:
            Embedding vector as list of floats
        """
        for attempt in range(self.max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=[text],
                    config=types.EmbedContentConfig(
                        task_type=self._map_task_type(task_type),
                        output_dimensionality=self.embedding_dimensions,
                    ),
                )
                if not result.embeddings:
                    raise ValueError("Embedding response did not contain vectors.")
                vector = list(result.embeddings[0].values)
                if len(vector) != self.embedding_dimensions:
                    raise ValueError(
                        f"Embedding dimension mismatch. Expected {self.embedding_dimensions}, got {len(vector)}."
                    )
                return vector

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise Exception("Rate limit exceeded. Please wait and try again.")
                else:
                    print(f"Error generating embedding: {str(e)}")
                    raise

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.generate(query, task_type="retrieval_query")

    def generate_document_embedding(self, document: str) -> List[float]:
        """
        Generate embedding for a document.

        Args:
            document: Document text

        Returns:
            Embedding vector
        """
        return self.generate(document, task_type="retrieval_document")

    @staticmethod
    def _map_task_type(task_type: str) -> str:
        """Map task labels to Gemini API task type values."""
        normalized = (task_type or "").strip().lower()
        mapping = {
            "retrieval_query": "RETRIEVAL_QUERY",
            "retrieval_document": "RETRIEVAL_DOCUMENT",
        }
        return mapping.get(normalized, "RETRIEVAL_DOCUMENT")
