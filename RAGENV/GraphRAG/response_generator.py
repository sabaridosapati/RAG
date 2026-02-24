"""
Response generation module using Google Gemini.
Handles generating responses from retrieved context.
"""

import random
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types


class ResponseGenerator:
    """Class for generating responses using Gemini LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        thinking_budget_min: int = 0,
        thinking_budget_max: int = 1024,
    ):
        """
        Initialize response generator.

        Args:
            api_key: Gemini API key
            model: Generation model to use
            max_retries: Maximum number of retries for rate limiting
            thinking_budget_min: Minimum thinking budget
            thinking_budget_max: Maximum thinking budget
        """
        self.api_key = api_key
        self.model_name = model
        self.max_retries = max_retries
        self.thinking_budget_min = max(0, thinking_budget_min)
        self.thinking_budget_max = max(self.thinking_budget_min, thinking_budget_max)
        self.client = genai.Client(api_key=self.api_key)

        print(f"Initialized Gemini Generation Model: {self.model_name}")

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response based on query and retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved relevant chunks

        Returns:
            Generated response text
        """
        context = self._build_context(context_chunks)
        prompt = self._create_prompt(query, context)
        thinking_budget = self._compute_thinking_budget(query, context_chunks)

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=thinking_budget
                        ),
                    ),
                )
                if response.text:
                    return response.text
                return "I could not generate a response for this query."

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise Exception("Rate limit exceeded. Please wait and try again.")
                else:
                    print(f"Error generating response: {str(e)}")
                    raise

    def _compute_thinking_budget(self, query: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Dynamically select a thinking budget based on query complexity and context size.
        """
        query_words = len((query or "").split())
        context_words = sum(len((chunk.get("text") or "").split()) for chunk in chunks[:4])

        complexity_markers = [
            "compare",
            "analyze",
            "why",
            "how",
            "derive",
            "impact",
            "tradeoff",
            "relationship",
        ]
        complexity_hits = sum(1 for marker in complexity_markers if marker in (query or "").lower())

        complexity_score = query_words + (context_words // 5) + (complexity_hits * 25)

        if complexity_score < 80:
            return self.thinking_budget_min
        if complexity_score > 320:
            return self.thinking_budget_max

        span = self.thinking_budget_max - self.thinking_budget_min
        scaled = int(self.thinking_budget_min + span * ((complexity_score - 80) / 240))
        return max(self.thinking_budget_min, min(self.thinking_budget_max, scaled))

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build formatted context from chunks.
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Context {i} - Source: {chunk['source']}, Chunk {chunk['chunk_index']}]\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the LLM.
        """
        return f"""You are a helpful AI assistant answering questions based on provided context.

Context Information:
{context}

User Question: {query}

Instructions:
- Answer the question using ONLY the information provided in the context above
- Be accurate and specific
- If the context does not contain enough information to fully answer the question, acknowledge this clearly
- Cite which context sources you used in your answer
- Keep your answer clear and concise

Answer:"""
