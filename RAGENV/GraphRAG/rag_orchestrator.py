"""
RAG Orchestrator - Main module that coordinates all RAG components.
"""

from typing import Any, Dict, Optional

from config import Config
from embeddings import EmbeddingGenerator
from pdf_processor import PDFProcessor
from response_generator import ResponseGenerator
from vector_store import VectorStore


class RAGOrchestrator:
    """Main orchestrator class that coordinates all RAG components."""

    def __init__(self, config: Config = None):
        """
        Initialize RAG orchestrator with all components.

        Args:
            config: Configuration object (creates default if None)
        """
        self.config = config if config else Config()

        print("\n" + "=" * 60)
        print("Initializing RAG System Components")
        print("=" * 60)

        self.pdf_processor = PDFProcessor(
            min_chunk_length=self.config.chunk_min_length
        )

        self.embedding_generator = EmbeddingGenerator(
            api_key=self.config.gemini_api_key,
            model=self.config.embedding_model,
            embedding_dimensions=self.config.embedding_dimensions,
            max_retries=self.config.max_retries,
        )

        self.vector_store = VectorStore(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database,
            embedding_dimensions=self.config.embedding_dimensions,
        )

        self.response_generator = ResponseGenerator(
            api_key=self.config.gemini_api_key,
            model=self.config.generation_model,
            max_retries=self.config.max_retries,
            thinking_budget_min=self.config.thinking_budget_min,
            thinking_budget_max=self.config.thinking_budget_max,
        )

        print("=" * 60)
        print("RAG System Initialized Successfully")
        print("=" * 60 + "\n")

    def process_and_store_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: Extract PDF, generate embeddings, store in Neo4j.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with processing statistics
        """
        print(f"\nProcessing PDF: {pdf_path}")
        print("-" * 60)

        chunks = self.pdf_processor.extract_chunks(pdf_path)

        if not chunks:
            return {
                "success": False,
                "message": "No chunks extracted from PDF",
                "chunks_processed": 0,
                "statistics": {"total_chunks": 0, "total_chars": 0, "avg_chunk_size": 0},
            }

        stats = self.pdf_processor.get_chunk_statistics(chunks)

        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        embeddings = []
        for i, chunk in enumerate(chunks, 1):
            embedding = self.embedding_generator.generate_document_embedding(chunk["text"])
            embeddings.append(embedding)
            if i % 10 == 0:
                print(f"Progress: {i}/{len(chunks)} embeddings generated")

        print("\nStoring chunks in Neo4j...")
        self.vector_store.store_chunks_batch(chunks, embeddings)

        print("-" * 60)
        print("PDF Processing Complete")

        return {
            "success": True,
            "message": f"Successfully processed {len(chunks)} chunks",
            "chunks_processed": len(chunks),
            "statistics": stats,
        }

    def _infer_source_filter(self, question: str) -> Optional[str]:
        """Infer document source filter from user query if a source name is mentioned."""
        q = (question or "").lower()
        sources = self.vector_store.get_all_sources()

        for source in sources:
            source_lower = source.lower()
            base_name = source_lower.rsplit(".", 1)[0]
            if source_lower in q or base_name in q:
                return source

        return None

    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Query the RAG system: retrieve relevant chunks and generate response.

        Args:
            question: User question
            top_k: Number of chunks to retrieve (uses config default if None)

        Returns:
            Dictionary with question, answer, and source chunks
        """
        if top_k is None:
            top_k = self.config.top_k_results

        print(f"\nProcessing Query: {question}")
        print("-" * 60)

        print("Generating query embedding...")
        query_embedding = self.embedding_generator.generate_query_embedding(question)

        source_filter = self._infer_source_filter(question)
        if source_filter:
            print(f"Detected source filter from question: {source_filter}")

        print(f"Searching for top {top_k} relevant chunks...")
        relevant_chunks = self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            source_filter=source_filter,
            min_similarity=self.config.min_similarity if self.config.min_similarity > -1 else None,
            expand_hops=self.config.graph_expand_hops,
        )

        if not relevant_chunks:
            print("No relevant chunks found")
            return {
                "question": question,
                "answer": "I could not find relevant information in the knowledge base to answer your question.",
                "sources": [],
            }

        print(f"Found {len(relevant_chunks)} relevant chunks")
        print("Generating response...")
        answer = self.response_generator.generate(question, relevant_chunks)

        print("-" * 60)
        print("Query Processing Complete")

        return {
            "question": question,
            "answer": answer,
            "sources": relevant_chunks,
        }

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database state."""
        chunk_count = self.vector_store.get_chunk_count()
        sources = self.vector_store.get_all_sources()

        return {
            "total_chunks": chunk_count,
            "total_documents": len(sources),
            "documents": sources,
        }

    def delete_document(self, source: str):
        """Delete all chunks from a specific document."""
        self.vector_store.delete_by_source(source)

    def close(self):
        """Close all connections."""
        self.vector_store.close()
        print("RAG System shut down successfully")
