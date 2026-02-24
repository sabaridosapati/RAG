"""
Configuration module for RAG system.
Handles loading and validating environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for RAG system."""
    
    def __init__(self):
        """Initialize and validate configuration."""
        # Neo4j Configuration
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "graphragdb")
        self.neo4j_instance = os.getenv("NEO4J_INSTANCE", "SAB_Test")
        
        # Gemini API Configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Model Configuration
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.embedding_dimensions = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))
        self.generation_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash-lite")
        
        # Thinking budget configuration for Gemini 2.5 models
        self.thinking_budget_min = int(os.getenv("THINKING_BUDGET_MIN", "0"))
        self.thinking_budget_max = int(os.getenv("THINKING_BUDGET_MAX", "1024"))
        
        # RAG Configuration
        self.chunk_min_length = 20  # Minimum chunk length in characters (reduced from 50)
        self.top_k_results = 5  # Default number of results to retrieve
        self.graph_expand_hops = int(os.getenv("GRAPH_EXPAND_HOPS", "1"))
        self.min_similarity = float(os.getenv("MIN_SIMILARITY", "-1.0"))
        self.max_retries = 3  # Max retries for API calls
        
        # Validate required settings
        self._validate()
    
    def _validate(self):
        """Validate that required configuration values are set."""
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def display_config(self):
        """Display current configuration (without sensitive data)."""
        print("="*60)
        print("RAG System Configuration")
        print("="*60)
        print(f"Neo4j URI: {self.neo4j_uri}")
        print(f"Neo4j User: {self.neo4j_user}")
        print(f"Neo4j Database: {self.neo4j_database}")
        print(f"Neo4j Instance: {self.neo4j_instance}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Embedding Dimensions: {self.embedding_dimensions}")
        print(f"Generation Model: {self.generation_model}")
        print(f"Thinking Budget Min: {self.thinking_budget_min}")
        print(f"Thinking Budget Max: {self.thinking_budget_max}")
        print(f"Chunk Min Length: {self.chunk_min_length}")
        print(f"Top K Results: {self.top_k_results}")
        print(f"Graph Expand Hops: {self.graph_expand_hops}")
        #print("Using embedding model:", self.model)
        print("="*60)
