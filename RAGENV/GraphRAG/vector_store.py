"""
Neo4j vector database module.
Handles storing and retrieving embeddings from Neo4j.
"""

from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


class VectorStore:
    """Class for managing vector storage in Neo4j."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "graphragdb",
        embedding_dimensions: int = 768,
    ):
        """
        Initialize Neo4j vector store.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.vector_index_name = "chunk_embedding"
        self.embedding_dimensions: int = int(embedding_dimensions)

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

        print(f"Connected to Neo4j database: {self.database}")
        self._setup_schema()
        self._ensure_vector_index(self.embedding_dimensions)

    def _setup_schema(self):
        """Create Neo4j schema with constraints for GraphRAG."""
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE CONSTRAINT chunk_id IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT document_name IF NOT EXISTS
                FOR (d:Document) REQUIRE d.name IS UNIQUE
            """)

    def _ensure_vector_index(self, embedding_dimensions: int):
        """Create or align vector index to the required dimensions."""
        current_dims = self._get_vector_index_dimensions()
        if current_dims is not None and int(current_dims) != int(embedding_dimensions):
            print(
                f"Vector index dimension mismatch: existing={current_dims}, required={embedding_dimensions}. "
                "Recreating index."
            )
            with self.driver.session(database=self.database) as session:
                session.run(f"DROP INDEX {self.vector_index_name} IF EXISTS")

        with self.driver.session(database=self.database) as session:
            session.run(f"""
                CREATE VECTOR INDEX {self.vector_index_name} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                }}}}
            """, {"dims": embedding_dimensions})

        self.embedding_dimensions = int(embedding_dimensions)

    def _get_vector_index_dimensions(self) -> Optional[int]:
        """Read current vector index dimensionality if index exists."""
        queries = [
            """
            SHOW VECTOR INDEXES
            YIELD name, options
            WHERE name = $name
            RETURN options AS options
            """,
            """
            SHOW INDEXES
            YIELD name, type, options
            WHERE name = $name AND type = 'VECTOR'
            RETURN options AS options
            """,
        ]
        for query in queries:
            try:
                with self.driver.session(database=self.database) as session:
                    record = session.run(query, {"name": self.vector_index_name}).single()
                    if not record:
                        return None
                    options = record.get("options") or {}
                    index_cfg = options.get("indexConfig") or {}
                    dims = index_cfg.get("vector.dimensions")
                    return int(dims) if dims is not None else None
            except Exception:
                continue
        return None

    def store_chunks_batch(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store multiple chunks with embeddings and graph links.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings lengths do not match")

        dims = len(embeddings[0])
        if dims != self.embedding_dimensions:
            raise ValueError(
                f"Document embedding dimension mismatch. Expected {self.embedding_dimensions}, got {dims}."
            )
        self._ensure_vector_index(dims)

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": int(chunk["chunk_index"]),
                "embedding": embedding,
                "metadata": str(chunk.get("metadata", {})),
            })

        with self.driver.session(database=self.database) as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (d:Document {name: row.source})
                MERGE (c:Chunk {id: row.id})
                SET c.text = row.text,
                    c.source = row.source,
                    c.chunk_index = row.chunk_index,
                    c.embedding = row.embedding,
                    c.metadata = row.metadata
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, {"rows": rows})

            session.run("""
                UNWIND $rows AS row
                MATCH (c1:Chunk {source: row.source, chunk_index: row.chunk_index})
                MATCH (c2:Chunk {source: row.source, chunk_index: row.chunk_index + 1})
                MERGE (c1)-[:NEXT]->(c2)
            """, {"rows": rows})

        print(f"Stored {len(chunks)} chunks in Neo4j")

    def _search_vector_candidates(
        self,
        query_embedding: List[float],
        top_k: int,
        source_filter: Optional[str],
        min_similarity: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Get seed chunks from vector search with dynamic Cypher."""
        candidate_k = max(top_k * 3, top_k)

        cypher = """
            CALL db.index.vector.queryNodes($index_name, $candidate_k, $query_embedding)
            YIELD node, score
            WITH node, score
            WHERE ($source_filter IS NULL OR node.source = $source_filter)
              AND ($min_similarity IS NULL OR score >= $min_similarity)
            RETURN node.id AS id,
                   node.text AS text,
                   node.source AS source,
                   node.chunk_index AS chunk_index,
                   score AS similarity
            ORDER BY similarity DESC
            LIMIT $candidate_k
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, {
                "index_name": self.vector_index_name,
                "candidate_k": candidate_k,
                "query_embedding": query_embedding,
                "source_filter": source_filter,
                "min_similarity": min_similarity,
            })
            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "source": record["source"],
                    "chunk_index": record["chunk_index"],
                    "similarity": float(record["similarity"]),
                }
                for record in result
            ]

    def _fallback_manual_similarity(
        self,
        query_embedding: List[float],
        top_k: int,
        source_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fallback when vector index procedures are unavailable."""
        cypher = """
            MATCH (c:Chunk)
            WHERE ($source_filter IS NULL OR c.source = $source_filter)
            WITH c,
                 reduce(dot = 0.0, i IN range(0, size(c.embedding)-1) |
                    dot + c.embedding[i] * $query_embedding[i]) AS similarity
            RETURN c.id AS id,
                   c.text AS text,
                   c.source AS source,
                   c.chunk_index AS chunk_index,
                   similarity
            ORDER BY similarity DESC
            LIMIT $top_k
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, {
                "query_embedding": query_embedding,
                "source_filter": source_filter,
                "top_k": max(top_k * 2, top_k),
            })
            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "source": record["source"],
                    "chunk_index": record["chunk_index"],
                    "similarity": float(record["similarity"]),
                }
                for record in result
            ]

    def _expand_candidates_with_graph(
        self,
        seeds: List[Dict[str, Any]],
        top_k: int,
        expand_hops: int,
        source_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Expand retrieval around vector seeds over the chunk graph."""
        if not seeds:
            return []
        if expand_hops <= 0:
            return seeds[:top_k]

        seed_rows = [
            {
                "id": seed["id"],
                "similarity": float(seed["similarity"]),
            }
            for seed in seeds
        ]
        hops = max(1, int(expand_hops))

        cypher = f"""
            UNWIND $seed_rows AS seed_row
            MATCH (seed:Chunk {{id: seed_row.id}})
            WITH seed, toFloat(seed_row.similarity) AS seed_score
            OPTIONAL MATCH (seed)-[:NEXT*1..{hops}]-(nbr:Chunk)
            WITH seed, seed_score, collect(DISTINCT nbr)[0..$max_neighbors] AS neighbors
            WITH [seed] + neighbors AS expanded, seed_score
            UNWIND expanded AS c
            WITH c, seed_score
            WHERE ($source_filter IS NULL OR c.source = $source_filter)
            WITH c, max(seed_score) AS graph_score
            RETURN c.id AS id,
                   c.text AS text,
                   c.source AS source,
                   c.chunk_index AS chunk_index,
                   graph_score AS similarity
            ORDER BY similarity DESC, chunk_index ASC
            LIMIT $top_k
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, {
                "seed_rows": seed_rows,
                "max_neighbors": top_k,
                "source_filter": source_filter,
                "top_k": top_k,
            })
            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "source": record["source"],
                    "chunk_index": record["chunk_index"],
                    "similarity": float(record["similarity"]),
                }
                for record in result
            ]

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        source_filter: Optional[str] = None,
        min_similarity: Optional[float] = None,
        expand_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using dynamic GraphRAG retrieval:
        vector candidates + optional graph expansion.
        """
        if not query_embedding:
            return []
        if len(query_embedding) != self.embedding_dimensions:
            raise ValueError(
                f"Query embedding dimension mismatch. Expected {self.embedding_dimensions}, got {len(query_embedding)}."
            )

        try:
            seeds = self._search_vector_candidates(
                query_embedding=query_embedding,
                top_k=top_k,
                source_filter=source_filter,
                min_similarity=min_similarity,
            )
        except Exception as e:
            print(f"Vector index search unavailable, using fallback similarity: {e}")
            seeds = self._fallback_manual_similarity(
                query_embedding=query_embedding,
                top_k=top_k,
                source_filter=source_filter,
            )

        return self._expand_candidates_with_graph(
            seeds=seeds,
            top_k=top_k,
            expand_hops=expand_hops,
            source_filter=source_filter,
        )

    def get_all_sources(self) -> List[str]:
        """Get list of all document sources in the database."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document)
                RETURN d.name AS source
                ORDER BY source
            """)
            return [record["source"] for record in result]

    def get_chunk_count(self) -> int:
        """Get total number of chunks in database."""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            return result.single()["count"]

    def delete_by_source(self, source: str):
        """Delete all chunks from a specific source and detach graph links."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {name: $source})-[:HAS_CHUNK]->(c:Chunk)
                WITH d, collect(c) AS chunks
                FOREACH (x IN chunks | DETACH DELETE x)
                DETACH DELETE d
                RETURN size(chunks) AS deleted_count
            """, {"source": source})
            deleted = result.single()["deleted_count"]
            print(f"Deleted {deleted} chunks from source: {source}")

    def close(self):
        """Close Neo4j database connection."""
        self.driver.close()
        print("Neo4j connection closed")
