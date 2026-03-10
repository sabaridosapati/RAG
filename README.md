# GraphRAG PDF Knowledge Base

A Python GraphRAG application that ingests PDF documents, stores chunk embeddings in Neo4j, and answers questions through a Gradio UI using Google Gemini.

## What this project does

- Extracts text from PDFs with Docling (with memory-safe fallback to page-by-page extraction).
- Splits extracted text into chunks and creates embeddings with Gemini.
- Stores chunks and embeddings in Neo4j with:
  - `Document` nodes
  - `Chunk` nodes
  - `HAS_CHUNK` and `NEXT` relationships
  - Vector index for semantic retrieval
- Retrieves relevant chunks using vector search plus optional graph expansion (`NEXT` hops).
- Generates answers with Gemini 2.5 Flash-Lite using retrieved context.
- Provides a Gradio UI for upload, query, and document deletion.

## Project structure

```text
GraphRAG/
  app.py                 # Gradio UI entrypoint
  rag_orchestrator.py    # End-to-end pipeline coordinator
  pdf_processor.py       # PDF extraction + chunking
  embeddings.py          # Gemini embeddings
  vector_store.py        # Neo4j storage + retrieval
  response_generator.py  # Gemini answer generation
  config.py              # Env-based configuration
  requirements.txt
  test/
```

## Requirements

- Python 3.11+ (3.12 also works)
- Running Neo4j instance with vector index support
- Google Gemini API key

## Installation

From `RAGENV/GraphRAG`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in `RAGENV/GraphRAG`.

```env
# Neo4j
NEO4J_URI=neo4j://<neo4jURL>
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=graphragdb
NEO4J_INSTANCE=SAB_Test

# Gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=768
GEMINI_GENERATION_MODEL=gemini-2.5-flash-lite

# Optional tuning
GRAPH_EXPAND_HOPS=1
MIN_SIMILARITY=-1.0
THINKING_BUDGET_MIN=0
THINKING_BUDGET_MAX=1024
```

Notes:

- `NEO4J_PASSWORD` and `GEMINI_API_KEY` are required; app startup fails if missing.
- If `MIN_SIMILARITY` is `-1.0`, similarity filtering is effectively disabled.

## Run the app

From `RAGENV/GraphRAG`:

```powershell
python app.py
```

Default Gradio server:

- Host: `0.0.0.0`
- Port: `portnumber`

Open the printed local URL in your browser.

## How to use

1. Open the **Upload Documents** tab and upload a PDF.
2. Wait for processing and indexing to finish.
3. Go to **Query Knowledge Base** and ask questions.
4. Optionally tune retrieval depth with the Top-K slider.
5. Use **Manage Database** to refresh status or delete a document by exact name.

## Retrieval and generation behavior

- Query embedding uses task type `RETRIEVAL_QUERY`.
- Document embeddings use task type `RETRIEVAL_DOCUMENT`.
- Retrieval pipeline:
  1. Vector search in Neo4j index (`chunk_embedding`)
  2. Optional graph expansion through `NEXT` relationships (`GRAPH_EXPAND_HOPS`)
  3. Source-aware filtering when a document name appears in the question
- Response generation uses a dynamic Gemini thinking budget based on query/context complexity.

## Troubleshooting

- No chunks extracted:
  - Reduce `chunk_min_length` in `config.py` (currently hardcoded).
  - Check if the PDF contains selectable text.
- Neo4j connection/auth errors:
  - Verify `NEO4J_URI`, credentials, and target database.
- Embedding dimension mismatch:
  - Ensure `GEMINI_EMBEDDING_DIMENSIONS` matches existing vector index dimensions.
  - The app attempts to recreate the index when dimensions change.
- API quota/rate limit:
  - Embedding and generation modules include retry with exponential backoff.

## Development notes

- The ingestion pipeline is orchestrated by `RAGOrchestrator.process_and_store_pdf`.
- Query flow is handled by `RAGOrchestrator.query`.
- Current tests are minimal; validate changes by running the app and testing upload/query flows.

## Next improvements

- Move `chunk_min_length` into `.env` and wire it through `Config` for runtime tuning.
- Add automated tests for PDF chunk extraction, retrieval quality, and response formatting.
- Support batch PDF uploads and background ingestion progress tracking in the UI.
- Add source-level metadata filters (date, tags, author) to improve retrieval precision.
- Add observability (structured logs + timing metrics) for ingestion and query latency.
