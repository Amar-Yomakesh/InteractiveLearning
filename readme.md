# Interactive Learning — RAG Documentation Assistant

A Retrieval-Augmented Generation (RAG) assistant that scrapes any web-based documentation site, builds a local vector database, and lets you chat with the content through a Streamlit UI. Currently configured for MuleSoft CloudHub docs, but designed to work with any documentation source.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Generalising to Any Documentation Source](#generalising-to-any-documentation-source)
- [Moving from Local to Cloud Vector DB](#moving-from-local-to-cloud-vector-db)
- [Use Cases](#use-cases)

---

## Architecture

The system follows a two-phase RAG pipeline: an offline **ingestion phase** that builds the knowledge base, and an online **query phase** that answers questions at runtime.

```
╔══════════════════════════════════════════════════════════════════╗
║                     INGESTION PIPELINE                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Docs Website                                                   ║
║   (any URL)   ──► Web Scraper ──► Raw HTML/Text                  ║
║                   (BFS crawl)                                    ║
║                       │                                          ║
║                       ▼                                          ║
║               Document Chunker                                   ║
║               (RecursiveCharacterTextSplitter,                   ║
║                800 tokens, 150 overlap)                          ║
║                       │                                          ║
║                       ▼                                          ║
║               Embedding Model                                    ║
║               (OpenAI text-embedding-3-small)                    ║
║                       │                                          ║
║                       ▼                                          ║
║               Vector Store  ◄── stored locally                   ║
║               (ChromaDB)        or in cloud                      ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                       QUERY PIPELINE                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   User Question                                                  ║
║       │                                                          ║
║       ▼                                                          ║
║   Embed Question ──► Query Embedding                             ║
║   (same embedding model)                                         ║
║       │                                                          ║
║       ▼                                                          ║
║   Vector Similarity Search                                       ║
║   (ChromaDB, top-k chunks)                                       ║
║       │                                                          ║
║       ▼                                                          ║
║   Build Prompt                                                   ║
║   [system prompt + retrieved context + question]                 ║
║       │                                                          ║
║       ▼                                                          ║
║   LLM  (GPT-4o-mini)                                             ║
║       │                                                          ║
║       ▼                                                          ║
║   Answer + Source Citations ──► Streamlit Chat UI                ║
╚══════════════════════════════════════════════════════════════════╝
```

### Key components

| Component | File | Description |
|---|---|---|
| Web Scraper | `app/utils.py` — `CloudHubDocsScraper` | BFS crawl scoped to a base URL; strips nav/footer noise |
| Document Chunker | `app/utils.py` — `DocumentChunker` | Recursive character splitting with configurable size & overlap |
| Vector Store | `app/utils.py` — `ChromaDBManager` | Embeds chunks via OpenAI, persists to ChromaDB |
| RAG Assistant | `app/utils.py` — `RAGAssistant` | Retrieves top-k chunks, builds prompt, calls the LLM |
| Chat UI | `app/streamlit_app.py` | Multi-turn Streamlit chat with source citation expanders |
| Config | `app/config.py` | All parameters driven by environment variables |

---

## Project Structure

```
InteractiveLearning/
├── app/
│   ├── config.py            # Environment-based configuration
│   ├── ingest.py            # Run this once to build the vector DB
│   ├── ingest.ipynb         # Notebook version of the ingestion pipeline
│   ├── streamlit_app.py     # Chat UI (run with streamlit run)
│   └── utils.py             # Scraper, Chunker, ChromaDBManager, RAGAssistant
├── data/
│   ├── raw/                 # (optional) raw scraped HTML cache
│   └── processed/           # (optional) intermediate processed docs
├── vector_db/               # Local ChromaDB persistence directory
├── tests/
├── .env                     # Your secrets (not committed)
├── .env.example             # Template for required env vars
├── requirements.txt
└── readme.md
```

---

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY and SCRAPE_BASE_URL
```

**3. Run ingestion** (builds the vector DB — run once, or whenever the source docs change)

```bash
cd app
python ingest.py
```

**4. Launch the chat UI**

```bash
streamlit run app/streamlit_app.py
```

---

## Configuration

All settings are environment variables with sane defaults (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `SCRAPE_BASE_URL` | MuleSoft CloudHub | The root URL to crawl |
| `SCRAPE_MAX_PAGES` | 100 | Maximum pages to scrape |
| `SCRAPE_DELAY` | 1.0 s | Polite crawl delay between requests |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between consecutive chunks |
| `VECTOR_DB_PATH` | `./vector_db` | Local path for ChromaDB storage |
| `COLLECTION_NAME` | `cloudhub_docs` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `MAX_RETRIEVAL_CHUNKS` | 10 | Top-k chunks retrieved per query |
| `MAX_CONTEXT_TOKENS` | 6000 | Token budget for retrieved context |

---

## Generalising to Any Documentation Source

The scraper is parameterised — pointing it at a different docs site requires only a `.env` change:

```bash
# Point to a different documentation site
SCRAPE_BASE_URL=https://docs.yourcompany.com/api-reference/
COLLECTION_NAME=yourproduct_docs
SCRAPE_MAX_PAGES=500
```

The scraper does a BFS crawl scoped to the domain and path prefix of `SCRAPE_BASE_URL`, so it stays contained to the section you specify. This makes it directly usable for:

**Public documentation sites**
- Any vendor docs (AWS, Azure, Stripe, Twilio, etc.)
- Open source project docs (FastAPI, Django, React, etc.)

**Company internal documentation**
- Confluence wikis — export to HTML or scrape directly if network-accessible
- Internal developer portals or GitBook instances
- SharePoint or Notion pages (export to HTML first)
- GitHub Pages or self-hosted static doc sites

For **internal sources behind authentication**, replace the scraper's `fetch_page` method with authenticated HTTP calls:

```python
# Example: Confluence with API token
headers = {
    "Authorization": "Bearer YOUR_CONFLUENCE_TOKEN",
    "Accept": "application/json",
}
# Or use session cookies for SSO-authenticated portals
```

For **non-HTML sources** (PDFs, Word docs, Markdown files), swap out the scraper for a document loader that reads from the filesystem and feeds the same `List[Dict]` format into the existing chunker and embedding pipeline:

```python
# The rest of the pipeline (chunker → embedder → ChromaDB) stays unchanged
scraped_docs = [
    {"url": str(path), "title": path.stem, "text": extract_text(path), "length": ...}
    for path in Path("./internal_docs").rglob("*.pdf")
]
```

---

## Moving from Local to Cloud Vector DB

The current setup persists ChromaDB to a local folder (`./vector_db`). This works well for a single developer but doesn't scale to teams or production deployments.

### Option 1 — ChromaDB Cloud (Chroma Cloud)

The lowest-friction upgrade. Replace `PersistentClient` with `HttpClient` pointing at a hosted Chroma instance:

```python
# Current (local)
self.client = chromadb.PersistentClient(path="./vector_db")

# Cloud-hosted Chroma
self.client = chromadb.HttpClient(
    host="https://api.trychroma.com",
    tenant="your-tenant",
    database="your-database",
    headers={"x-chroma-token": os.getenv("CHROMA_API_KEY")},
)
```

No other code changes needed — the collection API is identical.

### Option 2 — Pinecone

Pinecone is a managed vector database designed for production scale. The ChromaDB-specific code in `ChromaDBManager` is the only part that changes:

```python
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("your-index-name")

# Upsert embeddings
index.upsert(vectors=[
    {"id": chunk_id, "values": embedding, "metadata": metadata}
    for chunk_id, embedding, metadata in zip(ids, embeddings, metadatas)
])

# Query
results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
```

### Option 3 — pgvector (PostgreSQL)

If you already run PostgreSQL (e.g. on AWS RDS or Supabase), the `pgvector` extension adds a native vector column type. Good choice when you want to store vectors alongside relational data:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    id     SERIAL PRIMARY KEY,
    text   TEXT,
    source_url   TEXT,
    source_title TEXT,
    embedding    vector(1536)   -- dimension matches text-embedding-3-small
);
```

```python
# Query nearest neighbours in SQL
SELECT text, source_url, source_title,
       1 - (embedding <=> %s::vector) AS similarity
FROM documents
ORDER BY embedding <=> %s::vector
LIMIT 10;
```

### Option 4 — Azure AI Search / OpenSearch / Elasticsearch

For enterprise deployments already invested in these platforms, both support vector search natively and add full-text hybrid ranking on top:

```python
# Azure AI Search example
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

client = SearchClient(endpoint, index_name, credential)
vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=10, fields="embedding")
results = client.search(vector_queries=[vector_query])
```

### Summary: local → cloud migration path

```
Local ChromaDB  ──►  Chroma Cloud       (drop-in, same API)
                ──►  Pinecone           (managed, serverless)
                ──►  pgvector           (if you run Postgres)
                ──►  Azure AI Search    (enterprise / hybrid search)
```

The ingestion and query logic in `utils.py` stays completely unchanged — only the `ChromaDBManager.__init__` and the client calls need to be updated.

---

## Use Cases

| Scenario | What to change |
|---|---|
| Different public docs site | `SCRAPE_BASE_URL` + `COLLECTION_NAME` in `.env` |
| Company internal wiki (Confluence, Notion export) | Replace scraper with filesystem loader; rest unchanged |
| Internal docs behind SSO | Add auth headers/cookies to `fetch_page` |
| Multi-source knowledge base | Run ingestion multiple times into one collection, or separate collections per source |
| Production / team deployment | Swap local ChromaDB for Chroma Cloud, Pinecone, or pgvector |
| Switch LLM provider | Update `LLM_MODEL` and the OpenAI client in `ChromaDBManager` / `RAGAssistant` |
