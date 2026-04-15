# Qdrant AI Learning Project

A hands-on, day-by-day progression through **vector search** and **semantic retrieval** using [Qdrant](https://qdrant.tech/) — from basic 4D toy vectors to a production-grade 100K-document search engine.

---

## Project Structure

```
Qdrant/
├── Day 1/
│   ├── project-1.py       # Vector search fundamentals (4D product categories)
│   └── README.md
├── Day 2/
│   ├── project-1.py       # Semantic movie search with sentence embeddings
│   ├── project-2.py       # Text chunking strategies for recipe search
│   └── README.md
├── Day 3/
│   ├── project-1.py       # HNSW indexing with payload filters (1K products)
│   ├── project-2.py       # Large-scale HNSW + OpenAI embeddings (100K DBpedia)
│   └── README.md
├── Final Project/
│   ├── project-1.py       # HNSW benchmark across 4 configs on 5K IT tickets
│   └── README.md
├── requirements.txt
└── README.md              ← You are here
```

---

## Learning Progression

### Day 1 — Vector Search Fundamentals
**File:** [Day 1/project-1.py](Day%201/project-1.py)

The entry point. Understand what a vector database actually does by encoding 10 product categories as hand-crafted 4-dimensional vectors (`[affordability, quality, popularity, innovation]`). Covers:
- Creating a Qdrant collection and inserting `PointStruct` objects
- Running basic cosine similarity search
- Filtered search by payload field value and numeric range

> See [Day 1/README.md](Day%201/README.md) for full details.

---

### Day 2 — Semantic Search & Text Chunking

#### Project 1 — Semantic Movie Search
**File:** [Day 2/project-1.py](Day%202/project-1.py)

Replace hand-crafted vectors with real language embeddings via `sentence-transformers/all-MiniLM-L6-v2` (384 dims). Store 5 sci-fi movie descriptions and query them with free-form natural language (e.g. `"AI and robots"`).

#### Project 2 — Text Chunking Strategies
**File:** [Day 2/project-2.py](Day%202/project-2.py)

Explore how long text is split before embedding affects retrieval quality. Three chunking strategies (fixed-size, sentence-based, paragraph-based) are stored as **named multi-vectors** in a single collection and compared query-by-query on 11 international recipes.

> See [Day 2/README.md](Day%202/README.md) for full details.

---

### Day 3 — HNSW Indexing & Large-Scale Performance

#### Project 1 — HNSW with Payload Indexes
**File:** [Day 3/project-1.py](Day%203/project-1.py)

Production patterns for a 1,000-item store catalog. Focus on payload index creation (keyword + float), compound multi-condition filters, and HNSW `ef` parameter tuning.

#### Project 2 — Large-Scale Pipeline (100K DBpedia)
**File:** [Day 3/project-2.py](Day%203/project-2.py)

End-to-end large-scale pipeline: load a 100K HuggingFace dataset of pre-computed OpenAI embeddings, bulk-upload with `m=0` HNSW setting for speed, flip to `m=16` for production quality, then benchmark search latency with and without a text payload index.

> See [Day 3/README.md](Day%203/README.md) for full details.

---

### Final Project — HNSW Performance Optimization Benchmark
**File:** [Final Project/project-1.py](Final%20Project/project-1.py)

Puts everything together in a real-world benchmark using 5,000 IT helpdesk tickets from HuggingFace. Encodes ticket content with `all-MiniLM-L6-v2`, uploads to four collections with different HNSW parameters (`m` = 0 / 8 / 16 / 32), and measures upload time, search latency at multiple `hnsw_ef` values, and the speedup from adding a payload index on a computed `length` field. Also prints actual semantic search results for a configurable query.

> See [Final Project/README.md](Final%20Project/README.md) for full details.

---

## Prerequisites

### 1. Qdrant (Local)
All projects connect to a local Qdrant instance on `localhost:6333`.

A [`docker-compose.yml`](docker-compose.yml) is included so you don't have to remember the full `docker run` command. With Docker Desktop running, just do:

```bash
# Start Qdrant in the background
docker compose up -d

# Stop it when you're done
docker compose down
```

Your collections are **persisted** in a Docker volume (`qdrant_data`), so data survives restarts.  
To wipe everything and start fresh: `docker compose down -v`

### 2. Python Environment
Python 3.9+ recommended. Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file at the project root (required only for Day 3 / project-2):

```env
OPENAI_API_KEY=sk-...
```

---

## Dependencies

| Package | Version | Used In |
|---------|---------|---------|
| `qdrant-client` | latest | All projects |
| `python-dotenv` | latest | Day 1 p1, Day 3 p2 |
| `sentence-transformers` | latest | Day 2 p1 & p2, Final Project |
| `httpx` | latest | Day 2 p1 & p2, Day 3 p1 & p2, Final Project |
| `numpy` | latest | Day 3 p1 & p2, Final Project |
| `openai` | latest | Day 3 p2 |
| `datasets` | latest | Day 3 p2 (HuggingFace), Final Project |
| `tqdm` | latest | Day 3 p2 |

Full pinned dependency list: [requirements.txt](requirements.txt)

---

## Key Concepts by Day

| Concept | Day 1 | Day 2 | Day 3 | Final Project |
|---------|-------|-------|-------|---------------|
| Vector collections | ✅ | ✅ | ✅ | ✅ |
| Cosine similarity search | ✅ | ✅ | ✅ | ✅ |
| Payload filtering | ✅ | ✅ | ✅ | ✅ |
| Sentence embeddings | — | ✅ | — | ✅ |
| Named multi-vectors | — | ✅ | — | — |
| Text chunking strategies | — | ✅ | — | — |
| Payload indexes | — | ✅ | ✅ | ✅ |
| HNSW configuration | — | — | ✅ | ✅ |
| Bulk upload optimization | — | — | ✅ | ✅ |
| OpenAI embeddings API | — | — | ✅ | — |
| Performance benchmarking | — | — | ✅ | ✅ |
| Large-scale dataset (100K) | — | — | ✅ | — |
| Multi-config comparison | — | — | — | ✅ |
| Real-world dataset (HuggingFace) | — | — | ✅ | ✅ |

---

## Qdrant Resources
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [DBpedia 100K Dataset on HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K)
