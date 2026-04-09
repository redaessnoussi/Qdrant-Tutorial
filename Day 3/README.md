# Day 3 — HNSW Indexing & Large-Scale Performance

## Project 1: HNSW Indexing with Payload Filters

### Overview
Focuses on production-ready patterns: **payload indexes**, **HNSW search parameters**, and **filtered vector queries** at scale. A synthetic store catalog of 1,000 products is used to demonstrate how filtering efficiently narrows search without sacrificing speed.

### What It Does
1. **Creates** a Qdrant collection `store_demo` with 128-dimensional cosine vectors.
2. **Creates three payload indexes** before inserting data (best practice for performance):
   - `category` — keyword index (e.g. `"laptop"`, `"phone"`, `"tablet"`)
   - `brand` — keyword index (e.g. `"Apple"`, `"Dell"`, `"Samsung"`)
   - `price` — float index for range queries
3. **Generates 1,000 random product points** with realistic category/brand/price payloads using `numpy`.
4. **Runs a compound filtered query** combining:
   - Category must be `"laptop"`
   - Price must be ≤ `1000`
   - Brand must be one of `["Apple", "Dell", "HP"]`
5. Uses `hnsw_ef=128` search parameter to tune recall vs speed.

### Key Concepts Covered
- `create_payload_index` before data ingestion
- `PayloadSchemaType.KEYWORD` and `PayloadSchemaType.FLOAT`
- `MatchValue`, `MatchAny`, `Range` filter conditions
- `SearchParams(hnsw_ef=...)` for HNSW recall tuning
- Efficient compound filtering (must + must + must)

### Dependencies
```
qdrant-client
numpy
```

### How to Run
```bash
python "Day 3/project-1.py"
```

---

## Project 2: Large-Scale HNSW with OpenAI Embeddings & Performance Benchmarking

### Overview
End-to-end large-scale pipeline using a real-world 100K dataset from HuggingFace. Demonstrates **bulk upload optimization**, **HNSW configuration for bulk load**, **text payload indexes**, and **search latency benchmarking** with and without payload indexes.

### What It Does
1. **Loads** the `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K` dataset (100K Wikipedia articles, pre-embedded at 1536 dimensions with OpenAI `text-embedding-3-large`).
2. **Creates** a collection `dbpedia_100K` with:
   - `m=0` initially (fast bulk-load mode — no HNSW links built during ingest)
   - `ef_construct=100`
   - `indexing_threshold=10` (forces HNSW indexing even on small sets)
3. **Uploads 100K points in batches** of 640 with `tqdm` progress bar tracking.
4. **Enables HNSW** after upload by setting `m=16` (standard production quality).
5. **Benchmarks search latency** over 25 runs:
   - Baseline vector search (no filter)
   - Filtered search **without** a payload index (full scan, slow)
6. **Creates a text payload index** on the `text` field with word tokenization.
7. **Re-runs filtered search** with the index active and compares overhead vs baseline.
8. Uses OpenAI API to generate query embeddings (falls back to random vector if API key is missing).

### Performance Concepts

| Phase | Setting | Purpose |
|-------|---------|---------|
| Bulk ingest | `m=0` | Skip HNSW link building → faster uploads |
| After ingest | `m=16` | Build HNSW graph for production recall |
| Without index | `unindexed_filtering_retrieve=True` | Demo scan-based filtering (slow) |
| With index | Text index + `unindexed_filtering_retrieve=False` | Fast filtered search |

### Dataset
- **Source:** [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K)
- **Size:** 100,000 Wikipedia article embeddings
- **Dimensions:** 1,536 (OpenAI `text-embedding-3-large`)

### Key Concepts Covered
- Bulk-load HNSW pattern (`m=0` → `m=16`)
- Batch upload with `upload_points` and `tqdm`
- `StrictModeConfig` for unindexed filtering (demo use only)
- `TextIndexParams` with word tokenizer and phrase matching
- OpenAI embeddings API as query encoder
- Search latency measurement and comparison
- `ef_construct` bump trick after adding a payload index

### Environment Variables Required
```
OPENAI_API_KEY=sk-...
```
_(configured in `.env` at the project root)_

### Dependencies
```
qdrant-client
openai
datasets
tqdm
numpy
python-dotenv
httpx
```

### How to Run
```bash
python "Day 3/project-2.py"
```

> **Note:** First run will download the HuggingFace dataset (~several GB). The cache is stored at `C:/hf_cache` to avoid Windows MAX_PATH issues.
