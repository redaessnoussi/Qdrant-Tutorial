# Final Project — Qdrant Performance Optimization Benchmark

A production-oriented benchmark that measures **upload speed**, **search latency**, and **payload filtering performance** across four HNSW index configurations using a real IT support ticket dataset.

---

## What It Does

1. **Loads** 5,000 IT helpdesk tickets from the [`KameronB/synthetic-it-callcenter-tickets`](https://huggingface.co/datasets/KameronB/synthetic-it-callcenter-tickets) HuggingFace dataset
2. **Encodes** ticket `content` fields into 384-dimensional vectors using `all-MiniLM-L6-v2`
3. **Creates 4 Qdrant collections**, each with a different HNSW configuration
4. **Benchmarks upload time** for each configuration
5. **Benchmarks search latency** at multiple `hnsw_ef` probe values (64, 128, 256)
6. **Measures payload filtering impact** — compares query time with and without a payload index on the `length` field
7. **Displays actual search results** for a configurable query (default: `"PostgreSQL upgrade"`)

---

## HNSW Configurations Tested

| Config Name | `m` | `ef_construct` | Trade-off |
|---|---|---|---|
| `fast_initial_upload` | 0 | 100 | No HNSW graph — fastest ingestion, full scan only |
| `memory_optimized` | 8 | 100 | Low RAM usage, lower recall |
| `balanced` | 16 | 200 | Recommended default — good recall and speed |
| `high_quality` | 32 | 400 | Best recall, slower build and more RAM |

- **`m`**: number of edges per node in the HNSW graph. Higher = better recall, more memory.
- **`ef_construct`**: size of the candidate list during index build. Higher = better recall, slower build.

---

## Dataset

**[KameronB/synthetic-it-callcenter-tickets](https://huggingface.co/datasets/KameronB/synthetic-it-callcenter-tickets)**

27,602 synthetic IT support tickets. Each record includes:

| Field | Description |
|---|---|
| `content` | Full ticket body (embedded as the search vector) |
| `short_description` | One-line summary |
| `category` / `subcategory` | Ticket classification (e.g. `SOFTWARE / INSTALLATION`) |
| `issue/request` | Parsed issue label |
| `software/system` | Affected system |
| `priority` | Ticket priority |
| `resolution_time` | Time to resolve (minutes) |

The script uses the first **5,000 records** for a faster benchmark. Increase `range(5000)` for more accurate results.

---

## Payload Fields Stored in Qdrant

In addition to all raw dataset fields, each point gets three computed payload fields:

| Field | Type | Description |
|---|---|---|
| `length` | integer | Character count of `content` |
| `word_count` | integer | Word count of `content` |
| `has_keywords` | boolean | Whether `content` contains "important", "key", or "main" |

---

## Search Query

The benchmark and result display use:

```python
test_query = "PostgreSQL upgrade"
```

Change this to any natural language query relevant to your use case. The top 5 semantically similar tickets are printed with their score, short description, and content preview.

---

## Output Explained

```
Top 5 results for: PostgreSQL upgrade
  Score: 0.9123
  Short desc: Request for PostgreSQL upgrade to the latest version.
  Content: I would like to request an upgrade for our PostgreSQL...

============================================================
PERFORMANCE OPTIMIZATION RESULTS
============================================================

1) Upload Performance:
   fast_initial_upload: 3.66s
   memory_optimized: 2.95s
   balanced: 3.26s
   high_quality: 2.98s

2) Search Performance (hnsw_ef=128):
   memory_optimized: 9.86ms
   balanced: 9.45ms
   high_quality: 10.57ms

3) Filtering Impact:
   Without index: 67.57ms
   With index: 8.79ms
   Speedup: 7.7x
```

- **Upload Performance**: time to upload 5,000 pre-encoded points. Note: encoding time (~80s) is excluded since it's the same across all configs.
- **Search Performance**: average of 25 queries at `hnsw_ef=128`. Lower is better.
- **Filtering Impact**: speedup from adding a payload index on `length` before executing filtered queries.

> `fast_initial_upload` (`m=0`) is excluded from search benchmarks — it has no HNSW graph and would fall back to a full scan.

---

## How to Run

### 1. Start Qdrant
From the project root:
```bash
docker compose up -d
```

### 2. Activate the virtual environment
```bash
# Windows
.venv\Scripts\activate
```

### 3. Run the script
```bash
cd "Final Project"
py project-1.py
```

> First run downloads the dataset (~36 MB) and caches it at `C:/hf_cache`.  
> Subsequent runs use the cache and skip the download.

---

## Key Takeaways

- **Payload indexes** provide the largest measurable speedup (~7x or more on filtered queries)
- **`m=16, ef_construct=200`** (balanced) is a reliable default for most production use cases
- **`hnsw_ef` at query time** lets you tune the recall/latency trade-off without rebuilding the index
- Pre-encoding all vectors before upload is faster than encoding per-batch during upload
