# Day 2 — Semantic Search & Text Chunking Strategies

## Project 1: Semantic Movie Search

### Overview
Moves beyond hand-crafted vectors to **real language embeddings**. Movie descriptions are encoded with a pre-trained sentence transformer and stored in Qdrant, enabling fully natural-language similarity queries.

### What It Does
1. **Loads** the `sentence-transformers/all-MiniLM-L6-v2` model (384-dimensional embeddings).
2. **Creates** a Qdrant collection `movies_semantic` with 384-dim cosine vectors.
3. **Encodes** each movie's text description into a dense vector and upserts it.
4. **Runs semantic queries** — no keyword matching, pure meaning similarity:
   - `"AI and robots"` → Ex Machina, Blade Runner, The Matrix
   - `"space and time travel"` → Interstellar, Arrival
   - `"questioning reality"` → The Matrix, Ex Machina

### Dataset

| Movie | Year | Description Summary |
|-------|------|---------------------|
| The Matrix | 1999 | A hacker discovers reality is a simulation |
| Interstellar | 2014 | Space-time travel to save humanity |
| Ex Machina | 2014 | Programmer interacts with an AI robot |
| Arrival | 2016 | Linguist communicates with aliens |
| Blade Runner | 1982 | Detective hunts androids in dystopia |

### Key Concepts Covered
- SentenceTransformer model loading and encoding
- Real embedding vectors (384-dim) vs hand-crafted vectors
- Semantic similarity queries in natural language
- Corporate proxy SSL bypass via `httpx` monkey-patch

### Dependencies
```
qdrant-client
sentence-transformers
httpx
```

### How to Run
```bash
python "Day 2/project-1.py"
```

---

## Project 2: Text Chunking Strategies for Recipe Search

### Overview
Explores how chunking long text documents before embedding affects retrieval quality. Three strategies — **fixed-size**, **sentence-based**, and **paragraph-based** — are compared side-by-side using Qdrant's **named vectors** feature.

### What It Does
1. **Builds a recipe dataset** of 11 international dishes with rich, multi-paragraph descriptions.
2. **Applies three chunking strategies** to each recipe description:

   | Strategy | Method |
   |----------|--------|
   | `fixed` | Split into 100-word windows with 20-word overlap |
   | `sentence` | Group sentences in batches of 3 |
   | `paragraph` | Split on double line breaks |

3. **Creates a single collection** `recipes_semantic_search` with **three named vector spaces** (`fixed`, `sentence`, `paragraph`), each with its own embeddings.
4. **Indexes** `chunk_strategy` as a payload keyword index for efficient filtering.
5. **Runs 8 natural-language queries** and compares top-3 results per chunking strategy.
6. Includes an optional `analyze_chunking_effectiveness()` function to report chunk count, average size, and size range per strategy.

### Dataset — Recipe Highlights

| Recipe | Cuisine | Difficulty | Time |
|--------|---------|------------|------|
| Beef Bourguignon | French | Intermediate | 3 hours |
| Ramen Tonkotsu | Japanese | Advanced | 14 hours |
| Pad Thai | Thai | Beginner | 30 min |
| Peking Duck | Chinese | Advanced | 26 hours |
| Shakshuka | Middle Eastern | Beginner | 25 min |

### Key Concepts Covered
- **Named multi-vectors** in a single Qdrant collection
- Fixed-size, sentence, and paragraph chunking algorithms
- `using` parameter in `query_points` to target a specific vector space
- `scroll` API to enumerate collection points
- Payload index creation for keyword fields

### Dependencies
```
qdrant-client
sentence-transformers
httpx
```

### How to Run
```bash
python "Day 2/project-2.py"
```
