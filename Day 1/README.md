# Day 1 — Vector Search Fundamentals

## Project 1: Product Category Similarity Search

### Overview
Introduces the core concepts of vector search using Qdrant. A small 4-dimensional vector space is used to represent product categories, demonstrating how numeric vectors can encode semantic meaning and enable similarity-based retrieval.

### What It Does
1. **Creates a Qdrant collection** called `product_categories` with 4D vectors using **cosine similarity**.
2. **Inserts 10 product categories**, each represented by a vector of 4 attributes:

   | Dimension | Meaning |
   |-----------|---------|
   | `[0]` | Affordability (0 = expensive → 1 = cheap) |
   | `[1]` | Quality (0 = low → 1 = premium) |
   | `[2]` | Popularity (0 = niche → 1 = popular) |
   | `[3]` | Innovation (0 = traditional → 1 = cutting-edge) |

3. **Runs three types of queries**:

   | Search Type | Description |
   |-------------|-------------|
   | Basic similarity | Find the top 3 most similar categories to a query vector |
   | Filtered by segment | Same query, restricted to `"tech"` segment only |
   | Filtered by price | Query for affordable items with `avg_price_usd < 100` |

### Dataset — Sample Points

| ID | Name | Segment | Avg Price (USD) |
|----|------|---------|----------------|
| 1 | Electronics | tech | $350 |
| 2 | Books | education | $20 |
| 5 | Groceries | food | $15 |
| 6 | Luxury Watches | fashion | $5000 |
| 7 | Smartphones | tech | $700 |

### Key Qdrant Concepts Covered
- `create_collection` with `VectorParams` (size, distance)
- `PointStruct` with payload metadata
- `upsert` to insert points
- `query_points` for similarity search
- `Filter`, `FieldCondition`, `MatchValue`, `Range` for filtered search

### Dependencies
```
qdrant-client
python-dotenv
```

### How to Run
```bash
# Ensure Qdrant is running locally on port 6333
python "Day 1/project-1.py"
```
