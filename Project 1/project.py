from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Vector dimensions (4D — Product Categories)
#   [affordability, quality, popularity, innovation]
#    0 = low / expensive / niche / traditional
#    1 = high / premium / popular / cutting-edge
# ---------------------------------------------------------------------------

client = QdrantClient(host="localhost", port=6333)
collection_name = "product_categories"

# ── 1. Create (or recreate) the collection ──────────────────────────────────
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=4,
        distance=models.Distance.COSINE,
    ),
)
print(f"Collection '{collection_name}' created.\n")

# ── 2. Define points ─────────────────────────────────────────────────────────
#  [affordability, quality, popularity, innovation]
points = [
    models.PointStruct(
        id=1,
        vector=[0.8, 0.7, 0.9, 0.6],
        payload={"name": "Electronics", "segment": "tech", "avg_price_usd": 350},
    ),
    models.PointStruct(
        id=2,
        vector=[0.3, 0.9, 0.4, 0.8],
        payload={"name": "Books", "segment": "education", "avg_price_usd": 20},
    ),
    models.PointStruct(
        id=3,
        vector=[0.5, 0.6, 0.8, 0.4],
        payload={"name": "Clothing", "segment": "fashion", "avg_price_usd": 60},
    ),
    models.PointStruct(
        id=4,
        vector=[0.2, 0.8, 0.3, 0.9],
        payload={"name": "Software", "segment": "tech", "avg_price_usd": 80},
    ),
    models.PointStruct(
        id=5,
        vector=[0.9, 0.5, 0.7, 0.2],
        payload={"name": "Groceries", "segment": "food", "avg_price_usd": 15},
    ),
    models.PointStruct(
        id=6,
        vector=[0.1, 0.9, 0.2, 0.7],
        payload={"name": "Luxury Watches", "segment": "fashion", "avg_price_usd": 5000},
    ),
    models.PointStruct(
        id=7,
        vector=[0.7, 0.6, 0.9, 0.5],
        payload={"name": "Smartphones", "segment": "tech", "avg_price_usd": 700},
    ),
    models.PointStruct(
        id=8,
        vector=[0.6, 0.7, 0.6, 0.3],
        payload={"name": "Home Appliances", "segment": "home", "avg_price_usd": 250},
    ),
    models.PointStruct(
        id=9,
        vector=[0.4, 0.5, 0.5, 0.6],
        payload={"name": "Sports Equipment", "segment": "health", "avg_price_usd": 120},
    ),
    models.PointStruct(
        id=10,
        vector=[0.8, 0.4, 0.8, 0.3],
        payload={"name": "Toys", "segment": "entertainment", "avg_price_usd": 30},
    ),
]

client.upsert(collection_name=collection_name, points=points)
print(f"Inserted {len(points)} points into '{collection_name}'.\n")

# ── 3. Basic similarity search ────────────────────────────────────────────────
# Query: looking for something highly affordable, popular, and innovative → tech-like
query_vector = [0.8, 0.6, 0.9, 0.7]

print("=" * 55)
print("BASIC SIMILARITY SEARCH")
print("Query: [affordability=0.8, quality=0.6, popularity=0.9, innovation=0.7]")
print("=" * 55)

results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3,
)

for hit in results.points:
    print(f"  Score: {hit.score:.4f} | {hit.payload['name']} (${hit.payload['avg_price_usd']})")

# ── 4. Filtered similarity search ────────────────────────────────────────────
# Same query but restricted to the "tech" segment only
print("\n" + "=" * 55)
print("FILTERED SIMILARITY SEARCH  (segment = 'tech')")
print("Query: [affordability=0.8, quality=0.6, popularity=0.9, innovation=0.7]")
print("=" * 55)

filtered_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="segment",
                match=models.MatchValue(value="tech"),
            )
        ]
    ),
    limit=3,
)

for hit in filtered_results.points:
    print(f"  Score: {hit.score:.4f} | {hit.payload['name']} (${hit.payload['avg_price_usd']})")

# ── 5. Filtered search with a price range condition ───────────────────────────
# Find affordable (high affordability vector) items under $100
print("\n" + "=" * 55)
print("FILTERED SIMILARITY SEARCH  (avg_price_usd < 100)")
print("Query: [affordability=0.9, quality=0.5, popularity=0.8, innovation=0.3]")
print("=" * 55)

budget_query = [0.9, 0.5, 0.8, 0.3]

budget_results = client.query_points(
    collection_name=collection_name,
    query=budget_query,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="avg_price_usd",
                range=models.Range(lte=100),
            )
        ]
    ),
    limit=3,
)

for hit in budget_results.points:
    print(f"  Score: {hit.score:.4f} | {hit.payload['name']} (${hit.payload['avg_price_usd']})")

print("\nDone.")
