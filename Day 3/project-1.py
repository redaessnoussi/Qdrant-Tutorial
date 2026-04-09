# HNSW Indexing Fundamentals
import httpx
_orig_client = httpx.Client.__init__
_orig_async_client = httpx.AsyncClient.__init__
def _client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_client(self, *args, **kwargs)
def _async_client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_async_client(self, *args, **kwargs)
httpx.Client.__init__ = _client_no_ssl
httpx.AsyncClient.__init__ = _async_client_no_ssl
# ─────────────────────────────────────────────────────────────────────────────

from qdrant_client import QdrantClient, models
import time
import os
import numpy as np
import random

client = QdrantClient(host="localhost", port=6333)

# =========================
# CONFIG
# =========================
COLLECTION_NAME = "store_demo"
VECTOR_SIZE = 128

# =========================
# RESET COLLECTION
# =========================
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
    ),
)

print("✅ Collection created")

# =========================
# CREATE PAYLOAD INDEXES (IMPORTANT!)
# =========================
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="brand",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="price",
    field_schema=models.PayloadSchemaType.FLOAT,
)

print("✅ Payload indexes created")

# =========================
# GENERATE DATA
# =========================
categories = ["laptop", "phone", "tablet"]
brands = ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "Samsung"]

points = []

for i in range(1000):
    vector = np.random.rand(VECTOR_SIZE).tolist()

    payload = {
        "category": random.choice(categories),
        "brand": random.choice(brands),
        "price": random.randint(300, 2000),
    }

    points.append(models.PointStruct(
        id=i,
        vector=vector,
        payload=payload
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)

print("✅ Data inserted")

# =========================
# QUERY WITH FILTER
# =========================
query_vector = np.random.rand(VECTOR_SIZE).tolist()

filter_conditions = models.Filter(
    must=[
        models.FieldCondition(
            key="category",
            match=models.MatchValue(value="laptop"),
        ),
        models.FieldCondition(
            key="price",
            range=models.Range(lte=1000),
        ),
        models.FieldCondition(
            key="brand",
            match=models.MatchAny(any=["Apple", "Dell", "HP"]),
        ),
    ]
)

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=filter_conditions,
    limit=5,
    search_params=models.SearchParams(hnsw_ef=128),
)

print("\n🔎 Results:")
for r in results.points:
    print(r.payload, "score:", r.score)