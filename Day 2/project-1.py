# vector search fundmaentals

import os
import ssl

# ── SSL bypass for corporate inspection proxies ──────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import httpx
_orig_client = httpx.Client.__init__
_orig_async_client = httpx.AsyncClient.__init__
def _client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_client(self, *args, **kwargs)
def _async_client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_async_client(self, *args, **kwargs)
httpx.Client.__init__ = _client_no_ssl
httpx.AsyncClient.__init__ = _async_client_no_ssl
# ─────────────────────────────────────────────────────────────────────────────

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load embedding model
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 2. Connect to Qdrant
# -----------------------------
client = QdrantClient(host="localhost", port=6333)

collection_name = "movies_semantic"

# Recreate collection
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=384,  # embedding size
        distance=models.Distance.COSINE,
    ),
)

print("Collection created.\n")

# -----------------------------
# 3. Dataset (simplified)
# -----------------------------
movies = [
    {
        "name": "The Matrix",
        "year": 1999,
        "description": "A hacker discovers reality is a simulation and fights machines.",
    },
    {
        "name": "Interstellar",
        "year": 2014,
        "description": "A team travels through space and time to save humanity.",
    },
    {
        "name": "Ex Machina",
        "year": 2014,
        "description": "A programmer interacts with an advanced AI robot.",
    },
    {
        "name": "Arrival",
        "year": 2016,
        "description": "A linguist communicates with aliens to understand their language.",
    },
    {
        "name": "Blade Runner",
        "year": 1982,
        "description": "A detective hunts androids in a futuristic dystopian world.",
    },
]

# -----------------------------
# 4. Convert to vectors
# -----------------------------
points = []

for idx, movie in enumerate(movies):
    vector = model.encode(movie["description"]).tolist()

    points.append(
        models.PointStruct(
            id=idx,
            vector=vector,
            payload=movie,
        )
    )

# Upload to Qdrant
client.upsert(collection_name=collection_name, points=points)

print(f"Inserted {len(points)} movies.\n")

# -----------------------------
# 5. Search function
# -----------------------------
def search_movies(query):
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
    )

    print(f"\nQuery: {query}\n")

    for hit in results.points:
        print(
            f"Score: {hit.score:.4f} | {hit.payload['name']} ({hit.payload['year']})"
        )

# -----------------------------
# 6. Try searches
# -----------------------------
search_movies("AI and robots")
search_movies("space and time travel")
search_movies("questioning reality")