# HNSW Indexing Fundamentals

import httpx
_orig_client = httpx.Client.__init__
_orig_async_client = httpx.AsyncClient.__init__
def _client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_client(self, *args, **kwargs)
def _async_client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_async_client(self, *args, **kwargs)
httpx.Client.__init__ = _client_no_ssl
httpx.AsyncClient.__init__ = _async_client_no_ssl
# ─────────────────────────────────────────────────────────────────────────────

import time
import os
import numpy as np
import random
import openai
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# check this link:
# https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K

os.environ["HF_DATASETS_CACHE"] = "C:/hf_cache"  # short path avoids Windows MAX_PATH

from datasets import load_dataset

from dotenv import load_dotenv
import pathlib
load_dotenv(pathlib.Path(__file__).parent.parent / '.env')


# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

client = QdrantClient(host="localhost", port=6333, timeout=300) # Increase timeout limit depending on your connection speed

# Verify connection
try:
    collections = client.get_collections()
    print(f"Connected to Qdrant Cloud successfully!")
    print(f"Current collections: {len(collections.collections)}")
except Exception as e:
    print(f"Connection failed: {e}")
    print("Check your QDRANT_URL and QDRANT_API_KEY in .env file")

# Load the dataset (this may take a few minutes for first download)
print("Loading DBpedia 100K dataset...")
ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K")
collection_name = "dbpedia_100K"

print("Dataset loaded successfully!")
print(f"Dataset size: {len(ds['train'])} articles")

# Explore the dataset structure
print("\nDataset structure:")
print("Available columns:", ds["train"].column_names)

# Look at a sample entry
sample = ds["train"][0]
print(f"\nSample article:")
print(f"Title: {sample['title']}")
print(f"Text preview: {sample['text'][:200]}...")
print(f"Embedding dimensions: {len(sample['text-embedding-3-large-1536-embedding'])}")

# Delete collection if it exists (for clean restart)
try:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")
except Exception:
    pass  # Collection doesn't exist, which is fine

# Create collection with optimized settings
print(f"Creating collection: {collection_name}")

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1536,  # Matches dataset dims
        distance=models.Distance.COSINE,  # Good for normalized embeddings
    ),
    hnsw_config=models.HnswConfigDiff(
        m=0,  # Bulk load fast: m=0 (build links after ingest).
        ef_construct=100,  # Build quality: used after we set m>0
        full_scan_threshold=10,  # force HNSW instead of full scan
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=10
    ),  # Force indexing even on small sets for demo
    strict_mode_config=models.StrictModeConfig(
        enabled=False,
    ),  # More flexible while testing
)

print(f"Collection '{collection_name}' created successfully!")

# Verify collection settings
collection_info = client.get_collection(collection_name)
print(f"Vector size: {collection_info.config.params.vectors.size}")
print(f"Distance metric: {collection_info.config.params.vectors.distance}")
print(f"HNSW m: {collection_info.config.hnsw_config.m}")

def upload_batch(start_idx, end_idx):
    points = []
    for i in range(start_idx, min(end_idx, total_points)):
        example = ds["train"][i]

        # Get the pre-computed embedding
        embedding = example["text-embedding-3-large-1536-embedding"]

        # Create payload with fields for filtering tests
        payload = {
            "text": example["text"],
            "title": example["title"],
            "_id": example["_id"],
            "length": len(example["text"]),
            "has_numbers": any(char.isdigit() for char in example["text"]),
        }

        points.append(models.PointStruct(id=i, vector=embedding, payload=payload))

    if points:
        client.upload_points(
            collection_name=collection_name, 
            points=points,
            parallel=1,  # parallel>1 uses multiprocessing which requires __main__ guard on Windows
            )
        return len(points)
    return 0


batch_size = 64 * 10  # Reduce to 32 if connection times out
total_points = len(ds["train"])
print(f"Uploading {total_points} points in batches of {batch_size}")

# Upload all batches with progress tracking
total_uploaded = 0
for i in tqdm(range(0, total_points, batch_size), desc="Uploading points"):
    uploaded = upload_batch(i, i + batch_size)
    total_uploaded += uploaded

print(f"Upload completed! Total points uploaded: {total_uploaded}")

client.update_collection(
    collection_name=collection_name,
    hnsw_config=models.HnswConfigDiff(
        m=16  # Build HNSW now: m=16 after the bulk load.
    ),
)

print("HNSW indexing enabled with m=16")

def get_query_embedding(text):
    """Generate embedding using the same model as the dataset"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",  # Must match dataset model
            input=text,
            dimensions=1536  # Must match dataset dimensions
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting OpenAI embedding: {e}")
        print("Common issues:")
        print("   - Check your OPENAI_API_KEY in .env file")
        print("   - Ensure you have credits in your OpenAI account")
        print("   - Verify your API key has embedding permissions")
        print("Using random vector as fallback for demo purposes...")
        import numpy as np
        return np.random.normal(0, 1, 1536).tolist()

# Test embedding generation
print("Generating query embedding...")
test_query = "artificial intelligence"
query_embedding = get_query_embedding(test_query)
print(f"Generated embedding for: '{test_query}'")
print(f"Embedding dimensions: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")

print("Running baseline performance test...")

# Warm up the RAM index/vectors cache with a test query
client.query_points(collection_name=collection_name, query=query_embedding, limit=1)

# Measure vector search performance
search_times = []
for _ in range(25):  # Multiple runs for a stable average
    start_time = time.time()
    response = client.query_points(
        collection_name=collection_name, query=query_embedding, limit=10
    )
    search_time = (time.time() - start_time) * 1000
    search_times.append(search_time)

baseline_time = sum(search_times) / len(search_times)

print(f"Average search time: {baseline_time:.2f}ms")
print(f"Search times: {[f'{t:.2f}ms' for t in search_times]}")
print(f"Found {len(response.points)} results")
print(
    f"Top result: '{response.points[0].payload['title']}' (score: {response.points[0].score:.4f})"
)

# Show a few more results for context
print(f"\nTop 3 results:")
for i, point in enumerate(response.points[:3], 1):
    title = point.payload["title"]
    score = point.score
    text_preview = point.payload["text"][:100] + "..."
    print(f"   {i}. {title} (score: {score:.4f})")
    print(f"      {text_preview}")

print("Testing filtering without payload indexes")

# Warning: We enable unindexed_filtering_retrieve only for demonstration purposes. In production, don’t use it.
# Demo only: allow filtering without an index by scanning. Turn this off later.
client.update_collection(
    collection_name=collection_name,
    strict_mode_config=models.StrictModeConfig(unindexed_filtering_retrieve=True),
)

# Create a text-based filter
text_filter = models.Filter(
    must=[models.FieldCondition(key="text", match=models.MatchText(text="data"))]
)

# Warmup
client.query_points(collection_name=collection_name, query=query_embedding, limit=1)

# Run multiple times for more reliable measurement
unindexed_times = []
for i in range(10):
    time.sleep(2)  # Small delay to avoid noise; demo only
    start_time = time.time()
    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=10,
        search_params=models.SearchParams(hnsw_ef=100),
        query_filter=text_filter,
    )
    unindexed_times.append((time.time() - start_time) * 1000)

unindexed_filter_time = sum(unindexed_times) / len(unindexed_times)

print(f"Filtered search (WITHOUT index): {unindexed_filter_time:.2f}ms")
print(f"Individual times: {[f'{t:.2f}ms' for t in unindexed_times]}")
print(f"Overhead vs baseline: {unindexed_filter_time - baseline_time:.2f}ms")
print(f"Found {len(response.points)} matching results")
if response.points:
    print(
        f"Top result: '{response.points[0].payload['text']}'\nScore: {response.points[0].score:.4f}"
    )
else:
    print("No results found - try a different filter term")

# Create a payload index for 'text' so filters use an index, not a scan.
client.create_payload_index(
    collection_name=collection_name,
    field_name="text",
    wait=True,
    field_schema=models.TextIndexParams(
        type="text", tokenizer="word", phrase_matching=False
    ),
)

client.update_collection(
    collection_name=collection_name,
    hnsw_config=models.HnswConfigDiff(
        ef_construct=101
    ),  # Added payload index after HNSW; bump ef_construct (+1) to rebuild with filter data.
    strict_mode_config=models.StrictModeConfig(unindexed_filtering_retrieve=False),
)

print("Payload index created for 'text' field")