import os
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

os.environ["HF_DATASETS_CACHE"] = "C:/hf_cache"  # short path avoids Windows MAX_PATH
from datasets import load_dataset
import time
import numpy as np

# Initialize the Qdrant client
client = QdrantClient(host="localhost", port=6333)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Test configurations
configs = [
    {"name": "fast_initial_upload", "m": 0, "ef_construct": 100},  # m=0 = ingest-only
    {"name": "memory_optimized", "m": 8, "ef_construct": 100},  # m=8 = lower RAM
    {"name": "balanced", "m": 16, "ef_construct": 200},  # m=16 = balanced
    {"name": "high_quality", "m": 32, "ef_construct": 400},  # m=32 = higher recall, slower build
]

# Load dataset
# Check https://huggingface.co/datasets/KameronB/synthetic-it-callcenter-tickets
dataset = load_dataset("KameronB/synthetic-it-callcenter-tickets")

# Use train split (main data)
raw_data = dataset["train"]

print(raw_data[0])

# Create Multiple Test Collections

for config in configs:
    collection_name = f"my_domain_{config['name']}"
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(
            m=config["m"],
            ef_construct=config["ef_construct"],
            full_scan_threshold=10,  # force HNSW instead of full scan
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=10
        ),  # Force indexing even on small sets for demo
    )
    print(f"Created collection: {collection_name}")


# Upload and Time


def upload_with_timing(collection_name, data, config_name):
    embeddings = encoder.encode([d["content"] for d in data], show_progress_bar=True).tolist()

    points = []
    for i, item in enumerate(data):
        embedding = embeddings[i]

        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    **item,
                    "length": len(item["content"]),
                    "word_count": len(item["content"].split()),
                    "has_keywords": any(
                        keyword in item["content"].lower() for keyword in ["important", "key", "main"]
                    ),
                },
            )
        )

    # Warmup
    client.query_points(collection_name=collection_name, query=points[0].vector, limit=1)

    start_time = time.time()
    client.upload_points(collection_name=collection_name, points=points)
    upload_time = time.time() - start_time

    print(f"{config_name}: Uploaded {len(points)} points in {upload_time:.2f}s")
    return upload_time


# Load your dataset here. The larger the dataset, the more accurate the benchmark will be.
# your_dataset = [{"description": "This is a description of a product"}, ...]

your_dataset = [item for item in raw_data.select(range(5000)) if item.get("content")]
print(f"Loaded {len(your_dataset)} usable records")

# Upload to each collection
upload_times = {}
for config in configs:
    collection_name = f"my_domain_{config['name']}"
    upload_times[config["name"]] = upload_with_timing(collection_name, your_dataset, config["name"])


def wait_for_indexing(collection_name, timeout=60, poll_interval=1):
    print(f"Waiting for collection '{collection_name}' to be indexed...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        info = client.get_collection(collection_name=collection_name)

        if info.indexed_vectors_count > 0 and info.status == models.CollectionStatus.GREEN:
            print(f"Success! Collection '{collection_name}' is indexed and ready.")
            print(f" - Status: {info.status.value}")
            print(f" - Indexed vectors: {info.indexed_vectors_count}")
            return

        print(f" - Status: {info.status.value}, Indexed vectors: {info.indexed_vectors_count}. Waiting...")
        time.sleep(poll_interval)

    info = client.get_collection(collection_name=collection_name)
    raise Exception(
        f"Timeout reached after {timeout} seconds. Collection '{collection_name}' is not ready. "
        f"Final status: {info.status.value}, Indexed vectors: {info.indexed_vectors_count}"
    )


for config in configs:
    if config["m"] > 0:  # m=0 has no HNSW to wait for
        collection_name = f"my_domain_{config['name']}"
        wait_for_indexing(collection_name)


# Benchmark Search Performance

def benchmark_search(collection_name, query_embedding, ef_values=[64, 128, 256]):
    # Warmup
    client.query_points(collection_name=collection_name, query=query_embedding, limit=1)

    # hnsw_ef: higher = better recall, but slower. Tune per your latency goal.
    results = {}
    for hnsw_ef in ef_values:
        times = []

        # Run multiple queries for more reliable timing
        for _ in range(25):
            start_time = time.time()

            _ = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=10,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
                with_payload=False,
            )

            times.append((time.time() - start_time) * 1000)

        results[hnsw_ef] = {
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
        }

    return results


test_query = "PostgreSQL upgrade"
query_embedding = encoder.encode(test_query).tolist()

# Show actual search results
results = client.query_points(
    collection_name="my_domain_balanced",
    query=query_embedding,
    limit=5,
    with_payload=True,
)
print("\nTop 5 results for:", test_query)
for hit in results.points:
    print(f"\n  Score: {hit.score:.4f}")
    print(f"  Short desc: {hit.payload.get('short_description')}")
    print(f"  Content: {hit.payload.get('content')[:200]}...")

performance_results = {}
for config in configs:
    if config["m"] > 0:  # Skip m=0 collections for search
        collection_name = f"my_domain_{config['name']}"
        performance_results[config["name"]] = benchmark_search(
            collection_name, query_embedding
        )


# Measure Payload Indexing Impact

def test_filtering_performance(collection_name):
    query_embedding = encoder.encode("PostgreSQL upgrade").tolist()

    # Test filter without index
    filter_condition = models.Filter(
        must=[models.FieldCondition(key="length", range=models.Range(gte=10, lte=200))]
    )

    # Demo only: unindexed_filtering_retrieve=True forces a scan; turn it off right after measuring.
    client.update_collection(
        collection_name=collection_name,
        strict_mode_config=models.StrictModeConfig(unindexed_filtering_retrieve=True),
    )

    # Warmup
    client.query_points(collection_name=collection_name, query=query_embedding, limit=1)

    # Timing without payload index
    times = []
    for _ in range(25):
        start_time = time.time()
        _ = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=filter_condition,
            limit=10,
            with_payload=False,
        )
        times.append((time.time() - start_time) * 1000)
    time_without_index = np.mean(times)

    # Create payload index
    client.create_payload_index(
        collection_name=collection_name,
        field_name="length",
        field_schema=models.PayloadSchemaType.INTEGER,
        wait=True,
    )

    # HNSW was already built; adding the payload index doesn’t rebuild it.
    # Bump ef_construct (+1) once to trigger a safe rebuild.
    base_ef = client.get_collection(
        collection_name=collection_name
    ).config.hnsw_config.ef_construct
    new_ef_construct = base_ef + 1

    client.update_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(ef_construct=new_ef_construct),
        strict_mode_config=models.StrictModeConfig(
            unindexed_filtering_retrieve=False
        ),  # Turn off scanning and use payload index instead.
    )

    wait_for_indexing(collection_name)

    # Warmup
    client.query_points(collection_name=collection_name, query=query_embedding, limit=1)

    # Timing with index
    times = []
    for _ in range(25):
        start_time = time.time()
        _ = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=filter_condition,
            limit=10,
            with_payload=False,
        )
        times.append((time.time() - start_time) * 1000)
    time_with_index = np.mean(times)

    return {
        "without_index": time_without_index,
        "with_index": time_with_index,
        "speedup": time_without_index / time_with_index,
    }


# Test on your best performing collection
best_collection = "my_domain_balanced"  # Choose based on your results
filtering_results = test_filtering_performance(best_collection)


# Analyze Your Results

print("=" * 60)
print("PERFORMANCE OPTIMIZATION RESULTS")
print("=" * 60)

print("\n1) Upload Performance:")
for config_name, time_taken in upload_times.items():
    print(f"   {config_name}: {time_taken:.2f}s")

print("\n2) Search Performance (hnsw_ef=128):")
for config_name, results in performance_results.items():
    if 128 in results:
        print(f"   {config_name}: {results[128]['avg_time']:.2f}ms")

print("\n3) Filtering Impact:")
print(f"   Without index: {filtering_results['without_index']:.2f}ms")
print(f"   With index: {filtering_results['with_index']:.2f}ms")
print(f"   Speedup: {filtering_results['speedup']:.1f}x")