# vector search fundmaentals

from sentence_transformers import SentenceTransformer

import httpx
_orig_client = httpx.Client.__init__
_orig_async_client = httpx.AsyncClient.__init__
def _client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_client(self, *args, **kwargs)
def _async_client_no_ssl(self, *args, **kwargs): kwargs["verify"] = False; _orig_async_client(self, *args, **kwargs)
httpx.Client.__init__ = _client_no_ssl
httpx.AsyncClient.__init__ = _async_client_no_ssl
# ─────────────────────────────────────────────────────────────────────────────

from qdrant_client import QdrantClient, models
import os

client = QdrantClient(host="localhost", port=6333)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Example: Recipe collection
my_dataset = [
    {
        "title": "Classic Beef Bourguignon",
        "description": """A rich, wine-braised beef stew from Burgundy, France. 
        Tender chunks of beef are slowly simmered with pearl onions, mushrooms, 
        and bacon in a deep red wine sauce. The long, slow cooking process 
        develops complex flavors and creates a luxurious, velvety texture. 
        Perfect for cold winter evenings when you want something hearty and 
        comforting. Traditionally served with crusty bread or creamy mashed 
        potatoes to soak up the incredible sauce.""",
        "cuisine": "French",
        "difficulty": "Intermediate",
        "time": "3 hours"
    },
    {
        "title": "Chicken Tikka Masala",
        "description": """A beloved Indian-inspired dish featuring tender pieces of marinated, 
        grilled chicken bathed in a luscious, creamy tomato-based sauce fragrant with 
        cumin, coriander, garam masala, and fresh ginger. The chicken is first charred 
        in a tandoor-style process that seals in the juices before being finished in the 
        velvety sauce. Served over fluffy basmati rice or with warm naan bread, it is one 
        of the most popular dishes worldwide for good reason.""",
        "cuisine": "Indian",
        "difficulty": "Intermediate",
        "time": "1.5 hours"
    },
    {
        "title": "Ramen Tonkotsu",
        "description": """A soul-warming Japanese noodle soup built on a rich, milky pork bone 
        broth simmered for 12 or more hours until deeply creamy and intensely flavored. 
        Springy wheat noodles sit beneath a perfectly soft-boiled marinated egg, slices of 
        melt-in-your-mouth chashu pork belly, nori, bamboo shoots, and a drizzle of black 
        garlic oil. Every component is crafted separately and assembled to order, resulting 
        in a bowl that is at once comforting and complex.""",
        "cuisine": "Japanese",
        "difficulty": "Advanced",
        "time": "14 hours"
    },
    {
        "title": "Tacos al Pastor",
        "description": """A street-food icon from Mexico City, al pastor features thin slices 
        of pork marinated in dried chiles, achiote, pineapple juice, and warm spices, then 
        stacked on a vertical spit and slow-roasted until caramelized at the edges. Served 
        on small hand-pressed corn tortillas and topped with fresh pineapple chunks, 
        cilantro, diced white onion, and a squeeze of lime. The interplay of smoky, 
        sweet, and bright flavors makes every bite irresistible.""",
        "cuisine": "Mexican",
        "difficulty": "Intermediate",
        "time": "2 hours"
    },
    {
        "title": "Pad Thai",
        "description": """Thailand's most iconic stir-fried noodle dish combines rice noodles 
        with shrimp or chicken, egg, firm tofu, bean sprouts, and fragrant green onions in 
        a glossy tamarind-fish sauce glaze balanced with a touch of palm sugar. Finished 
        tableside with a squeeze of fresh lime, crushed roasted peanuts, and dried chili 
        flakes, the dish is a masterclass in sweet, sour, salty, and umami harmony. Quick 
        to cook but deeply satisfying, it is a staple of Thai households and night markets alike.""",
        "cuisine": "Thai",
        "difficulty": "Beginner",
        "time": "30 minutes"
    },
    {
        "title": "Moroccan Lamb Tagine",
        "description": """A fragrant North African slow-braise featuring bone-in lamb shoulder 
        simmered with saffron, preserved lemons, green olives, and a warming blend of ras 
        el hanout spices including cinnamon, cumin, and turmeric. The conical earthenware 
        tagine traps steam and bastes the meat continuously, resulting in impossibly tender 
        lamb that falls off the bone in silky, spice-steeped juices. Served alongside 
        fluffy couscous studded with toasted almonds and plump raisins.""",
        "cuisine": "Moroccan",
        "difficulty": "Intermediate",
        "time": "2.5 hours"
    },
    {
        "title": "Spaghetti Carbonara",
        "description": """Rome's most celebrated pasta is deceptively simple: al dente spaghetti 
        tossed with crispy guanciale, freshly cracked black pepper, a generous handful of 
        Pecorino Romano, and an emulsified sauce made entirely from egg yolks. No cream is 
        ever added — the silky texture comes solely from the technique of tempering eggs 
        with hot pasta water. When executed correctly, carbonara coats every strand in a 
        rich, glossy sauce that is simultaneously indulgent and elegant.""",
        "cuisine": "Italian",
        "difficulty": "Intermediate",
        "time": "30 minutes"
    },
    {
        "title": "Greek Moussaka",
        "description": """A hearty Mediterranean casserole layered with slices of roasted 
        eggplant, a spiced ground lamb and tomato ragù scented with cinnamon and allspice, 
        and topped with a thick, cloud-like béchamel sauce baked to a golden crust. Each 
        layer absorbs the flavors of the others during baking, creating a cohesive dish 
        that is both rustic and refined. Best made a day ahead so the layers settle and 
        intensify, then reheated and served in generous square portions.""",
        "cuisine": "Greek",
        "difficulty": "Intermediate",
        "time": "2 hours"
    },
    {
        "title": "Peking Duck",
        "description": """A centuries-old imperial Chinese delicacy involving a whole duck 
        that is air-dried for 24 hours to achieve paper-thin, lacquered skin, then roasted 
        in a wood-fired oven until the skin crackles like glass while the meat stays 
        succulent. At the table, the crispy skin is carved tableside, wrapped in delicate 
        steamed pancakes with julienned cucumber, scallions, and a swipe of rich hoisin 
        sauce. The contrast of textures and the depth of flavor make it a true culinary spectacle.""",
        "cuisine": "Chinese",
        "difficulty": "Advanced",
        "time": "26 hours"
    },
    {
        "title": "Shakshuka",
        "description": """A vibrant Middle Eastern and North African breakfast staple of eggs 
        gently poached directly in a bubbling, boldly spiced tomato and pepper sauce seasoned 
        with cumin, smoked paprika, and harissa. The egg whites set while the yolks remain 
        gloriously runny, ready to be broken and swirled into the sauce. Scattered with 
        crumbled feta, fresh parsley, and a drizzle of good olive oil, it is served straight 
        from the skillet with crusty sourdough to mop up every last drop.""",
        "cuisine": "Middle Eastern",
        "difficulty": "Beginner",
        "time": "25 minutes"
    },
    {
        "title": "Lobster Bisque",
        "description": """An elegant French seafood soup that extracts every ounce of flavor 
        from lobster shells, roasting them with mirepoix and tomato paste before simmering 
        in cognac, dry sherry, and fish stock. The strained broth is enriched with heavy 
        cream until it reaches a lush, coral-colored velvet consistency. Finished with a 
        swirl of tarragon butter and tender chunks of poached lobster meat, this bisque 
        transforms humble shells into a luxurious first course worthy of the finest table.""",
        "cuisine": "French",
        "difficulty": "Advanced",
        "time": "2 hours"
    },
]

def fixed_size_chunks(text, chunk_size=100, overlap=20):
    """Split text into fixed-size chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # Only add non-empty chunks
            chunks.append(' '.join(chunk_words))
    
    return chunks

def sentence_chunks(text, max_sentences=3):
    """Group sentences into chunks"""
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        if chunk_sentences:
            chunks.append('. '.join(chunk_sentences) + '.')
    
    return chunks

def paragraph_chunks(text):
    """Split by paragraphs or double line breaks"""
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    return chunks if chunks else [text]  # Fallback to full text

collection_name = "recipes_semantic_search"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

# Create a collection with three named vectors
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "fixed": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "sentence": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "paragraph": models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
)

# Index fields for filtering (more on this on day 2)
client.create_payload_index(
    collection_name=collection_name,
    field_name="chunk_strategy",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# Process and upload data
points = []
point_id = 0

for item in my_dataset:
    description = item["description"]

    # Process with each chunking strategy
    strategies = {
        "fixed": fixed_size_chunks(description),
        "sentence": sentence_chunks(description),
        "paragraph": paragraph_chunks(description),
    }

    for strategy_name, chunks in strategies.items():
        for chunk_idx, chunk in enumerate(chunks):
            # Create vectors for this chunk
            vectors = {strategy_name: encoder.encode(chunk).tolist()}

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload={
                        **item,  # Include all original metadata
                        "chunk": chunk,
                        "chunk_strategy": strategy_name,
                        "chunk_index": chunk_idx,
                    },
                )
            )
            point_id += 1

client.upload_points(collection_name=collection_name, points=points)
print(f"Uploaded {len(points)} chunks across three strategies")

def compare_search_results(query):
    """Compare search results across all chunking strategies"""
    print(f"{'=' * 55}\n")
    print(f"Query: '{query}'\n")

    for strategy in ["fixed", "sentence", "paragraph"]:
        results = client.query_points(
            collection_name=collection_name,
            query=encoder.encode(query).tolist(),
            using=strategy,
            limit=3,
        )

        print(f"--- {strategy.upper()} CHUNKING ---")
        for i, point in enumerate(results.points, 1):
            print(f"{i}. {point.payload['title']}")
            print(f"   Score: {point.score:.3f}")
            print(f"   Chunk: {point.payload['chunk'][:80]}...")
        print()


# Test with domain-specific queries
test_queries = [
    "comfort food for winter",
    "quick and easy beginner recipe",
    "elegant dish for a dinner party",
    "spicy street food",
    "slow-cooked meat with rich sauce",
    "noodle soup",
    "Mediterranean casserole",
    "breakfast with eggs",
]

for query in test_queries:
    compare_search_results(query)

def analyze_chunking_effectiveness():
    """Analyze which chunking strategy works best for your domain"""

    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 40)

    # Get chunk statistics for each strategy
    for strategy in ["fixed", "sentence", "paragraph"]:
        # Count chunks per strategy
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_strategy", match=models.MatchValue(value=strategy)
                    )
                ]
            ),
            limit=100,
        )

        chunks = results[0]
        chunk_sizes = [len(chunk.payload["chunk"]) for chunk in chunks]

        print(f"\n{strategy.upper()} STRATEGY:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
        print(f"  Size range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")


# analyze_chunking_effectiveness()