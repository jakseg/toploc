import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

print("Testing IVF Pipeline...")

# ================= 1. Load Components =================

index_path = "data/ivf_index.index"
ivf_index = faiss.read_index(index_path)
print(f"Loaded IVF index: {ivf_index.ntotal} vectors, trained={ivf_index.is_trained}")

with open("data/passage_id_map.json", "r") as f:
    id_map = json.load(f)
print(f"Loaded ID mapping: {len(id_map)} passages")

# Set nprobe (how many clusters to scan). 8 is safe for a small subset.
ivf_index.nprobe = 8
print(f"Set nprobe={ivf_index.nprobe}")

# ================= 2. Load Model & Embed Query =================

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
print("Snowflake model loaded")

test_query = "What was the Chinese Exclusion Acts In the 1850s?"

# CRITICAL: normalize_embeddings=True matches how passages were encoded
query_emb = model.encode(
    [test_query],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")
print(f"Query embedded: shape={query_emb.shape}, dim={query_emb.shape[1]}")

# ================= 3. Run Search =================

k = 5
scores, indices = ivf_index.search(query_emb, k)

# ================= 4. Map & Print Results =================

print(f"\nQuery: '{test_query}'")
print(f"Top {k} retrieved passages:")
for rank in range(k):
    idx = int(indices[0][rank])
    score = float(scores[0][rank])
    passage_id = id_map.get(str(idx), "UNKNOWN_ID")
    print(f"  Rank {rank+1:2d} | ID: {passage_id:<12} | Score: {score:.4f}")

# ================= 5. Sanity Checks =================

assert ivf_index.is_trained, "Index is not trained!"
assert query_emb.shape[1] == ivf_index.d, "Query dimension mismatch!"
assert all(str(i) in id_map for i in indices[0]), "ID mapping incomplete!"

print("\nAll sanity checks passed. Pipeline is working correctly!")