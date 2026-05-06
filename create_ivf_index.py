import sys
import faiss
import numpy as np
import json
import time

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
data_dir = f"data/{model_name}"

# 1. Load your embeddings
print("Loading embeddings...")
embeddings = np.load(f"{data_dir}/passage_embeddings.npy")
print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

# 2. Load ID map (for later evaluation)
with open(f"{data_dir}/passage_id_map.json", "r") as f:
    id_map = json.load(f)

# 3. IVF Parameters (adjusted for small dataset)
dim = embeddings.shape[1]  # 1024 for Snowflake, 768 for Dragon
num_centroids = 64  # For 2k docs, use 64-128 (paper uses 32768 for 38M docs)
nprobe = 8  # Number of centroids to search (we'll tune this later)

print(f"\nBuilding IVF index with {num_centroids} centroids...")

# 4. Create IVF Index
# IndexFlatIP = Inner Product (works as cosine since we normalized)
quantizer = faiss.IndexFlatIP(dim)
ivf_index = faiss.IndexIVFFlat(quantizer, dim, num_centroids, faiss.METRIC_INNER_PRODUCT)

# 5. Train the index (runs K-Means to find centroids)
print("Training IVF (K-Means clustering)...")
start_time = time.time()
ivf_index.train(embeddings)
print(f"Training completed in {time.time() - start_time:.2f}s")

# 6. Add all embeddings to the index
print("Adding embeddings to index...")
start_time = time.time()
ivf_index.add(embeddings)
print(f"Indexing completed in {time.time() - start_time:.2f}s")

# 7. Set nprobe (how many centroids to search)
ivf_index.nprobe = nprobe
print(f"Set nprobe={nprobe}")

# 8. Save the index
faiss.write_index(ivf_index, f"{data_dir}/ivf_index.index")
print(f"\n IVF index saved to: {data_dir}/ivf_index.index")

# 9. Verify index stats
print(f"\n Index Statistics:")
print(f"  - Total vectors: {ivf_index.ntotal}")
print(f"  - Centroids: {num_centroids}")
print(f"  - Dimension: {dim}")
print(f"  - Is trained: {ivf_index.is_trained}")
