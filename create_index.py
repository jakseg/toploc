import sys
import faiss
import numpy as np
import json
import time

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
data_dir = f"data/{model_name}"

# 1. Load your embeddings
print("Loading embeddings...")
embeddings = np.load(f"{data_dir}/passage_embeddings.npy")
print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

# 2. Load ID map (for later evaluation)
with open(f"{data_dir}/passage_id_map.json", "r") as f:
    id_map = json.load(f)

dim = embeddings.shape[1]  # 1024 for Snowflake, 768 for Dragon

if index_type == "ivf":
    # 3. IVF Parameters (adjusted for small dataset)
    num_centroids = 64  # For 2k docs, use 64-128 (paper uses 32768 for 38M docs)
    nprobe = 8  # Number of centroids to search (we'll tune this later)

    print(f"\nBuilding IVF index with {num_centroids} centroids...")

    # 4. Create IVF Index
    # IndexFlatIP = Inner Product (works as cosine since we normalized)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, num_centroids, faiss.METRIC_INNER_PRODUCT)

    # 5. Train the index (runs K-Means to find centroids)
    print("Training IVF (K-Means clustering)...")
    start_time = time.time()
    index.train(embeddings)
    print(f"Training completed in {time.time() - start_time:.2f}s")

    # 6. Add all embeddings to the index
    print("Adding embeddings to index...")
    start_time = time.time()
    index.add(embeddings)
    print(f"Indexing completed in {time.time() - start_time:.2f}s")

    # 7. Set nprobe (how many centroids to search)
    index.nprobe = nprobe
    print(f"Set nprobe={nprobe}")

elif index_type == "exact":
    # Brute-force exact search (upper bound reference)
    print(f"\nBuilding Exact (Flat) index...")

    index = faiss.IndexFlatIP(dim)

    print("Adding embeddings to index...")
    start_time = time.time()
    index.add(embeddings)
    print(f"Indexing completed in {time.time() - start_time:.2f}s")

elif index_type == "hnsw":
    # HNSW: graph-based approximate nearest neighbor search
    M = 16  # Number of connections per node (for 2k docs; paper uses 32-64 for 38M docs)
    ef_construction = 32  # Controls index build quality
    ef_search = 8  # Controls search quality (paper tests 2, 4, 8, 16)

    print(f"\nBuilding HNSW index with M={M}, ef_construction={ef_construction}...")

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    print("Adding embeddings to index (includes graph construction)...")
    start_time = time.time()
    index.add(embeddings)
    print(f"Indexing completed in {time.time() - start_time:.2f}s")

    # Set ef_search (how many candidates to explore at search time)
    index.hnsw.efSearch = ef_search
    print(f"Set efSearch={ef_search}")

else:
    print(f"Unknown index type: {index_type}. Use 'ivf', 'exact', or 'hnsw'.")
    sys.exit(1)

# 8. Save the index
index_path = f"{data_dir}/{index_type}_index.index"
faiss.write_index(index, index_path)
print(f"\n{index_type.upper()} index saved to: {index_path}")

# 9. Verify index stats
print(f"\nIndex Statistics:")
print(f"  - Total vectors: {index.ntotal}")
print(f"  - Dimension: {dim}")
print(f"  - Is trained: {index.is_trained}")
