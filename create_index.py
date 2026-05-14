import sys
import os
import glob
import faiss
import numpy as np
import json
import time
import pyarrow.parquet as pq

# ================= CONFIGURATION =================
EMBEDDINGS_BASE = "/home/toploc2/Datasets/conversational/CAST2019"
CACHE_BASE = "/home/toploc2/Datasets/toploc2"

EMBEDDING_DIRS = {
    "snowflake": os.path.join(EMBEDDINGS_BASE, "snowflake_embeddings"),
    # "dragon": os.path.join(EMBEDDINGS_BASE, "dragon_embeddings"),
}

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    # "dragon": os.path.join(CACHE_BASE, "dragon"),
}

# Paper parameters (Table 1, Section 3 — full 38M collection)
INDEX_PARAMS = {
    "snowflake": {
        "ivf": {"num_centroids": 2**15, "nprobe": 128},
        "hnsw": {"M": 32, "ef_construction": 200, "ef_search": 64},
    },
    "dragon": {
        "ivf": {"num_centroids": 2**18, "nprobe": 128},
        "hnsw": {"M": 32, "ef_construction": 200, "ef_search": 64},
    },
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
emb_dir = EMBEDDING_DIRS[model_name]
cache_dir = CACHE_DIRS[model_name]
os.makedirs(cache_dir, exist_ok=True)

# ================= LOAD PARQUET EMBEDDINGS =================
def load_parquet_embeddings(emb_dir):
    parquet_files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {emb_dir}")
    print(f"Found {len(parquet_files)} parquet files in {emb_dir}")

    total_rows = sum(pq.read_metadata(pf).num_rows for pf in parquet_files)
    sample = pq.read_table(parquet_files[0])
    print(f"Schema: {sample.schema}")
    dim = len(sample.column("embedding")[0].as_py())
    print(f"Total: {total_rows:,} passages, dim={dim}")

    embeddings = np.empty((total_rows, dim), dtype="float32")
    all_ids = []
    offset = 0

    for i, pf in enumerate(parquet_files):
        table = pq.read_table(pf)
        n_rows = len(table)
        all_ids.extend(table.column("id").to_pylist())

        # Arrow -> numpy directly in C, no Python object overhead
        flat = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        embeddings[offset:offset + n_rows] = flat.reshape(n_rows, dim)
        offset += n_rows

        if (i + 1) % 10 == 0 or (i + 1) == len(parquet_files):
            print(f"  Loaded {i + 1}/{len(parquet_files)} files ({offset:,}/{total_rows:,} passages)")

    embeddings = embeddings[:offset]
    id_map = {str(i): str(pid) for i, pid in enumerate(all_ids)}
    return embeddings, id_map


npy_path = os.path.join(cache_dir, "passage_embeddings.npy")
id_map_path = os.path.join(cache_dir, "passage_id_map.json")

if os.path.exists(npy_path) and os.path.exists(id_map_path):
    print("Loading cached npy embeddings...")
    embeddings = np.load(npy_path)
    with open(id_map_path, "r") as f:
        id_map = json.load(f)
    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
else:
    print("Loading embeddings from parquet files...")
    embeddings, id_map = load_parquet_embeddings(emb_dir)
    print("Caching as npy for faster future loads...")
    np.save(npy_path, embeddings)
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)
    print(f"Saved {npy_path} and {id_map_path}")

dim = embeddings.shape[1]
params = INDEX_PARAMS[model_name][index_type] if index_type in INDEX_PARAMS.get(model_name, {}) else {}

# ================= BUILD INDEX =================
index_path = os.path.join(cache_dir, f"{index_type}_index.index")

if os.path.exists(index_path):
    print(f"\nIndex already exists at {index_path}, skipping build.")
    print("Delete it to force rebuild.")
    index = faiss.read_index(index_path)
else:
    if index_type == "ivf":
        num_centroids = params.get("num_centroids", 2**15)
        nprobe = params.get("nprobe", 128)

        print(f"\nBuilding IVF index with {num_centroids} centroids...")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, num_centroids, faiss.METRIC_INNER_PRODUCT)

        print("Training IVF (K-Means clustering)...")
        start_time = time.time()
        index.train(embeddings)
        print(f"Training completed in {time.time() - start_time:.2f}s")

        print("Adding embeddings to index...")
        start_time = time.time()
        index.add(embeddings)
        print(f"Indexing completed in {time.time() - start_time:.2f}s")

        index.nprobe = nprobe
        print(f"Set nprobe={nprobe}")

    elif index_type == "exact":
        print(f"\nBuilding Exact (Flat) index...")

        index = faiss.IndexFlatIP(dim)

        print("Adding embeddings to index...")
        start_time = time.time()
        index.add(embeddings)
        print(f"Indexing completed in {time.time() - start_time:.2f}s")

    elif index_type == "hnsw":
        M = params.get("M", 32)
        ef_construction = params.get("ef_construction", 200)
        ef_search = params.get("ef_search", 64)

        print(f"\nBuilding HNSW index with M={M}, ef_construction={ef_construction}...")

        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction

        print("Adding embeddings to index (includes graph construction)...")
        start_time = time.time()
        index.add(embeddings)
        print(f"Indexing completed in {time.time() - start_time:.2f}s")

        index.hnsw.efSearch = ef_search
        print(f"Set efSearch={ef_search}")

    else:
        print(f"Unknown index type: {index_type}. Use 'ivf', 'exact', or 'hnsw'.")
        sys.exit(1)

    faiss.write_index(index, index_path)
    print(f"\n{index_type.upper()} index saved to: {index_path}")

print(f"\nIndex Statistics:")
print(f"  - Total vectors: {index.ntotal}")
print(f"  - Dimension: {dim}")
print(f"  - Is trained: {index.is_trained}")
