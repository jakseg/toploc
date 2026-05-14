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

TRAIN_SAMPLE_SIZE = 2_000_000
BATCH_SIZE = 500_000

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
emb_dir = EMBEDDING_DIRS[model_name]
cache_dir = CACHE_DIRS[model_name]
os.makedirs(cache_dir, exist_ok=True)

# ================= HELPERS =================
def get_parquet_info(emb_dir):
    parquet_files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {emb_dir}")
    total_rows = sum(pq.read_metadata(pf).num_rows for pf in parquet_files)
    sample = pq.read_table(parquet_files[0], columns=["embedding"])
    dim = len(sample.column("embedding")[0].as_py())
    return parquet_files, total_rows, dim


def iter_parquet_batches(parquet_files, dim):
    for pf in parquet_files:
        table = pq.read_table(pf)
        ids = table.column("id").to_pylist()
        try:
            flat = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
            embs = flat.reshape(len(ids), dim).astype("float32")
        except Exception:
            embs = np.array(table.column("embedding").to_pylist(), dtype=np.float32)
        yield ids, embs


def load_training_sample(parquet_files, dim, n_sample):
    collected = []
    total = 0
    for _, embs in iter_parquet_batches(parquet_files, dim):
        collected.append(embs)
        total += len(embs)
        if total >= n_sample:
            break
    sample = np.concatenate(collected)[:n_sample]
    print(f"Training sample: {len(sample):,} vectors")
    return sample


# ================= ID MAP =================
id_map_path = os.path.join(cache_dir, "passage_id_map.json")
parquet_files, total_rows, dim = get_parquet_info(emb_dir)
print(f"Collection: {total_rows:,} passages, dim={dim}, {len(parquet_files)} parquet files")

if not os.path.exists(id_map_path):
    print("Building passage ID map...")
    all_ids = []
    for i, pf in enumerate(parquet_files):
        table = pq.read_table(pf, columns=["id"])
        all_ids.extend(table.column("id").to_pylist())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(parquet_files)} files")
    id_map = {str(i): str(pid) for i, pid in enumerate(all_ids)}
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)
    print(f"Saved ID map: {len(id_map):,} entries")

params = INDEX_PARAMS[model_name].get(index_type, {})

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

        print(f"Training on {TRAIN_SAMPLE_SIZE:,} sample vectors (not full collection)...")
        train_data = load_training_sample(parquet_files, dim, TRAIN_SAMPLE_SIZE)
        start_time = time.time()
        index.train(train_data)
        del train_data
        print(f"Training completed in {time.time() - start_time:.2f}s")

        print("Adding embeddings in batches (streaming from parquet)...")
        start_time = time.time()
        added = 0
        for ids, embs in iter_parquet_batches(parquet_files, dim):
            index.add(embs)
            added += len(embs)
            if added % BATCH_SIZE < len(embs):
                print(f"  Added {added:,}/{total_rows:,} vectors")
        print(f"Indexing completed in {time.time() - start_time:.2f}s")

        index.nprobe = nprobe
        print(f"Set nprobe={nprobe}")

    elif index_type == "exact":
        print(f"\nBuilding Exact (Flat) index...")
        index = faiss.IndexFlatIP(dim)

        print("Adding embeddings in batches...")
        start_time = time.time()
        added = 0
        for ids, embs in iter_parquet_batches(parquet_files, dim):
            index.add(embs)
            added += len(embs)
            if added % BATCH_SIZE < len(embs):
                print(f"  Added {added:,}/{total_rows:,} vectors")
        print(f"Indexing completed in {time.time() - start_time:.2f}s")

    elif index_type == "hnsw":
        M = params.get("M", 32)
        ef_construction = params.get("ef_construction", 200)
        ef_search = params.get("ef_search", 64)

        print(f"\nBuilding HNSW index with M={M}, ef_construction={ef_construction}...")
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction

        print("Adding embeddings in batches (includes graph construction)...")
        start_time = time.time()
        added = 0
        for ids, embs in iter_parquet_batches(parquet_files, dim):
            index.add(embs)
            added += len(embs)
            if added % BATCH_SIZE < len(embs):
                print(f"  Added {added:,}/{total_rows:,} vectors")
        print(f"Indexing completed in {time.time() - start_time:.2f}s")

        index.hnsw.efSearch = ef_search
        print(f"Set efSearch={ef_search}")

    else:
        print(f"Unknown index type: {index_type}. Use 'ivf', 'exact', or 'hnsw'.")
        sys.exit(1)

    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    print("Done.")

print(f"\nIndex Statistics:")
print(f"  - Total vectors: {index.ntotal}")
print(f"  - Dimension: {dim}")
print(f"  - Is trained: {index.is_trained}")
