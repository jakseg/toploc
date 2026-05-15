import sys
import os
import gc
import glob
import random
import faiss
import numpy as np
import json
import time
import logging
import pyarrow.parquet as pq

random.seed(42)

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
SAVE_EVERY = 50  # checkpoint every N parquet files

VALID_INDEX_TYPES = ("exact", "ivf", "hnsw")


# ================= HELPERS =================
def get_parquet_info(emb_dir):
    parquet_files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {emb_dir}")
    sample = pq.read_table(parquet_files[0], columns=["embedding"])
    dim = len(sample.column("embedding")[0].as_py())
    return parquet_files, dim


def read_parquet(pf, dim):
    table = pq.read_table(pf)
    ids = table.column("id").to_pylist()
    try:
        flat = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        embs = flat.reshape(len(ids), dim).astype("float32")
    except Exception:
        embs = np.array(table.column("embedding").to_pylist(), dtype=np.float32)
    return ids, embs


def setup_logger(log_path):
    log = logging.getLogger(log_path)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ================= BUILD ONE INDEX =================
def build_index(model_name, index_type, parquet_files, dim, cache_dir):
    log_path = os.path.join(cache_dir, f"{index_type}_indexCreation.log")
    log = setup_logger(log_path)

    index_path = os.path.join(cache_dir, f"{index_type}_index.index")
    ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
    checkpoint_path = os.path.join(cache_dir, f"{index_type}_checkpoint.json")
    params = INDEX_PARAMS[model_name].get(index_type, {})

    # Resume or init
    start_file = 0
    all_ids = []
    index = None

    if os.path.exists(checkpoint_path) and os.path.exists(index_path) and os.path.exists(ids_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_file = ckpt["last_file_index"] + 1
        all_ids = list(np.load(ids_path, allow_pickle=True))
        index = faiss.read_index(index_path)
        log.info(f"[{index_type}] Resuming from file {start_file}/{len(parquet_files)}, "
                 f"index has {index.ntotal:,} vectors")

    elif os.path.exists(index_path) and not os.path.exists(checkpoint_path):
        log.info(f"[{index_type}] Final index already exists at {index_path}, skipping.")
        index = faiss.read_index(index_path)
        log.info(f"[{index_type}] Stats: ntotal={index.ntotal:,}, trained={index.is_trained}")
        return

    else:
        if index_type == "ivf":
            num_centroids = params.get("num_centroids", 2**15)
            log.info(f"[ivf] Building with {num_centroids} centroids...")
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, num_centroids, faiss.METRIC_INNER_PRODUCT)

            log.info(f"[ivf] Collecting training sample (target {TRAIN_SAMPLE_SIZE:,} vectors)...")
            shuffled = list(parquet_files)
            random.shuffle(shuffled)
            train_chunks = []
            collected = 0
            for pf in shuffled:
                _, embs = read_parquet(pf, dim)
                faiss.normalize_L2(embs)
                train_chunks.append(embs)
                collected += len(embs)
                if collected >= TRAIN_SAMPLE_SIZE:
                    break
            train_data = np.concatenate(train_chunks)[:TRAIN_SAMPLE_SIZE]
            del train_chunks
            log.info(f"[ivf] Training on {len(train_data):,} vectors "
                     f"(FAISS needs >= {39 * num_centroids:,})...")
            t0 = time.time()
            index.train(train_data)
            del train_data
            gc.collect()
            log.info(f"[ivf] Training completed in {time.time() - t0:.2f}s")

        elif index_type == "exact":
            log.info("[exact] Building IndexFlatIP...")
            index = faiss.IndexFlatIP(dim)

        elif index_type == "hnsw":
            M = params.get("M", 32)
            ef_construction = params.get("ef_construction", 200)
            log.info(f"[hnsw] Building with M={M}, ef_construction={ef_construction}...")
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction

        else:
            log.error(f"Unknown index type: {index_type}")
            return

    # Add loop with checkpointing
    log.info(f"[{index_type}] Adding embeddings (streaming from parquet)...")
    t0 = time.time()
    for i in range(start_file, len(parquet_files)):
        pf = parquet_files[i]
        try:
            ids, embs = read_parquet(pf, dim)
            faiss.normalize_L2(embs)
            index.add(embs)
            all_ids.extend(ids)
        except Exception as e:
            log.error(f"[{index_type}] [{i + 1}/{len(parquet_files)}] "
                      f"Failed on {os.path.basename(pf)}: {e}")
            continue

        if (i + 1) % 10 == 0:
            log.info(f"[{index_type}] [{i + 1}/{len(parquet_files)}] "
                     f"Indexed: {index.ntotal:,} vectors")

        if (i + 1) % SAVE_EVERY == 0:
            log.info(f"[{index_type}] Saving checkpoint at file {i + 1}...")
            faiss.write_index(index, index_path)
            np.save(ids_path, np.array(all_ids, dtype=object))
            with open(checkpoint_path, "w") as f:
                json.dump({"last_file_index": i, "ntotal": index.ntotal}, f)
            log.info(f"[{index_type}] Checkpoint saved ({index.ntotal:,} vectors).")

    log.info(f"[{index_type}] Add loop completed in {time.time() - t0:.2f}s")

    # Set search-time params
    if index_type == "ivf":
        index.nprobe = params.get("nprobe", 128)
        log.info(f"[ivf] Set nprobe={index.nprobe}")
    elif index_type == "hnsw":
        index.hnsw.efSearch = params.get("ef_search", 64)
        log.info(f"[hnsw] Set efSearch={index.hnsw.efSearch}")

    # Final save
    log.info(f"[{index_type}] Saving final index and IDs...")
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(all_ids, dtype=object))
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    log.info(f"[{index_type}] Done. Total vectors: {index.ntotal:,}")
    log.info(f"[{index_type}] Index:  {index_path}")
    log.info(f"[{index_type}] IDs:    {ids_path}")


# ================= MAIN =================
model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
type_arg = sys.argv[2] if len(sys.argv) > 2 else "ivf"

if type_arg == "all":
    index_types = list(VALID_INDEX_TYPES)
else:
    index_types = [t.strip() for t in type_arg.split(",")]
    for t in index_types:
        if t not in VALID_INDEX_TYPES:
            print(f"Unknown index type: {t}. Use one of {VALID_INDEX_TYPES} or 'all'.")
            sys.exit(1)

emb_dir = EMBEDDING_DIRS[model_name]
cache_dir = CACHE_DIRS[model_name]
os.makedirs(cache_dir, exist_ok=True)

parquet_files, dim = get_parquet_info(emb_dir)
print(f"Model: {model_name} | Types: {index_types} | "
      f"{len(parquet_files)} parquet files, dim={dim}")

for it in index_types:
    print(f"\n{'=' * 60}\nBuilding: {it}\n{'=' * 60}")
    t0 = time.time()
    build_index(model_name, it, parquet_files, dim, cache_dir)
    print(f"[{it}] total wall time: {(time.time() - t0) / 60:.1f} min")
    gc.collect()

print("\nAll requested index types complete.")
