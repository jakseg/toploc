import os
import sys
import numpy as np
import faiss
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR

# ================= CONFIGURATION =================
CACHE_BASE = "/home/toploc2/Datasets/toploc2"
DATASET_DIR = "/home/toploc2/Datasets/conversational/CAST2019/topics"

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
cache_dir = CACHE_DIRS[model_name]

USE_MMAP = os.environ.get("MMAP", "0") == "1"

# efSearch values to sweep. Override with EF_LIST="64,128,256,512,1024,2048".
EF_LIST = [int(x) for x in os.environ.get(
    "EF_LIST", "64,128,256,512,1024,2048").split(",") if x.strip()]

faiss.omp_set_num_threads(os.cpu_count() or 1)


# ================= QUERY ENCODER (batched) =================
def load_query_encoder(model_name):
    if model_name == "snowflake":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode_batch(queries):
            return model.encode(
                queries,
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=True,
            ).astype("float32")

        return encode_batch

    elif model_name == "dragon":
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
        model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
        model.eval()

        def encode_batch(queries, chunk=32):
            outs = []
            for i in range(0, len(queries), chunk):
                batch = queries[i:i + chunk]
                tokens = tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**tokens)
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================= PRECOMPUTED QUERY EMBEDDINGS (optional) =================
def load_precomputed_query_embeddings(model_name, keys):
    import pyarrow.parquet as pq

    emb_path = os.path.join(DATASET_DIR, f"topics_{model_name}_embeddings.parquet")
    if not os.path.exists(emb_path):
        return None

    table = pq.read_table(emb_path)
    cols = table.column_names
    id_col = next((c for c in ("id", "qid", "turn_id", "topic_id", "tid") if c in cols), None)
    emb_col = next((c for c in ("embedding", "embeddings", "vector", "emb") if c in cols), None)
    if id_col is None or emb_col is None:
        return None

    emb_map = {str(i): e for i, e in zip(
        table.column(id_col).to_pylist(), table.column(emb_col).to_pylist())}
    if any(k not in emb_map for k in keys):
        return None

    matrix = np.ascontiguousarray([emb_map[k] for k in keys], dtype="float32")
    faiss.normalize_L2(matrix)
    print(f"  Loaded precomputed query embeddings {matrix.shape}")
    return matrix


# ================= LOAD INDEX (once) =================
print(f"efSearch sweep for: {model_name} (hnsw)")
print(f"FAISS threads: {faiss.omp_get_max_threads()}")

index_path = os.path.join(cache_dir, "hnsw_index.index")
if not os.path.exists(index_path):
    print(f"ERROR: Index not found at {index_path}")
    sys.exit(1)

if USE_MMAP:
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    index = faiss.read_index(index_path)
print(f"HNSW index loaded: {index.ntotal} vectors, mmap={USE_MMAP}")

ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())

# ================= TOPICS / QRELS =================
topics = {}
with open(os.path.join(DATASET_DIR, "topics.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            topics[parts[0].strip()] = parts[1].strip()

qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        qid, _, pid, score = (p.strip() for p in parts)
        try:
            score = int(score)
        except ValueError:
            continue
        if score > 0:
            qrels[qid][pid] = score

filtered_qrels = {}
for turn_key, pid_scores in qrels.items():
    valid = {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
    if valid:
        filtered_qrels[turn_key] = valid

eval_keys = [k for k in topics if k in filtered_qrels]
eval_queries = [topics[k] for k in eval_keys]
N = len(eval_queries)
print(f"Eval set: {N} turns")

# ================= QUERY EMBEDDINGS (once) =================
query_matrix = load_precomputed_query_embeddings(model_name, eval_keys)
if query_matrix is None:
    print(f"Encoding {N} queries...")
    query_matrix = load_query_encoder(model_name)(eval_queries)

k = 10
measures = [nDCG @ 3, nDCG @ k, RR @ k]


def evaluate(ef):
    index.hnsw.efSearch = ef
    run = defaultdict(dict)
    scores, indices = index.search(query_matrix, k)
    for row, turn_key in enumerate(eval_keys):
        for idx, sc in zip(indices[row], scores[row]):
            if idx < 0:
                continue
            pid = id_map.get(str(idx))
            if pid is not None:
                run[turn_key][pid] = float(sc)
    return ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))


# ================= SWEEP =================
print("\n" + "=" * 60)
print(f"{'efSearch':>10} | {'NDCG@3':>8} | {'NDCG@10':>8} | {'MRR@10':>8}")
print("-" * 60)
prev = None
for ef in EF_LIST:
    r = evaluate(ef)
    n3, n10, rr = r[nDCG @ 3], r[nDCG @ k], r[RR @ k]
    delta = "" if prev is None else f"  (ΔNDCG@10 {n10 - prev:+.4f})"
    print(f"{ef:>10} | {n3:8.4f} | {n10:8.4f} | {rr:8.4f}{delta}")
    prev = n10
print("=" * 60)
print("Paper target (snowflake HNSW): NDCG@3=0.548 NDCG@10=0.500 MRR@10=0.814")
print("Pick the smallest efSearch where NDCG@10 stops improving (plateau).")
