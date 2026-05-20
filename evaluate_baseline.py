import os
import sys
import time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR

# ================= CONFIGURATION =================
CACHE_BASE = "/home/toploc2/Datasets/toploc2"
DATASET_DIR = "/home/toploc2/Datasets/conversational/CAST2019/topics"

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    # "dragon": os.path.join(CACHE_BASE, "dragon"),
}

# Latency benchmarking config
BATCH_WARMUP_RUNS = 2
BATCH_TIMED_RUNS = 5

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
cache_dir = CACHE_DIRS[model_name]

# Use all available cores for FAISS
faiss.omp_set_num_threads(os.cpu_count() or 1)


# ================= QUERY ENCODER (batched) =================
def load_query_encoder(model_name):
    """Return a function: list[str] -> np.ndarray of shape (N, dim)."""
    if model_name == "snowflake":
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


# ================= LOAD DATA =================
print(f"Evaluating baseline for: {model_name} ({index_type})")
print(f"FAISS threads: {faiss.omp_get_max_threads()}")
print("Loading components...")

# 1. Load pre-built index (created by create_index.py)
index_path = os.path.join(cache_dir, f"{index_type}_index.index")
if not os.path.exists(index_path):
    print(f"ERROR: Index not found at {index_path}")
    print("Run create_index.py first to build it.")
    sys.exit(1)

index = faiss.read_index(index_path)
print(f"{index_type.upper()} index loaded: {index.ntotal} vectors, trained={index.is_trained}")

if index_type == "ivf":
    index.nprobe = 128
    print(f"Set nprobe={index.nprobe}")
elif index_type == "hnsw":
    index.hnsw.efSearch = 64
    print(f"Set efSearch={index.hnsw.efSearch}")

# 2. ID Mapping (built inline with the index by create_index.py)
ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())
print(f"ID map loaded: {len(id_map)} passages")

# 3. Topics (Queries)
topics = {}
topics_path = os.path.join(DATASET_DIR, "topics.tsv")
with open(topics_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            topics[parts[0].strip()] = parts[1].strip()
print(f"Topics loaded: {len(topics)} turns")

# 4. QRELS (Relevance Judgments)
qrels = defaultdict(dict)
qrels_path = os.path.join(DATASET_DIR, "qrels.qrel")
with open(qrels_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        qid, _, pid, score = parts
        qid = qid.strip()
        pid = pid.strip()
        try:
            score = int(score)
        except ValueError:
            continue
        if score > 0:
            qrels[qid][pid] = score

filtered_qrels = {}
total_turns = len(qrels)
for turn_key, pid_scores in qrels.items():
    valid_pids = {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
    if valid_pids:
        filtered_qrels[turn_key] = valid_pids

print(f"Turns with relevant passages in index: {len(filtered_qrels)}/{total_turns}")

# ================= COLLECT EVAL QUERIES IN STABLE ORDER =================
eval_keys = [k for k in topics if k in filtered_qrels]
eval_queries = [topics[k] for k in eval_keys]
N = len(eval_queries)
print(f"\nEval set: {N} turns")

# ================= ENCODE ALL QUERIES (BATCH) =================
print(f"Loading {model_name} query encoder...")
encode_batch = load_query_encoder(model_name)
print(f"Batch-encoding {N} queries...")
t0 = time.perf_counter()
query_matrix = encode_batch(eval_queries)
enc_ms = (time.perf_counter() - t0) * 1000
print(f"Query encoding done in {enc_ms:.1f} ms total "
      f"({enc_ms / N:.2f} ms/query) — shape={query_matrix.shape}")

# ================= BATCH SEARCH WITH WARMUP + REPEATED RUNS =================
k = 10
print(f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed batch searches...")

# Warmup runs (not timed, let caches and any JIT settle)
for _ in range(BATCH_WARMUP_RUNS):
    _ = index.search(query_matrix, k)

# Timed runs
batch_times_ms = []
last_scores, last_indices = None, None
for run_i in range(BATCH_TIMED_RUNS):
    t0 = time.perf_counter()
    last_scores, last_indices = index.search(query_matrix, k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    batch_times_ms.append(elapsed_ms)
    print(f"  run {run_i + 1}: batch={elapsed_ms:.2f} ms  "
          f"per-query={elapsed_ms / N:.3f} ms")

batch_ms_min = min(batch_times_ms)
batch_ms_median = float(np.median(batch_times_ms))
batch_ms_mean = float(np.mean(batch_times_ms))

# ================= BUILD RUN FROM BATCH RESULTS =================
run = defaultdict(dict)
for row, turn_key in enumerate(eval_keys):
    for idx, score in zip(last_indices[row], last_scores[row]):
        if idx < 0:
            continue
        pid = id_map.get(str(idx))
        if pid is not None:
            run[turn_key][pid] = float(score)

# Compute metrics with ir_measures
measures = [nDCG @ 3, nDCG @ k, RR @ k]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

# ================= RESULTS =================
print("\n" + "=" * 70)
print(f"BASELINE EVALUATION RESULTS ({index_type.upper()}, {model_name})")
print("=" * 70)
print(f"Turns evaluated: {N}")
print(f"NDCG@3:  {results[nDCG @ 3]:.4f}")
print(f"NDCG@10: {results[nDCG @ k]:.4f}")
print(f"MRR@10:  {results[RR @ k]:.4f}")
print()
print(f"Batch latency  (min / median / mean over {BATCH_TIMED_RUNS} runs):")
print(f"  total:     {batch_ms_min:8.2f}  /  {batch_ms_median:8.2f}  /  "
      f"{batch_ms_mean:8.2f}  ms")
print(f"  per query: {batch_ms_min / N:8.3f}  /  {batch_ms_median / N:8.3f}  /  "
      f"{batch_ms_mean / N:8.3f}  ms")
print("=" * 70)

# Paper Table 1 — TREC CAsT 2019 baselines (full collection)
paper_targets = {
    "snowflake": {
        "exact": {"NDCG@3": 0.550, "NDCG@10": 0.502, "MRR@10": 0.817},
        "ivf":   {"NDCG@3": 0.544, "NDCG@10": 0.497, "MRR@10": 0.815, "Time": 24.9},
        "hnsw":  {"NDCG@3": 0.548, "NDCG@10": 0.500, "MRR@10": 0.814, "Time": 1.8},
    },
    "dragon": {
        "exact": {"NDCG@3": 0.522, "NDCG@10": 0.492, "MRR@10": 0.799},
        "ivf":   {"NDCG@3": 0.528, "NDCG@10": 0.486, "MRR@10": 0.813, "Time": 33.0},
        "hnsw":  {"NDCG@3": 0.508, "NDCG@10": 0.469, "MRR@10": 0.789, "Time":  8.3},
    },
}
if model_name in paper_targets and index_type in paper_targets[model_name]:
    t = paper_targets[model_name][index_type]
    parts = [f"{m} = {v}" for m, v in t.items()]
    print(f"\nPaper targets (CAsT 2019): {' | '.join(parts)}")
