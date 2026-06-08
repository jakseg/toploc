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
    "dragon": os.path.join(CACHE_BASE, "dragon"),
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


# ================= PRECOMPUTED QUERY EMBEDDINGS (optional) =================
def load_precomputed_query_embeddings(model_name, keys):
    """Try to load precomputed query embeddings for `keys` (turn IDs) from
    topics_<model>_embeddings.parquet in DATASET_DIR.

    Returns an (N, dim) float32 matrix aligned to `keys` on full success,
    otherwise None (caller falls back to on-the-fly encoding).
    """
    import pyarrow.parquet as pq

    emb_path = os.path.join(DATASET_DIR, f"topics_{model_name}_embeddings.parquet")
    if not os.path.exists(emb_path):
        return None

    table = pq.read_table(emb_path)
    cols = table.column_names
    id_col = next((c for c in ("id", "qid", "turn_id", "topic_id", "tid") if c in cols), None)
    emb_col = next((c for c in ("embedding", "embeddings", "vector", "emb") if c in cols), None)
    if id_col is None or emb_col is None:
        print(f"  WARN: {os.path.basename(emb_path)} has columns {cols}; "
              f"could not find id/embedding columns — encoding instead.")
        return None

    emb_map = {str(i): e for i, e in zip(
        table.column(id_col).to_pylist(), table.column(emb_col).to_pylist())}
    missing = [k for k in keys if k not in emb_map]
    if missing:
        print(f"  WARN: {len(missing)}/{len(keys)} eval turns missing from "
              f"{os.path.basename(emb_path)} (e.g. {missing[:3]}) — encoding instead.")
        return None

    matrix = np.ascontiguousarray([emb_map[k] for k in keys], dtype="float32")
    # Index is inner-product over L2-normalized document vectors, so normalize
    # queries too (idempotent if they are already normalized).
    faiss.normalize_L2(matrix)
    print(f"  Loaded precomputed query embeddings {matrix.shape} "
          f"from {os.path.basename(emb_path)}")
    return matrix


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
# Turn keys look like "<conv_id>_<turn>". We mirror the TopLoc evaluation by
# processing one conversation at a time: the first (judged) turn q0 is searched
# alone, and the remaining turns of that conversation are searched as one batch.
# This matches how TopLoc must work — its per-conversation cache is built on q0
# and reused by the follow-ups, so a single FAISS batch cannot span queries from
# different conversations. Using the same per-conversation granularity here keeps
# the baseline vs TopLoc latency comparison fair.
eval_keys = [k for k in topics if k in filtered_qrels]
eval_queries = [topics[k] for k in eval_keys]
N = len(eval_queries)

# ================= OBTAIN QUERY EMBEDDINGS =================
# Prefer precomputed topic embeddings if available; otherwise encode on the fly.
print(f"\nObtaining query embeddings for {N} queries...")
query_matrix = load_precomputed_query_embeddings(model_name, eval_keys)
if query_matrix is None:
    print(f"Loading {model_name} query encoder...")
    encode_batch = load_query_encoder(model_name)
    print(f"Batch-encoding {N} queries...")
    t0 = time.perf_counter()
    query_matrix = encode_batch(eval_queries)
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"Query encoding done in {enc_ms:.1f} ms total "
          f"({enc_ms / N:.2f} ms/query) — shape={query_matrix.shape}")

# Group the encoded eval rows by conversation, preserving topics order.
conv_rows = defaultdict(list)
for row, key in enumerate(eval_keys):
    conv_rows[key.split("_")[0]].append(row)

first_rows = [rows[0] for rows in conv_rows.values()]          # one q0 per conversation
followup_rows_per_conv = [rows[1:] for rows in conv_rows.values()]
first_n = len(first_rows)
followup_n = sum(len(r) for r in followup_rows_per_conv)
print(f"\nEval set: {N} turns "
      f"({first_n} first-turn, {followup_n} follow-up) "
      f"across {len(conv_rows)} conversations")

k = 10

# ================= BUILD RUN (correctness, untimed) =================
# Retrieval results are independent of how queries are batched, so we run all
# eval queries once (untimed) and build the run dict from that. Latency is
# measured separately below. This mirrors the single/batch split used in
# toploc_ivf_batched_all.py (results vs timing).
run = defaultdict(dict)
all_scores, all_indices = index.search(query_matrix, k)
for row, turn_key in enumerate(eval_keys):
    for idx, score in zip(all_indices[row], all_scores[row]):
        if idx < 0:
            continue
        pid = id_map.get(str(idx))
        if pid is not None:
            run[turn_key][pid] = float(score)

measures = [nDCG @ 3, nDCG @ k, RR @ k]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))


# ================= TIMING: PER-CONVERSATION SWEEP =================
def timed_sweep():
    """One full pass: per conversation, time q0 (single) + follow-ups (one batch).

    Returns the summed first-turn and follow-up latency over all conversations.
    """
    first_total_ms, followup_total_ms = 0.0, 0.0
    for q0_row, fu_rows in zip(first_rows, followup_rows_per_conv):
        t0 = time.perf_counter()
        index.search(query_matrix[q0_row:q0_row + 1], k)
        first_total_ms += (time.perf_counter() - t0) * 1000
        if fu_rows:
            t0 = time.perf_counter()
            index.search(query_matrix[fu_rows], k)
            followup_total_ms += (time.perf_counter() - t0) * 1000
    return first_total_ms, followup_total_ms


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


print(f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed "
      f"per-conversation sweeps...")
for _ in range(BATCH_WARMUP_RUNS):
    timed_sweep()

first_times, followup_times = [], []
for run_i in range(BATCH_TIMED_RUNS):
    f_ms, u_ms = timed_sweep()
    first_times.append(f_ms)
    followup_times.append(u_ms)
    f_pq = f_ms / first_n if first_n else float("nan")
    u_pq = u_ms / followup_n if followup_n else float("nan")
    print(f"  run {run_i + 1}: first-turn total={f_ms:7.2f} ms (per-query={f_pq:.3f}) | "
          f"follow-up total={u_ms:7.2f} ms (per-query={u_pq:.3f})")

# ================= RESULTS =================
f_min, f_med, f_mean = latency_stats(first_times)
u_min, u_med, u_mean = latency_stats(followup_times)

# Overall = first-turn + follow-up combined, per sweep. This collapses the
# single/batch split back into one average query latency over all N turns,
# which is what the paper's "Time" column reports (mean ms per query).
overall_times = [f + u for f, u in zip(first_times, followup_times)]
o_min, o_med, o_mean = latency_stats(overall_times)


def per_query_line(stats, n):
    lo, md, mn = stats
    if not n:
        return "n/a"
    return f"{lo / n:8.3f}  /  {md / n:8.3f}  /  {mn / n:8.3f}  ms"


print("\n" + "=" * 70)
print(f"BASELINE EVALUATION RESULTS ({index_type.upper()}, {model_name})")
print("=" * 70)
print(f"Turns evaluated: {N}  ({first_n} first-turn, {followup_n} follow-up, "
      f"{len(conv_rows)} conversations)")
print(f"NDCG@3:  {results[nDCG @ 3]:.4f}")
print(f"NDCG@10: {results[nDCG @ k]:.4f}")
print(f"MRR@10:  {results[RR @ k]:.4f}")
print()
print(f"Latency, summed over conversations (min / median / mean over "
      f"{BATCH_TIMED_RUNS} sweeps):")
print(f"  first-turn total:     {f_min:8.2f}  /  {f_med:8.2f}  /  {f_mean:8.2f}  ms")
print(f"  first-turn per query: {per_query_line((f_min, f_med, f_mean), first_n)}")
print(f"  follow-up  total:     {u_min:8.2f}  /  {u_med:8.2f}  /  {u_mean:8.2f}  ms")
print(f"  follow-up  per query: {per_query_line((u_min, u_med, u_mean), followup_n)}")
print(f"  overall    total:     {o_min:8.2f}  /  {o_med:8.2f}  /  {o_mean:8.2f}  ms")
print(f"  overall    per query: {per_query_line((o_min, o_med, o_mean), N)}"
      f"   <- compare to paper Time")
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
