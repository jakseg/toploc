#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import faiss
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR
from toploc_search import toploc_ivf_search, toploc_ivf_search_ptr  # C++ version

# ================= CONFIGURATION =================
CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc2/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc2/Datasets/conversational/CAST2019/topics"
)

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
cache_dir = CACHE_DIRS[model_name]

H_CACHED_CENTROIDS = int(os.environ.get("H_CACHED", 1024))
NP = int(os.environ.get("NP", 128))  # match baseline nprobe default
USE_MMAP = os.environ.get("MMAP", "0") == "1"

# Latency benchmarking config — mirror the baseline
BATCH_WARMUP_RUNS = 2
BATCH_TIMED_RUNS = 5

# Use all available cores for FAISS — match the baseline
faiss.omp_set_num_threads(os.cpu_count() or 1)


# ================= QUERY ENCODER (Batched) =================
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
                batch = queries[i : i + chunk]
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = model(**tokens)
                    emb = torch.nn.functional.normalize(
                        outputs.last_hidden_state[:, 0, :], p=2, dim=1
                    )
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================= LOAD INDEX =================
print(f"Evaluating TopLoc-IVF for: {model_name} ({index_type})")
print(f"FAISS threads: {faiss.omp_get_max_threads()}")
index_path = os.path.join(cache_dir, f"{index_type}_index.index")

if USE_MMAP:
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)
ivf_index.nprobe = NP

try:
    ivf_index.make_direct_map()
except Exception:
    pass

print(
    f"Index loaded: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}"
)

H = min(H_CACHED_CENTROIDS, ivf_index.nlist)
if H != H_CACHED_CENTROIDS:
    print(f"Reducing cached centroids to nlist: {H}")

# ================= LOAD ID MAPPING =================
ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())

# ================= LOAD TOPICS =================
# Match baseline: split on first comma only, preserving commas in query text
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

if not topics:
    raise RuntimeError("Parsed 0 topics.")

# ================= LOAD QRELS =================
# Match baseline: qrels are comma-separated
qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
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

filtered_qrels = {
    k: {p: s for p, s in v.items() if p in indexed_pids}
    for k, v in qrels.items()
    if any(p in indexed_pids for p in v)
}

if not filtered_qrels:
    raise RuntimeError("No qrels survived filtering against indexed pids.")

# ================= GROUP TURNS BY CONVERSATION =================
conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)

print(f"Grouped {len(topics)} turns into {len(conversations)} conversations")

# ================= ENCODE ONCE + BUILD RUN (untimed) =================
# Retrieval results are independent of timing, so we run everything once
# (untimed) to build the run dict and populate the per-conversation cache.
# Latency is measured separately below, mirroring the baseline.
print(f"\nLoading {model_name} query encoder...")
encode_batch = load_query_encoder(model_name)
print("Encoding queries and building run dict (untimed)...")

conv_cache = {}  # conv_id -> cached centroid ids (int64)
conv_q0_emb = {}  # conv_id -> q0 embedding (1, d)   [only if q0 judged]
conv_fu_embs = {}  # conv_id -> follow-up embeddings (nq, d)
conv_fu_keys = {}  # conv_id -> list of follow-up turn keys
k = 1000
run = defaultdict(dict)

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    followup_keys = [t for t in turns[1:] if t in filtered_qrels]

    # ---- TURN 0: build cache (ALWAYS) + full search ----
    q0_emb = encode_batch([topics[q0_key]])
    _, c0_indices = ivf_index.quantizer.search(q0_emb, H)
    conv_cache[conv_id] = c0_indices[0].astype("int64")
    scores_0, indices_0 = base_index.search(q0_emb, k)

    if q0_key in filtered_qrels:
        conv_q0_emb[conv_id] = q0_emb
        for idx, score in zip(indices_0[0], scores_0[0]):
            if idx >= 0 and id_map.get(str(idx)):
                run[q0_key][id_map[str(idx)]] = float(score)

    # ---- TURNS 1+: batched cached search ----
    if not followup_keys:
        continue

    fu_embs = encode_batch([topics[tk] for tk in followup_keys])
    conv_fu_embs[conv_id] = fu_embs
    conv_fu_keys[conv_id] = followup_keys

    scores_fu, indices_fu = toploc_ivf_search_ptr(
        int(ivf_index.this), fu_embs, conv_cache[conv_id], NP, k
    )
    for row_idx, turn_key in enumerate(followup_keys):
        for idx, score in zip(indices_fu[row_idx], scores_fu[row_idx]):
            if idx >= 0 and id_map.get(str(idx)):
                run[turn_key][id_map[str(idx)]] = float(score)

first_n = len(conv_q0_emb)
followup_n = sum(len(v) for v in conv_fu_keys.values())

# ================= COMPUTE METRICS (deterministic, untimed) =================
print(f"\nDEBUG: I have {len(run)} turns in my run dict.")

measures = [nDCG @ 3, nDCG @ k, RR @ k]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))


# ================= TIMING: PER-CONVERSATION SWEEP =================
# One full pass: per conversation, time q0 (full search + cache build) and the
# follow-up batch (cached search) separately. The cache is already built above,
# so follow-up timing reuses it — exactly how it would work at serving time.
def timed_sweep():
    first_total_ms, followup_total_ms = 0.0, 0.0
    for conv_id, turns in conversations.items():
        q0_key = turns[0]

        # Time q0 only if judged (mirror run-dict membership)
        if conv_id in conv_q0_emb:
            q0_emb = conv_q0_emb[conv_id]
            t0 = time.perf_counter()
            ivf_index.quantizer.search(q0_emb, H)  # cache-build cost
            base_index.search(q0_emb, k)  # full search
            first_total_ms += (time.perf_counter() - t0) * 1000

        # Time the follow-up batch using the prebuilt cache
        if conv_id in conv_fu_keys:
            fu_embs = conv_fu_embs[conv_id]
            cache = conv_cache[conv_id]
            t0 = time.perf_counter()
            toploc_ivf_search_ptr(int(ivf_index.this), fu_embs, cache, NP, k)
            followup_total_ms += (time.perf_counter() - t0) * 1000

    return first_total_ms, followup_total_ms


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


print(
    f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed "
    f"per-conversation sweeps..."
)
for _ in range(BATCH_WARMUP_RUNS):
    timed_sweep()

first_times, followup_times = [], []
for run_i in range(BATCH_TIMED_RUNS):
    f_ms, u_ms = timed_sweep()
    first_times.append(f_ms)
    followup_times.append(u_ms)
    f_pq = f_ms / first_n if first_n else float("nan")
    u_pq = u_ms / followup_n if followup_n else float("nan")
    print(
        f"  run {run_i + 1}: first-turn total={f_ms:7.2f} ms (per-query={f_pq:.3f}) | "
        f"follow-up total={u_ms:7.2f} ms (per-query={u_pq:.3f})"
    )

# ================= RESULTS =================
f_min, f_med, f_mean = latency_stats(first_times)
u_min, u_med, u_mean = latency_stats(followup_times)


def per_query_line(stats, n):
    lo, md, mn = stats
    if not n:
        return "n/a"
    return f"{lo / n:8.3f}  /  {md / n:8.3f}  /  {mn / n:8.3f}  ms"


print("\n" + "=" * 70)
print(f"TOPLOC-IVF EVALUATION RESULTS ({index_type.upper()}, {model_name})")
print("=" * 70)
print(
    f"Turns evaluated: {first_n + followup_n}  "
    f"({first_n} first-turn, {followup_n} follow-up, {len(conversations)} conversations)"
)
print(f"NDCG@3:  {results[nDCG @ 3]:.4f}")
print(f"NDCG@10: {results[nDCG @ k]:.4f}")
print(f"MRR@10:  {results[RR @ k]:.4f}")
print()
print(
    f"Latency, summed over conversations (min / median / mean over "
    f"{BATCH_TIMED_RUNS} sweeps):"
)
print(f"  first-turn total:     {f_min:8.2f}  /  {f_med:8.2f}  /  {f_mean:8.2f}  ms")
print(f"  first-turn per query: {per_query_line((f_min, f_med, f_mean), first_n)}")
print(f"  follow-up  total:     {u_min:8.2f}  /  {u_med:8.2f}  /  {u_mean:8.2f}  ms")
print(f"  follow-up  per query: {per_query_line((u_min, u_med, u_mean), followup_n)}")
print(f"\nCentroids cached per conv: {H}")
print(f"nprobe:                    {NP}")
print("=" * 70)
