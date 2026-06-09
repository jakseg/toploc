#!/usr/bin/env python3
"""
TopLoc-IVF evaluation — MEGA-BATCHED pure Python.

Key optimization over previous version:
  - ALL follow-ups across ALL conversations are collected and issued as
    ONE search_preassigned call, maximizing FAISS batch parallelism.
  - ALL first turns are also batched into one quantizer.search + one
    search_preassigned call.
  - All centroids reconstructed ONCE at startup.

Run:
    python3 toploc_ivf_megabatch.py snowflake ivf
    NP=8   python3 toploc_ivf_megabatch.py snowflake ivf
    NP=32  python3 toploc_ivf_megabatch.py snowflake ivf
"""

import os
import sys
import time
import numpy as np
import faiss
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR

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
NP = int(os.environ.get("NP", 128))
USE_MMAP = os.environ.get("MMAP", "0") == "1"

BATCH_WARMUP_RUNS = 2
BATCH_TIMED_RUNS = 5

faiss.omp_set_num_threads(os.cpu_count() or 1)


# ================= QUERY ENCODER =================
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
        enc_model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
        enc_model.eval()

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
                    emb = torch.nn.functional.normalize(
                        enc_model(**tokens).last_hidden_state[:, 0, :], p=2, dim=1
                    )
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================= LOAD INDEX =================
print(f"Evaluating TopLoc-IVF (mega-batched) for: {model_name} ({index_type})")
print(f"FAISS threads: {faiss.omp_get_max_threads()}")
index_path = os.path.join(cache_dir, f"{index_type}_index.index")

if USE_MMAP:
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)

print(
    f"Index loaded: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}"
)

H = min(H_CACHED_CENTROIDS, ivf_index.nlist)
d = ivf_index.d
USE_IP = ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT

# ================= RECONSTRUCT ALL CENTROIDS ONCE =================
print(f"Reconstructing all {ivf_index.nlist} centroids (one-time cost)...")
t_recon = time.perf_counter()
all_centroids = np.zeros((ivf_index.nlist, d), dtype="float32")
for i in range(ivf_index.nlist):
    all_centroids[i] = ivf_index.quantizer.reconstruct(i)
print(f"  Done in {time.perf_counter() - t_recon:.1f}s — shape {all_centroids.shape}")

# ================= LOAD ID MAPPING =================
ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())

# ================= LOAD TOPICS =================
topics = {}
with open(os.path.join(DATASET_DIR, "topics.tsv"), "r", encoding="utf-8") as f:
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
qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        qid, _, pid, score = parts
        qid, pid = qid.strip(), pid.strip()
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
    raise RuntimeError("No qrels survived filtering.")

# ================= GROUP TURNS BY CONVERSATION =================
conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)
print(f"Grouped {len(topics)} turns into {len(conversations)} conversations")

# ================= ENCODE ALL QUERIES ONCE =================
print(f"\nLoading {model_name} query encoder...")
encode_batch = load_query_encoder(model_name)

# Encode ALL queries, build lookup
all_turn_keys = list(topics.keys())
all_turn_texts = [topics[k] for k in all_turn_keys]
print(f"Encoding {len(all_turn_keys)} queries...")
all_embs = encode_batch(all_turn_texts)  # (N_total, d)
emb_lookup = {k: all_embs[i] for i, k in enumerate(all_turn_keys)}

k = 10
actual_nprobe = min(NP, H)

# ================= BUILD RUN DICT (untimed, mega-batched) =================
print("Building run dict (untimed)...")
run = defaultdict(dict)

# ---- Organize conversations ----
conv_ids_ordered = list(conversations.keys())
conv_q0_keys = [conversations[cid][0] for cid in conv_ids_ordered]
conv_q0_embs = np.array([emb_lookup[k] for k in conv_q0_keys], dtype="float32")

# ---- Phase 1: ALL first turns batched ----
# One quantizer.search → top-H centroids per conversation
D_c, C_batch = ivf_index.quantizer.search(conv_q0_embs, H)  # (n_conv, H)

# Store per-conversation cache (for follow-ups)
conv_cache = {}  # conv_id -> (H,) centroid IDs
conv_cvecs = {}  # conv_id -> (H, d) centroid vectors (from pre-reconstructed)
for ci, conv_id in enumerate(conv_ids_ordered):
    c_ids = C_batch[ci].astype("int64")
    conv_cache[conv_id] = c_ids
    conv_cvecs[conv_id] = all_centroids[
        c_ids
    ]  # zero-cost lookup into pre-reconstructed

# First-turn search: search_preassigned with top-NP of the H cached
sel_ids_first = np.ascontiguousarray(C_batch[:, :actual_nprobe], dtype="int64")
sel_dists_first = np.ascontiguousarray(D_c[:, :actual_nprobe], dtype="float32")
ivf_index.nprobe = actual_nprobe
D_first, I_first = ivf_index.search_preassigned(
    conv_q0_embs, k, sel_ids_first, sel_dists_first
)

# Record judged first turns into run
for ci, q0_key in enumerate(conv_q0_keys):
    if q0_key in filtered_qrels:
        for idx, score in zip(I_first[ci], D_first[ci]):
            if idx >= 0 and id_map.get(str(idx)):
                run[q0_key][id_map[str(idx)]] = float(score)

# ---- Phase 2: ALL follow-ups mega-batched ----
# Per-conversation: numpy coarse scoring against cached centroids
# Then ONE search_preassigned for all follow-ups across all conversations
conv_fu_keys = {}  # conv_id -> list of follow-up turn keys
conv_fu_embs = {}  # conv_id -> (n_fu, d) embeddings
all_fu_qids = []
all_fu_embs_list = []
all_a_ids_list = []
all_a_dists_list = []

for conv_id in conv_ids_ordered:
    turns = conversations[conv_id]
    followup_keys = [t for t in turns[1:] if t in filtered_qrels]
    if not followup_keys:
        continue

    fu_embs = np.array([emb_lookup[k] for k in followup_keys], dtype="float32")
    conv_fu_keys[conv_id] = followup_keys
    conv_fu_embs[conv_id] = fu_embs

    c_ids = conv_cache[conv_id]
    c_vecs = conv_cvecs[conv_id]  # (H, d)

    # Coarse scoring: numpy BLAS matrix multiply
    if USE_IP:
        scores = fu_embs @ c_vecs.T  # (n_fu, H)
        top_local = np.argpartition(-scores, actual_nprobe, axis=1)[:, :actual_nprobe]
    else:
        q_sq = np.sum(fu_embs**2, axis=1, keepdims=True)
        c_sq = np.sum(c_vecs**2, axis=1).reshape(1, -1)
        scores = q_sq + c_sq - 2.0 * (fu_embs @ c_vecs.T)
        top_local = np.argpartition(scores, actual_nprobe, axis=1)[:, :actual_nprobe]

    a_ids = c_ids[top_local].astype("int64")
    a_dists = np.take_along_axis(scores, top_local, axis=1).astype("float32")

    all_fu_qids.extend(followup_keys)
    all_fu_embs_list.append(fu_embs)
    all_a_ids_list.append(a_ids)
    all_a_dists_list.append(a_dists)

# ONE mega-batch search_preassigned for ALL follow-ups
if all_fu_qids:
    mega_embs = np.ascontiguousarray(np.vstack(all_fu_embs_list), dtype="float32")
    mega_ids = np.ascontiguousarray(np.vstack(all_a_ids_list), dtype="int64")
    mega_dists = np.ascontiguousarray(np.vstack(all_a_dists_list), dtype="float32")

    ivf_index.nprobe = actual_nprobe
    D_fu, I_fu = ivf_index.search_preassigned(mega_embs, k, mega_ids, mega_dists)

    for i, qid in enumerate(all_fu_qids):
        for idx, score in zip(I_fu[i], D_fu[i]):
            if idx >= 0 and id_map.get(str(idx)):
                run[qid][id_map[str(idx)]] = float(score)

first_n = sum(1 for k in conv_q0_keys if k in filtered_qrels)
followup_n = len(all_fu_qids)

# ================= COMPUTE METRICS =================
print(f"\nDEBUG: I have {len(run)} turns in my run dict.")
measures = [nDCG @ 3, nDCG @ k, RR @ k]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

# ================= TIMING: MEGA-BATCHED SWEEPS =================
# Precompute the judged first-turn batch for timing
judged_first_indices = [ci for ci, k in enumerate(conv_q0_keys) if k in filtered_qrels]
judged_first_embs = np.ascontiguousarray(
    conv_q0_embs[judged_first_indices], dtype="float32"
)


def timed_sweep():
    # ---- Phase 1: all judged first turns (ONE quantizer.search + ONE search_preassigned) ----
    t0 = time.perf_counter()
    D_c, C_ids = ivf_index.quantizer.search(judged_first_embs, H)
    sel_ids = np.ascontiguousarray(C_ids[:, :actual_nprobe], dtype="int64")
    sel_dists = np.ascontiguousarray(D_c[:, :actual_nprobe], dtype="float32")
    ivf_index.nprobe = actual_nprobe
    ivf_index.search_preassigned(judged_first_embs, k, sel_ids, sel_dists)
    first_ms = (time.perf_counter() - t0) * 1000

    # ---- Phase 2: all follow-ups (per-conv numpy scoring + ONE mega search_preassigned) ----
    t0 = time.perf_counter()
    fu_e_list, a_i_list, a_d_list = [], [], []

    for conv_id in conv_fu_keys:
        fu_embs = conv_fu_embs[conv_id]
        c_ids = conv_cache[conv_id]
        c_vecs = conv_cvecs[conv_id]

        if USE_IP:
            scores = fu_embs @ c_vecs.T
            top_local = np.argpartition(-scores, actual_nprobe, axis=1)[
                :, :actual_nprobe
            ]
        else:
            q_sq = np.sum(fu_embs**2, axis=1, keepdims=True)
            c_sq = np.sum(c_vecs**2, axis=1).reshape(1, -1)
            scores = q_sq + c_sq - 2.0 * (fu_embs @ c_vecs.T)
            top_local = np.argpartition(scores, actual_nprobe, axis=1)[
                :, :actual_nprobe
            ]

        fu_e_list.append(fu_embs)
        a_i_list.append(c_ids[top_local].astype("int64"))
        a_d_list.append(np.take_along_axis(scores, top_local, axis=1).astype("float32"))

    m_embs = np.ascontiguousarray(np.vstack(fu_e_list), dtype="float32")
    m_ids = np.ascontiguousarray(np.vstack(a_i_list), dtype="int64")
    m_dists = np.ascontiguousarray(np.vstack(a_d_list), dtype="float32")

    ivf_index.nprobe = actual_nprobe
    ivf_index.search_preassigned(m_embs, k, m_ids, m_dists)
    followup_ms = (time.perf_counter() - t0) * 1000

    return first_ms, followup_ms


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


print(
    f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed "
    f"mega-batched sweeps..."
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
    f"({first_n} first-turn, {followup_n} follow-up, "
    f"{len(conversations)} conversations)"
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
