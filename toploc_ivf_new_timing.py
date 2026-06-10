#!/usr/bin/env python3
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
                    emb = torch.nn.functional.normalize(
                        model(**tokens).last_hidden_state[:, 0, :], p=2, dim=1
                    )
                    outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================= HELPER: Reconstruct centroids =================
def get_centroid_vectors(quantizer, centroid_indices):
    idx = np.asarray(centroid_indices, dtype="int64")
    try:
        return quantizer.reconstruct_batch(idx).astype("float32")
    except:
        d = quantizer.d
        vecs = np.empty((len(idx), d), dtype="float32")
        for li, gi in enumerate(idx):
            vecs[li] = quantizer.reconstruct(int(gi))
        return vecs


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
USE_IP = ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT
try:
    ivf_index.make_direct_map()
except:
    pass
print(
    f"Index loaded: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, metric={'IP' if USE_IP else 'L2'}"
)

H = min(H_CACHED_CENTROIDS, ivf_index.nlist)
if H != H_CACHED_CENTROIDS:
    print(f"Reducing cached centroids to nlist: {H}")

# ================= LOAD DATA =================
ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())

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
        qid, _, pid, score = parts
        qid, pid = qid.strip(), pid.strip()
        try:
            score = int(score)
        except:
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

# ================= GROUP BY CONVERSATION =================
conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)
print(f"Grouped {len(topics)} turns into {len(conversations)} conversations")

# ================= ENCODE ALL QUERIES =================
print(f"\nLoading {model_name} query encoder...")
encode_batch = load_query_encoder(model_name)
all_turn_keys = list(topics.keys())
print(f"Encoding {len(all_turn_keys)} queries...")
all_embs = encode_batch([topics[k] for k in all_turn_keys])
emb_lookup = {k: all_embs[i : i + 1] for i, k in enumerate(all_turn_keys)}

# FIX: Separate Retrieval K (fetch 1000) from Metric K (grade top 10)
RETRIEVE_K = 1000
METRIC_K = 10

# ================= BUILD RUN (untimed) =================
print("Building run dict (untimed)...")
run = defaultdict(dict)
conv_cache = {}
conv_q0_emb = {}
conv_fu_embs = {}
conv_fu_keys = {}

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    followup_keys = [t for t in turns[1:] if t in filtered_qrels]

    # TURN 0
    q0_emb = emb_lookup[q0_key]
    _, c0_indices = ivf_index.quantizer.search(q0_emb, H)
    c0_ids = c0_indices[0].astype("int64")
    c0_vecs = get_centroid_vectors(ivf_index.quantizer, c0_ids)
    conv_cache[conv_id] = {"c0": c0_ids, "c0_vecs": c0_vecs}

    scores_0, indices_0 = base_index.search(q0_emb, RETRIEVE_K)
    if q0_key in filtered_qrels:
        conv_q0_emb[conv_id] = q0_emb
        for idx, score in zip(indices_0[0], scores_0[0]):
            if idx >= 0 and id_map.get(str(idx)):
                run[q0_key][id_map[str(idx)]] = float(score)

    # TURNS 1+ (FAST NUMPY SCORING)
    if not followup_keys:
        continue
    fu_embs = encode_batch([topics[tk] for tk in followup_keys])
    conv_fu_embs[conv_id] = fu_embs
    conv_fu_keys[conv_id] = followup_keys

    nq = fu_embs.shape[0]
    # 1. Fast NumPy BLAS scoring (Faster than FAISS IndexFlat for small H)
    if USE_IP:
        all_scores = fu_embs @ c0_vecs.T
        top_local = np.argpartition(-all_scores, NP, axis=1)[:, :NP]
    else:
        q_sq = np.sum(fu_embs**2, axis=1, keepdims=True)
        c_sq = np.sum(c0_vecs**2, axis=1).reshape(1, -1)
        all_scores = q_sq + c_sq - 2.0 * (fu_embs @ c0_vecs.T)
        top_local = np.argpartition(all_scores, NP, axis=1)[:, :NP]

    sel_scores = np.take_along_axis(all_scores, top_local, axis=1).astype("float32")
    sel_ids = c0_ids[top_local].astype("int64")

    # 2. FAISS Fine Search
    q_c = np.ascontiguousarray(fu_embs, dtype="float32")
    sel_ids_c = np.ascontiguousarray(sel_ids, dtype="int64")
    sel_scores_c = np.ascontiguousarray(sel_scores, dtype="float32")

    old_nprobe = ivf_index.nprobe
    ivf_index.nprobe = NP
    scores_fu, indices_fu = ivf_index.search_preassigned(
        q_c, RETRIEVE_K, sel_ids_c, sel_scores_c
    )
    ivf_index.nprobe = old_nprobe

    for row_idx, turn_key in enumerate(followup_keys):
        for idx, score in zip(indices_fu[row_idx], scores_fu[row_idx]):
            if idx >= 0 and id_map.get(str(idx)):
                run[turn_key][id_map[str(idx)]] = float(score)

first_n = len(conv_q0_emb)
followup_n = sum(len(v) for v in conv_fu_keys.values())

# ================= METRICS =================
print(f"\nDEBUG: I have {len(run)} turns in my run dict.")
measures = [nDCG @ 3, nDCG @ METRIC_K, RR @ METRIC_K]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))


# ================= TIMING =================
def timed_sweep():
    first_total_ms, followup_total_ms = 0.0, 0.0
    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        if conv_id in conv_q0_emb:
            q0_emb = conv_q0_emb[conv_id]
            t0 = time.perf_counter()
            ivf_index.quantizer.search(q0_emb, H)
            base_index.search(q0_emb, RETRIEVE_K)
            first_total_ms += (time.perf_counter() - t0) * 1000

        if conv_id in conv_fu_keys:
            fu_embs = conv_fu_embs[conv_id]
            cache = conv_cache[conv_id]
            c0_ids = cache["c0"]
            c0_vecs = cache["c0_vecs"]

            t0 = time.perf_counter()
            nq = fu_embs.shape[0]

            # 1. Fast NumPy BLAS scoring
            if USE_IP:
                all_scores = fu_embs @ c0_vecs.T
                top_local = np.argpartition(-all_scores, NP, axis=1)[:, :NP]
            else:
                q_sq = np.sum(fu_embs**2, axis=1, keepdims=True)
                c_sq = np.sum(c0_vecs**2, axis=1).reshape(1, -1)
                all_scores = q_sq + c_sq - 2.0 * (fu_embs @ c0_vecs.T)
                top_local = np.argpartition(all_scores, NP, axis=1)[:, :NP]

            sel_scores = np.take_along_axis(all_scores, top_local, axis=1).astype(
                "float32"
            )
            sel_ids = c0_ids[top_local].astype("int64")

            # 2. FAISS Fine Search
            q_c = np.ascontiguousarray(fu_embs, dtype="float32")
            sel_ids_c = np.ascontiguousarray(sel_ids, dtype="int64")
            sel_scores_c = np.ascontiguousarray(sel_scores, dtype="float32")

            old_nprobe = ivf_index.nprobe
            ivf_index.nprobe = NP
            ivf_index.search_preassigned(q_c, RETRIEVE_K, sel_ids_c, sel_scores_c)
            ivf_index.nprobe = old_nprobe

            followup_total_ms += (time.perf_counter() - t0) * 1000

    return first_total_ms, followup_total_ms


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


print(
    f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed per-conversation sweeps..."
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
        f"  run {run_i + 1}: first-turn total={f_ms:7.2f} ms (per-query={f_pq:.3f}) | follow-up total={u_ms:7.2f} ms (per-query={u_pq:.3f})"
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
    f"Turns evaluated: {first_n + followup_n}  ({first_n} first-turn, {followup_n} follow-up, {len(conversations)} conversations)"
)
print(f"NDCG@3:  {results[nDCG @ 3]:.4f}")
print(f"NDCG@10: {results[nDCG @ METRIC_K]:.4f}")
print(f"MRR@10:  {results[RR @ METRIC_K]:.4f}")
print()
print(
    f"Latency, summed over conversations (min / median / mean over {BATCH_TIMED_RUNS} sweeps):"
)
print(f"  first-turn total:     {f_min:8.2f}  /  {f_med:8.2f}  /  {f_mean:8.2f}  ms")
print(f"  first-turn per query: {per_query_line((f_min, f_med, f_mean), first_n)}")
print(f"  follow-up  total:     {u_min:8.2f}  /  {u_med:8.2f}  /  {u_mean:8.2f}  ms")
print(f"  follow-up  per query: {per_query_line((u_min, u_med, u_mean), followup_n)}")
print(f"  Total ::: {((f_mean+u_mean)/2):8.2f}  ms")
print(f"\nCentroids cached per conv: {H}")
print(f"nprobe:                    {NP}")
print("=" * 70)
