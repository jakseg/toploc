#!/usr/bin/env python3
"""
toploc_eval_combined.py
=======================
Baseline IVF  +  TopLoc-IVF  in ONE script.

The whole point: load the (huge) index and encode the queries ONCE, then run
as many nprobe experiments as you want without paying the ~30 min load again.

Two modes
---------
  Single nprobe (uses NP env var, default 128):
      python3 toploc_eval_combined.py snowflake ivf

  Full sweep (nprobe = 1,2,4,...,256) -> writes a CSV for the report/slides:
      python3 toploc_eval_combined.py snowflake ivf --sweep

  Custom sweep list:
      python3 toploc_eval_combined.py snowflake ivf --sweep 1,4,16,64,256

Output
------
  * Prints a results table to the terminal.
  * In sweep mode also writes  results_<model>_<index>.csv  with one row per
    nprobe (baseline time, toploc time, speedup, and all metrics for both).

Everything expensive (index read, query encoding, TopLoc centroid cache) is
done a single time at startup; the nprobe loop only re-times the search and
re-scores the metrics, which is fast.
"""

import os
import sys
import time
import csv
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
NP_SINGLE = int(os.environ.get("NP", 128))
USE_MMAP = os.environ.get("MMAP", "0") == "1"

BATCH_WARMUP_RUNS = 2
BATCH_TIMED_RUNS = 5
k = 10

# ---- parse --sweep flag ----
SWEEP = False
SWEEP_NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
if "--sweep" in sys.argv:
    SWEEP = True
    idx = sys.argv.index("--sweep")
    # optional comma-list right after --sweep
    if idx + 1 < len(sys.argv) and "," in sys.argv[idx + 1]:
        SWEEP_NPROBES = [int(x) for x in sys.argv[idx + 1].split(",")]

faiss.omp_set_num_threads(1)


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
                batch_size=32,
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


# ================= HELPERS =================
def get_centroid_vectors(quantizer, centroid_indices):
    idx = np.asarray(centroid_indices, dtype="int64")
    try:
        return quantizer.reconstruct_batch(idx).astype("float32")
    except (AttributeError, RuntimeError):
        d = quantizer.d
        vecs = np.empty((len(idx), d), dtype="float32")
        for local_i, global_i in enumerate(idx):
            vecs[local_i] = quantizer.reconstruct(int(global_i))
        return vecs


def toploc_ivf_search(ivf_index, q_emb, cached_ids, cached_vecs, nprobe, k):
    """TopLoc restricted search: score cached centroids, pick top-nprobe,
    then search_preassigned over only those lists."""
    nq = q_emb.shape[0]
    H = len(cached_ids)
    actual_nprobe = min(nprobe, H)
    use_ip = ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT

    # Step 1: score cached centroids (numpy BLAS)
    if use_ip:
        all_scores = q_emb @ cached_vecs.T
    else:
        q_sq = np.sum(q_emb**2, axis=1, keepdims=True)
        c_sq = np.sum(cached_vecs**2, axis=1).reshape(1, -1)
        all_scores = q_sq + c_sq - 2.0 * (q_emb @ cached_vecs.T)

    # Step 2: top-nprobe selection
    if use_ip:
        top_local = np.argpartition(-all_scores, actual_nprobe, axis=1)[
            :, :actual_nprobe
        ]
    else:
        top_local = np.argpartition(all_scores, actual_nprobe, axis=1)[
            :, :actual_nprobe
        ]

    sel_scores = np.take_along_axis(all_scores, top_local, axis=1).astype("float32")
    sel_ids = cached_ids[top_local].astype("int64")

    q_c = np.ascontiguousarray(q_emb, dtype="float32")
    sel_ids_c = np.ascontiguousarray(sel_ids, dtype="int64")
    sel_scores_c = np.ascontiguousarray(sel_scores, dtype="float32")

    # Step 3: restricted search
    old_nprobe = ivf_index.nprobe
    ivf_index.nprobe = actual_nprobe
    D, I = ivf_index.search_preassigned(q_c, k, sel_ids_c, sel_scores_c)
    ivf_index.nprobe = old_nprobe
    return D, I


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


# ================= LOAD INDEX (ONCE) =================
print(
    f"Evaluating {model_name} ({index_type})  |  mode={'SWEEP' if SWEEP else 'SINGLE'}"
)
print(f"FAISS threads: {faiss.omp_get_max_threads()}")
print("Loading index (this is the slow part — done only once)...")
t_load = time.perf_counter()

index_path = os.path.join(cache_dir, f"{index_type}_index.index")
if USE_MMAP:
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)
try:
    ivf_index.make_direct_map()
except Exception:
    pass

print(
    f"Index loaded in {time.perf_counter() - t_load:.1f}s: "
    f"ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
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
    kk: {p: s for p, s in v.items() if p in indexed_pids}
    for kk, v in qrels.items()
    if any(p in indexed_pids for p in v)
}
if not filtered_qrels:
    raise RuntimeError("No qrels survived filtering against indexed pids.")

# ================= GROUP TURNS BY CONVERSATION =================
conversations = defaultdict(list)
for turn_key in topics:
    conversations[turn_key.split("_")[0]].append(turn_key)
print(f"Grouped {len(topics)} turns into {len(conversations)} conversations")

# ================= ENCODE ALL QUERIES (ONCE) =================
print(f"\nLoading {model_name} encoder + encoding all queries (once)...")
encode_batch = load_query_encoder(model_name)
all_keys = list(topics.keys())
t_enc = time.perf_counter()
all_embs = encode_batch([topics[kk] for kk in all_keys])
emb_map = {kk: all_embs[i : i + 1] for i, kk in enumerate(all_keys)}  # each (1,dim)
print(f"Encoded {len(all_keys)} queries in {time.perf_counter() - t_enc:.1f}s")

# ================= BUILD TOPLOC CACHE (ONCE, nprobe-independent) =================
# The cached centroid set C0 depends only on q0 and H, NOT on nprobe.
print("Building TopLoc per-conversation centroid cache (once)...")
conv_cache = {}  # conv_id -> cached centroid IDs (int64)
conv_cvecs = {}  # conv_id -> centroid vectors (float32)
conv_q0_emb = {}  # conv_id -> q0 embedding (only if judged)
conv_fu_embs = {}  # conv_id -> follow-up embeddings (np array)
conv_fu_keys = {}  # conv_id -> follow-up turn keys

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    followup_keys = [t for t in turns[1:] if t in filtered_qrels]

    q0_emb = emb_map[q0_key]
    _, c0_indices = ivf_index.quantizer.search(q0_emb, H)
    c0_ids = c0_indices[0].astype("int64")
    c0_vecs = get_centroid_vectors(ivf_index.quantizer, c0_ids)

    conv_cache[conv_id] = c0_ids
    conv_cvecs[conv_id] = c0_vecs
    if q0_key in filtered_qrels:
        conv_q0_emb[conv_id] = q0_emb

    if followup_keys:
        conv_fu_embs[conv_id] = np.vstack(
            [emb_map[tk][0] for tk in followup_keys]
        ).astype("float32")
        conv_fu_keys[conv_id] = followup_keys

first_n = len(conv_q0_emb)
followup_n = sum(len(v) for v in conv_fu_keys.values())

# ================= BASELINE eval-key set (judged turns) =================
eval_keys = [kk for kk in topics if kk in filtered_qrels]
query_matrix = np.vstack([emb_map[kk][0] for kk in eval_keys]).astype("float32")
conv_rows = defaultdict(list)
for row, key in enumerate(eval_keys):
    conv_rows[key.split("_")[0]].append(row)
base_first_rows = [rows[0] for rows in conv_rows.values()]
base_fu_rows = [rows[1:] for rows in conv_rows.values()]
base_first_n = len(base_first_rows)
base_fu_n = sum(len(r) for r in base_fu_rows)

print(
    f"Eval turns: {len(eval_keys)} "
    f"(TopLoc: {first_n} q0 + {followup_n} follow-up | "
    f"Baseline: {base_first_n} q0 + {base_fu_n} follow-up)"
)


# ================= PER-NPROBE EVALUATION =================
def eval_toploc(np_val):
    """Return (metrics_dict, fu_per_query_ms, overall_per_query_ms)."""
    ivf_index.nprobe = np_val

    # ----- build run dict (untimed, for metrics) -----
    run = defaultdict(dict)
    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        if conv_id in conv_q0_emb:
            scores_0, indices_0 = base_index.search(conv_q0_emb[conv_id], k)
            for idx, sc in zip(indices_0[0], scores_0[0]):
                if idx >= 0 and id_map.get(str(idx)):
                    run[q0_key][id_map[str(idx)]] = float(sc)
        if conv_id in conv_fu_keys:
            D, I = toploc_ivf_search(
                ivf_index,
                conv_fu_embs[conv_id],
                conv_cache[conv_id],
                conv_cvecs[conv_id],
                np_val,
                k,
            )
            for r, tk in enumerate(conv_fu_keys[conv_id]):
                for idx, sc in zip(I[r], D[r]):
                    if idx >= 0 and id_map.get(str(idx)):
                        run[tk][id_map[str(idx)]] = float(sc)

    metrics = ir_measures.calc_aggregate(
        [nDCG @ 3, nDCG @ k, RR @ k], dict(filtered_qrels), dict(run)
    )

    # ----- timing -----
    def sweep():
        f_ms, u_ms = 0.0, 0.0
        for conv_id in conversations:
            if conv_id in conv_q0_emb:
                t0 = time.perf_counter()
                ivf_index.quantizer.search(conv_q0_emb[conv_id], H)
                base_index.search(conv_q0_emb[conv_id], k)
                f_ms += (time.perf_counter() - t0) * 1000
            if conv_id in conv_fu_keys:
                t0 = time.perf_counter()
                toploc_ivf_search(
                    ivf_index,
                    conv_fu_embs[conv_id],
                    conv_cache[conv_id],
                    conv_cvecs[conv_id],
                    np_val,
                    k,
                )
                u_ms += (time.perf_counter() - t0) * 1000
        return f_ms, u_ms

    for _ in range(BATCH_WARMUP_RUNS):
        sweep()
    fts, uts = [], []
    for _ in range(BATCH_TIMED_RUNS):
        f, u = sweep()
        fts.append(f)
        uts.append(u)

    _, _, u_mean = latency_stats(uts)
    _, _, f_mean = latency_stats(fts)
    fu_pq = u_mean / followup_n if followup_n else float("nan")
    overall_pq = (
        (f_mean + u_mean) / (first_n + followup_n)
        if (first_n + followup_n)
        else float("nan")
    )
    return metrics, fu_pq, overall_pq


def eval_baseline(np_val):
    if index_type == "ivf":
        base_index.nprobe = np_val
    ivf_index.nprobe = np_val

    # run dict
    run = defaultdict(dict)
    all_scores, all_indices = base_index.search(query_matrix, k)
    for row, tk in enumerate(eval_keys):
        for idx, sc in zip(all_indices[row], all_scores[row]):
            if idx >= 0 and id_map.get(str(idx)):
                run[tk][id_map[str(idx)]] = float(sc)
    metrics = ir_measures.calc_aggregate(
        [nDCG @ 3, nDCG @ k, RR @ k], dict(filtered_qrels), dict(run)
    )

    # timing (per-conversation, like baseline file)
    def sweep():
        f_ms, u_ms = 0.0, 0.0
        for q0_row, fu_rows in zip(base_first_rows, base_fu_rows):
            t0 = time.perf_counter()
            base_index.search(query_matrix[q0_row : q0_row + 1], k)
            f_ms += (time.perf_counter() - t0) * 1000
            if fu_rows:
                t0 = time.perf_counter()
                base_index.search(query_matrix[fu_rows], k)
                u_ms += (time.perf_counter() - t0) * 1000
        return f_ms, u_ms

    for _ in range(BATCH_WARMUP_RUNS):
        sweep()
    fts, uts = [], []
    for _ in range(BATCH_TIMED_RUNS):
        f, u = sweep()
        fts.append(f)
        uts.append(u)

    _, _, u_mean = latency_stats(uts)
    _, _, f_mean = latency_stats(fts)
    fu_pq = u_mean / base_fu_n if base_fu_n else float("nan")
    overall_pq = (f_mean + u_mean) / len(eval_keys) if eval_keys else float("nan")
    return metrics, fu_pq, overall_pq


# ================= RUN =================
nprobe_list = SWEEP_NPROBES if SWEEP else [NP_SINGLE]
rows = []

print("\n" + "=" * 100)
header = (
    f"{'nprobe':>6} | {'base ms':>9} {'topl ms':>9} {'speedup':>8} | "
    f"{'b_NDCG3':>8} {'t_NDCG3':>8} | {'b_NDCG10':>9} {'t_NDCG10':>9} | "
    f"{'b_MRR10':>8} {'t_MRR10':>8}"
)
print(header)
print("-" * 100)

for npv in nprobe_list:
    b_metrics, b_fu_pq, b_overall = eval_baseline(npv)
    t_metrics, t_fu_pq, t_overall = eval_toploc(npv)

    speedup = b_fu_pq / t_fu_pq if t_fu_pq and t_fu_pq == t_fu_pq else float("nan")

    b_n3, b_n10, b_mrr = b_metrics[nDCG @ 3], b_metrics[nDCG @ k], b_metrics[RR @ k]
    t_n3, t_n10, t_mrr = t_metrics[nDCG @ 3], t_metrics[nDCG @ k], t_metrics[RR @ k]

    print(
        f"{npv:>6} | {b_fu_pq:>9.3f} {t_fu_pq:>9.3f} {speedup:>7.2f}x | "
        f"{b_n3:>8.4f} {t_n3:>8.4f} | {b_n10:>9.4f} {t_n10:>9.4f} | "
        f"{b_mrr:>8.4f} {t_mrr:>8.4f}"
    )

    rows.append(
        {
            "nprobe": npv,
            "baseline_followup_ms_per_query": round(b_fu_pq, 4),
            "toploc_followup_ms_per_query": round(t_fu_pq, 4),
            "speedup_followup": round(speedup, 3),
            "baseline_overall_ms_per_query": round(b_overall, 4),
            "toploc_overall_ms_per_query": round(t_overall, 4),
            "baseline_NDCG@3": round(b_n3, 4),
            "toploc_NDCG@3": round(t_n3, 4),
            "baseline_NDCG@10": round(b_n10, 4),
            "toploc_NDCG@10": round(t_n10, 4),
            "baseline_MRR@10": round(b_mrr, 4),
            "toploc_MRR@10": round(t_mrr, 4),
        }
    )

print("=" * 100)
print(
    f"\nH (cached centroids): {H}   |   warmup={BATCH_WARMUP_RUNS} timed={BATCH_TIMED_RUNS}"
)
print("Times shown are FOLLOW-UP per-query (ms). 'overall' columns are in the CSV.")

# ================= WRITE CSV (sweep mode) =================
if SWEEP:
    out_csv = f"results_{model_name}_{index_type}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {out_csv}  ({len(rows)} rows)")
    print(
        "Columns: baseline/toploc follow-up + overall ms-per-query, speedup, "
        "and NDCG@3 / NDCG@10 / MRR@10 for both."
    )
