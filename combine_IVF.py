#!/usr/bin/env python3
"""
combine_all3.py
===============
Baseline IVF  +  TopLoc-IVF  +  TopLoc-IVF+   in ONE script.

Loads the (huge) index and encodes the queries ONCE, then evaluates all THREE
methods across either a single nprobe or a full sweep — so every method is
measured under identical conditions (same index, same embeddings, same k,
same thread count).

Methods
-------
  * Baseline IVF : full search, scores all nlist centroids.
  * TopLoc IVF   : per-conversation cache of H centroids built on q0; follow-ups
                   only score the cached set. Pure Python (numpy) scoring.
  * TopLoc IVF+  : same as TopLoc, but refreshes the cache when the query drifts
                   away from q0 (I0 overlap < alpha * nprobe). Sequential.

Paper-faithful timing (SIGIR'25 TopLoc, Sec. 3 "Implementation details")
------------------------------------------------------------------------
The paper (1) pins the process to ONE CPU socket + its local memory via
numactl, (2) uses ALL physical cores of that socket for FAISS, and
(3) times BATCHES of queries (per-query time = batch time / batch size).
This script now times batched searches. Run it like:

    numactl --cpunodebind=0 --membind=0 \
    env NUM_THREADS=<physical cores of node 0> \
        OMP_NUM_THREADS=<same> OPENBLAS_NUM_THREADS=<same> MKL_NUM_THREADS=<same> \
    python3 combine_all8.py snowflake ivf --sweep

Find your core count per NUMA node with:  numactl --hardware  /  lscpu

Modes
-----
  Single nprobe (NP env, default 128):
      python3 combine_all3.py snowflake ivf
  Full sweep (writes results3_<model>_<index>.csv):
      python3 combine_all3.py snowflake ivf --sweep
  Custom sweep:
      python3 combine_all3.py snowflake ivf --sweep 1,4,16,64,256
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
CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc1/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc1/Datasets/conversational/CAST2019/topics"
)
CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
cache_dir = CACHE_DIRS[model_name]

# H grid search (paper tests h in {512, 1024, 4096, 8192}).
# Override with H_GRID="512,1024" or pin a single value with H_CACHED=1024.
if "H_CACHED" in os.environ:
    H_GRID = [int(os.environ["H_CACHED"])]
else:
    H_GRID = [
        int(x) for x in os.environ.get("H_GRID", "512,1024,4096,8192").split(",")
    ]
NP_SINGLE = int(os.environ.get("NP", 128))
ALPHA = float(os.environ.get("ALPHA", 0.1))  # TopLoc+ drift threshold (single value fallback)
# Alpha grid search (paper tests alpha in {0.0, 0.05, 0.1, 0.2}).
# Override with ALPHA_GRID="0.0,0.1" or pin a single value with ALPHA=0.1.
if "ALPHA_GRID" not in os.environ and "ALPHA" in os.environ:
    ALPHA_GRID = [float(os.environ["ALPHA"])]
else:
    ALPHA_GRID = [
        float(x) for x in os.environ.get("ALPHA_GRID", "0.0,0.05,0.1,0.2").split(",")
    ]
USE_MMAP = os.environ.get("MMAP", "0") == "1"

# Retrieve depth: 1000 to match paper methodology, metrics measured @3/@10
# RETRIEVE_K = int(os.environ.get("RETRIEVE_K", 1000))
RETRIEVE_K = int(os.environ.get("RETRIEVE_K", 10))
METRIC_K = 10

BATCH_WARMUP_RUNS = 2
BATCH_TIMED_RUNS = 5

# ---- parse --sweep flag ----
SWEEP = False
SWEEP_NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
if "--sweep" in sys.argv:
    SWEEP = True
    idx = sys.argv.index("--sweep")
    if idx + 1 < len(sys.argv) and "," in sys.argv[idx + 1]:
        SWEEP_NPROBES = [int(x) for x in sys.argv[idx + 1].split(",")]

# ---- thread count ----
# Paper setup: numactl pins the process to ONE CPU socket + its local memory,
# and FAISS uses all physical cores of that socket (NOT a single thread).
# Set NUM_THREADS to the number of PHYSICAL cores of one NUMA node.
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 1))
faiss.omp_set_num_threads(NUM_THREADS)


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
_CENTROID_MATRIX = None  # (nlist, d) float32, precomputed once after index load


def get_centroid_vectors(quantizer, centroid_indices):
    idx = np.asarray(centroid_indices, dtype="int64")
    if _CENTROID_MATRIX is not None:
        return _CENTROID_MATRIX[idx]  # O(H) memory copy — no reconstruct calls
    try:
        return quantizer.reconstruct_batch(idx).astype("float32")
    except (AttributeError, RuntimeError):
        d = quantizer.d
        vecs = np.empty((len(idx), d), dtype="float32")
        for local_i, global_i in enumerate(idx):
            vecs[local_i] = quantizer.reconstruct(int(global_i))
        return vecs


def toploc_assign(q_emb, cached_ids, cached_vecs, nprobe, use_ip):
    """Coarse step of TopLoc: score queries against the H cached centroids only
    and return (sel_ids, sel_scores) for search_preassigned. This is the part
    that replaces the full quantizer search."""
    actual_nprobe = min(nprobe, len(cached_ids) - 1)

    if use_ip:
        all_scores = q_emb @ cached_vecs.T
        top_local = np.argpartition(-all_scores, actual_nprobe, axis=1)[
            :, :actual_nprobe
        ]
    else:
        q_sq = np.sum(q_emb**2, axis=1, keepdims=True)
        c_sq = np.sum(cached_vecs**2, axis=1).reshape(1, -1)
        all_scores = q_sq + c_sq - 2.0 * (q_emb @ cached_vecs.T)
        top_local = np.argpartition(all_scores, actual_nprobe, axis=1)[
            :, :actual_nprobe
        ]

    sel_scores = np.take_along_axis(all_scores, top_local, axis=1).astype("float32")
    sel_ids = cached_ids[top_local].astype("int64")
    return sel_ids, sel_scores


def preassigned_search(ivf_index, q_emb, sel_ids, sel_scores, nprobe, k):
    """Fine step: scan the preassigned inverted lists.

    IMPORTANT: with the default parallel_mode=0, FAISS parallelizes over
    queries in the outer search() call, NOT inside search_preassigned().
    Calling search_preassigned() directly therefore runs SINGLE-threaded.
    parallel_mode=2 moves the parallelism inside search_preassigned()
    (over queries x probes), so the scan uses all FAISS threads.
    """
    q_c = np.ascontiguousarray(q_emb, dtype="float32")
    sel_ids_c = np.ascontiguousarray(sel_ids, dtype="int64")
    sel_scores_c = np.ascontiguousarray(sel_scores, dtype="float32")

    old_nprobe = ivf_index.nprobe
    old_pmode = ivf_index.parallel_mode
    ivf_index.nprobe = sel_ids_c.shape[1]
    ivf_index.parallel_mode = 2
    try:
        D, I = ivf_index.search_preassigned(q_c, k, sel_ids_c, sel_scores_c)
    finally:
        ivf_index.nprobe = old_nprobe
        ivf_index.parallel_mode = old_pmode
    return D, I


def toploc_ivf_search(ivf_index, q_emb, cached_ids, cached_vecs, nprobe, k):
    """TopLoc restricted search (numpy coarse scoring + FAISS preassigned scan)."""
    use_ip = ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT
    sel_ids, sel_scores = toploc_assign(q_emb, cached_ids, cached_vecs, nprobe, use_ip)
    return preassigned_search(ivf_index, q_emb, sel_ids, sel_scores, nprobe, k)


def rank_within_cache(centroid_vecs, q_emb, nprobe, use_ip):
    """Local indices (within the cache) of the top-nprobe centroids — used by
    TopLoc+ for the drift (I0 overlap) check. q_emb is shape (1, d)."""
    if use_ip:
        coarse = (centroid_vecs @ q_emb.T).reshape(-1)
        order = np.argsort(-coarse)
    else:
        coarse = ((centroid_vecs - q_emb) ** 2).sum(axis=1)
        order = np.argsort(coarse)
    return order[:nprobe]


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


# ================= LOAD INDEX (ONCE) =================
print(
    f"Evaluating {model_name} ({index_type})  |  mode={'SWEEP' if SWEEP else 'SINGLE'}"
)
print(f"FAISS threads: {faiss.omp_get_max_threads()}  (NUM_THREADS={NUM_THREADS})")
print(f"Retrieve k={RETRIEVE_K}, metrics @3/@{METRIC_K}, alpha(IVF+)={ALPHA}")
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

USE_IP = ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT
print(
    f"Index loaded in {time.perf_counter() - t_load:.1f}s: "
    f"ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if USE_IP else 'L2'}"
)

# Precompute ALL centroid vectors once (the "C++ advantage" fix):
# every cache build and every TopLoc+ refresh becomes a numpy slice instead
# of per-vector reconstruct calls. Memory: nlist x d x 4 bytes
# (Dragon: 262,144 x 768 ~ 0.8 GB; Snowflake: 32,768 x 1024 ~ 0.13 GB).
print("Precomputing centroid matrix (once)...")
_t0 = time.perf_counter()
_CENTROID_MATRIX = ivf_index.quantizer.reconstruct_n(0, ivf_index.nlist).astype(
    "float32"
)
print(
    f"  Centroid matrix {_CENTROID_MATRIX.shape} "
    f"({_CENTROID_MATRIX.nbytes / 1e9:.2f} GB) in {time.perf_counter() - _t0:.1f}s"
)

H_GRID = sorted({min(h, ivf_index.nlist) for h in H_GRID})
print(f"H grid: {H_GRID}")
H = H_GRID[0]  # current H; updated by rebuild_centroid_caches() in the run loop

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

# ================= BUILD TOPLOC CACHE =================
# H-independent parts (q0 embeddings, follow-up batches) are built ONCE.
# The centroid caches depend on H, so rebuild_centroid_caches(h) is called
# once per H grid value — this is cheap (a few quantizer searches), unlike
# loading the index.
conv_cache = {}  # conv_id -> cached centroid IDs (int64)      [depends on H]
conv_cvecs = {}  # conv_id -> centroid vectors (float32)       [depends on H]
conv_q0_emb = {}  # conv_id -> q0 embedding (only if judged)
conv_fu_embs = {}  # conv_id -> follow-up embeddings (np array, batched)
conv_fu_keys = {}  # conv_id -> follow-up turn keys

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    followup_keys = [t for t in turns[1:] if t in filtered_qrels]

    if q0_key in filtered_qrels:
        conv_q0_emb[conv_id] = emb_map[q0_key]

    if followup_keys:
        conv_fu_embs[conv_id] = np.vstack(
            [emb_map[tk][0] for tk in followup_keys]
        ).astype("float32")
        conv_fu_keys[conv_id] = followup_keys


def rebuild_centroid_caches(h):
    """Rebuild every conversation's cached centroid set for a new H value.
    Batched: one quantizer search over all q0 embeddings."""
    global H
    H = h
    t0 = time.perf_counter()
    conv_ids = list(conversations.keys())
    q0_matrix_all = np.vstack(
        [emb_map[conversations[c][0]][0] for c in conv_ids]
    ).astype("float32")
    _, c0_indices = ivf_index.quantizer.search(q0_matrix_all, h)
    for row, conv_id in enumerate(conv_ids):
        c0_ids = c0_indices[row].astype("int64")
        conv_cache[conv_id] = c0_ids
        conv_cvecs[conv_id] = get_centroid_vectors(ivf_index.quantizer, c0_ids)
    print(f"  Rebuilt centroid caches for H={h} in {time.perf_counter() - t0:.1f}s")

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

print(f"Eval turns: {len(eval_keys)} (q0={first_n}, follow-up={followup_n})")


# ================= EVAL: BASELINE =================
def eval_baseline(np_val):
    if index_type == "ivf":
        base_index.nprobe = np_val
    ivf_index.nprobe = np_val

    run = defaultdict(dict)
    # WORKAROUND (accuracy only): batched IVF search returns wrong results on
    # the Dragon index (verified: single-query search of the same rows gives
    # 9-10/10 relevant hits, the same rows inside one 173-query batch give 0).
    # Accuracy is therefore computed per query; this is untimed, so the
    # paper-style batched TIMING below is unaffected.
    for row, tk in enumerate(eval_keys):
        sc_row, idx_row = base_index.search(query_matrix[row : row + 1], RETRIEVE_K)
        for idx, sc in zip(idx_row[0], sc_row[0]):
            if idx >= 0 and id_map.get(str(idx)):
                run[tk][id_map[str(idx)]] = float(sc)
    metrics = ir_measures.calc_aggregate(
        [nDCG @ 3, nDCG @ METRIC_K, RR @ METRIC_K], dict(filtered_qrels), dict(run)
    )

    # Paper-style timing: one batch of all q0 queries, one batch of ALL
    # follow-up queries. Per-query time = batch time / batch size.
    q0_matrix_b = np.ascontiguousarray(query_matrix[base_first_rows], dtype="float32")
    fu_rows_flat = [r for rws in base_fu_rows for r in rws]
    fu_matrix_b = np.ascontiguousarray(query_matrix[fu_rows_flat], dtype="float32")

    def sweep():
        t0 = time.perf_counter()
        base_index.search(q0_matrix_b, RETRIEVE_K)
        f_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        base_index.search(fu_matrix_b, RETRIEVE_K)
        u_ms = (time.perf_counter() - t0) * 1000
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


# ================= EVAL: TOPLOC =================
def eval_toploc(np_val):
    ivf_index.nprobe = np_val

    run = defaultdict(dict)
    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        if conv_id in conv_q0_emb:
            s0, i0 = base_index.search(conv_q0_emb[conv_id], RETRIEVE_K)
            for idx, sc in zip(i0[0], s0[0]):
                if idx >= 0 and id_map.get(str(idx)):
                    run[q0_key][id_map[str(idx)]] = float(sc)
        if conv_id in conv_fu_keys:
            D, I = toploc_ivf_search(
                ivf_index,
                conv_fu_embs[conv_id],
                conv_cache[conv_id],
                conv_cvecs[conv_id],
                np_val,
                RETRIEVE_K,
            )
            for r, tk in enumerate(conv_fu_keys[conv_id]):
                for idx, sc in zip(I[r], D[r]):
                    if idx >= 0 and id_map.get(str(idx)):
                        run[tk][id_map[str(idx)]] = float(sc)

    metrics = ir_measures.calc_aggregate(
        [nDCG @ 3, nDCG @ METRIC_K, RR @ METRIC_K], dict(filtered_qrels), dict(run)
    )

    # Paper-style timing (batched):
    #   q0:        one batched cache build (quantizer) + one batched full search
    #   follow-up: numpy coarse scoring against each conversation's H cached
    #              centroids, then ONE global search_preassigned over ALL
    #              follow-up queries (FAISS scans the lists multi-threaded).
    q0_conv_ids = [c for c in conversations if c in conv_q0_emb]
    q0_matrix_t = (
        np.ascontiguousarray(
            np.vstack([conv_q0_emb[c][0] for c in q0_conv_ids]), dtype="float32"
        )
        if q0_conv_ids
        else None
    )
    fu_conv_ids = [c for c in conversations if c in conv_fu_keys]

    def sweep():
        f_ms = 0.0
        if q0_matrix_t is not None:
            t0 = time.perf_counter()
            ivf_index.quantizer.search(q0_matrix_t, H)
            base_index.search(q0_matrix_t, RETRIEVE_K)
            f_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        q_list, ids_list, sc_list = [], [], []
        for conv_id in fu_conv_ids:
            sel_ids, sel_scores = toploc_assign(
                conv_fu_embs[conv_id],
                conv_cache[conv_id],
                conv_cvecs[conv_id],
                np_val,
                USE_IP,
            )
            q_list.append(conv_fu_embs[conv_id])
            ids_list.append(sel_ids)
            sc_list.append(sel_scores)
        preassigned_search(
            ivf_index,
            np.vstack(q_list),
            np.vstack(ids_list),
            np.vstack(sc_list),
            np_val,
            RETRIEVE_K,
        )
        u_ms = (time.perf_counter() - t0) * 1000
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


# ================= EVAL: TOPLOC+ (drift refresh) =================
# Sequential: follow-ups processed one-by-one because a refresh persists to
# later turns. The cache is reset to the q0 state at the start of each sweep.
def eval_toploc_plus(np_val):
    ivf_index.nprobe = np_val

    # ----- build run dict (untimed) + count refreshes -----
    run = defaultdict(dict)
    refresh_count = 0
    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        c_ids = conv_cache[conv_id].copy()
        c_vecs = conv_cvecs[conv_id]
        top_0_local = rank_within_cache(c_vecs, emb_map[q0_key], np_val, USE_IP)

        if conv_id in conv_q0_emb:
            s0, i0 = base_index.search(conv_q0_emb[conv_id], RETRIEVE_K)
            for idx, sc in zip(i0[0], s0[0]):
                if idx >= 0 and id_map.get(str(idx)):
                    run[q0_key][id_map[str(idx)]] = float(sc)

        for tk in turns[1:]:
            if tk not in filtered_qrels:
                continue
            qj = emb_map[tk]
            top_j_local = rank_within_cache(c_vecs, qj, np_val, USE_IP)
            i0_size = len(np.intersect1d(top_j_local, top_0_local))

            if i0_size < ALPHA * np_val:
                _, new_c0 = ivf_index.quantizer.search(qj, H)
                c_ids = new_c0[0].astype("int64")
                c_vecs = get_centroid_vectors(ivf_index.quantizer, c_ids)
                top_0_local = rank_within_cache(c_vecs, qj, np_val, USE_IP)
                refresh_count += 1

            D, I = toploc_ivf_search(ivf_index, qj, c_ids, c_vecs, np_val, RETRIEVE_K)
            for idx, sc in zip(I[0], D[0]):
                if idx >= 0 and id_map.get(str(idx)):
                    run[tk][id_map[str(idx)]] = float(sc)

    metrics = ir_measures.calc_aggregate(
        [nDCG @ 3, nDCG @ METRIC_K, RR @ METRIC_K], dict(filtered_qrels), dict(run)
    )

    # ----- timing (paper-style, batched) -----
    # The drift check + coarse scoring is inherently sequential within a
    # conversation (a refresh changes the cache for later turns), so that part
    # stays in a Python loop. The expensive fine scan over inverted lists is
    # batched into ONE global search_preassigned call at the end.
    q0_conv_ids = [c for c in conversations if c in conv_q0_emb]
    q0_matrix_p = (
        np.ascontiguousarray(
            np.vstack([conv_q0_emb[c][0] for c in q0_conv_ids]), dtype="float32"
        )
        if q0_conv_ids
        else None
    )

    def sweep():
        f_ms = 0.0
        if q0_matrix_p is not None:
            t0 = time.perf_counter()
            ivf_index.quantizer.search(q0_matrix_p, H)
            base_index.search(q0_matrix_p, RETRIEVE_K)
            f_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        q_list, ids_list, sc_list = [], [], []
        for conv_id, turns in conversations.items():
            q0_key = turns[0]
            # reset cache to q0 state for this conversation
            c_ids = conv_cache[conv_id].copy()
            c_vecs = conv_cvecs[conv_id]
            top_0_local = rank_within_cache(c_vecs, emb_map[q0_key], np_val, USE_IP)

            for tk in turns[1:]:
                if tk not in filtered_qrels:
                    continue
                qj = emb_map[tk]
                top_j_local = rank_within_cache(c_vecs, qj, np_val, USE_IP)
                i0_size = len(np.intersect1d(top_j_local, top_0_local))
                if i0_size < ALPHA * np_val:
                    _, new_c0 = ivf_index.quantizer.search(qj, H)
                    c_ids = new_c0[0].astype("int64")
                    c_vecs = get_centroid_vectors(ivf_index.quantizer, c_ids)
                    top_0_local = rank_within_cache(c_vecs, qj, np_val, USE_IP)
                sel_ids, sel_scores = toploc_assign(qj, c_ids, c_vecs, np_val, USE_IP)
                q_list.append(qj)
                ids_list.append(sel_ids)
                sc_list.append(sel_scores)
        preassigned_search(
            ivf_index,
            np.vstack(q_list),
            np.vstack(ids_list),
            np.vstack(sc_list),
            np_val,
            RETRIEVE_K,
        )
        u_ms = (time.perf_counter() - t0) * 1000
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
    return metrics, fu_pq, overall_pq, refresh_count


# ================= RUN =================
nprobe_list = SWEEP_NPROBES if SWEEP else [NP_SINGLE]
rows = []

# Baseline doesn't depend on H or alpha — compute once per nprobe.
baseline_cache = {}


def get_baseline(np_val):
    if np_val not in baseline_cache:
        baseline_cache[np_val] = eval_baseline(np_val)
    return baseline_cache[np_val]


# TopLoc doesn't depend on alpha — compute once per (H, nprobe).
toploc_cache = {}


def get_toploc(h, np_val):
    key = (h, np_val)
    if key not in toploc_cache:
        toploc_cache[key] = eval_toploc(np_val)
    return toploc_cache[key]


# ----- incremental CSV (index loads once; results survive a crash) -----
out_csv = None
if SWEEP:
    os.makedirs("results/raw/ivf/testing", exist_ok=True)
    out_csv = (
        f"results/raw/ivf/testing"
        f"testing_ivf_{model_name}_Hgrid_Agrid_mmap{int(USE_MMAP)}.csv"
    )


def write_csv():
    if out_csv and rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


header = (
    f"{'H':>6} {'alpha':>6} {'nprobe':>6} | {'base ms':>8} {'topl ms':>8} {'topl+ ms':>8} | "
    f"{'spd_T':>6} {'spd_T+':>6} | "
    f"{'b_N3':>6} {'t_N3':>6} {'p_N3':>6} | "
    f"{'b_N10':>6} {'t_N10':>6} {'p_N10':>6} | "
    f"{'b_MRR':>6} {'t_MRR':>6} {'p_MRR':>6} | {'refr':>4}"
)

for h in H_GRID:
    print(f"\n########## H = {h} ##########")
    rebuild_centroid_caches(h)

    for alpha_val in ALPHA_GRID:
        ALPHA = alpha_val
        print(f"\n  ---- alpha = {alpha_val} ----")
        print("=" * 140)
        print(header)
        print("-" * 140)

        for npv in nprobe_list:
            b_m, b_fu, b_ov = get_baseline(npv)
            t_m, t_fu, t_ov = get_toploc(h, npv)
            p_m, p_fu, p_ov, p_refr = eval_toploc_plus(npv)

            spd_t = b_fu / t_fu if t_fu and t_fu == t_fu else float("nan")
            spd_p = b_fu / p_fu if p_fu and p_fu == p_fu else float("nan")

            b3, b10, bmrr = b_m[nDCG @ 3], b_m[nDCG @ METRIC_K], b_m[RR @ METRIC_K]
            t3, t10, tmrr = t_m[nDCG @ 3], t_m[nDCG @ METRIC_K], t_m[RR @ METRIC_K]
            p3, p10, pmrr = p_m[nDCG @ 3], p_m[nDCG @ METRIC_K], p_m[RR @ METRIC_K]

            print(
                f"{h:>6} {alpha_val:>6.2f} {npv:>6} | {b_fu:>8.3f} {t_fu:>8.3f} {p_fu:>8.3f} | "
                f"{spd_t:>5.2f}x {spd_p:>5.2f}x | "
                f"{b3:>6.4f} {t3:>6.4f} {p3:>6.4f} | "
                f"{b10:>6.4f} {t10:>6.4f} {p10:>6.4f} | "
                f"{bmrr:>6.4f} {tmrr:>6.4f} {pmrr:>6.4f} | {p_refr:>4}"
            )

            rows.append(
                {
                    "H": h,
                    "alpha": alpha_val,
                    "nprobe": npv,
                    "baseline_fu_ms": round(b_fu, 4),
                    "toploc_fu_ms": round(t_fu, 4),
                    "toplocplus_fu_ms": round(p_fu, 4),
                    "baseline_overall_ms": round(b_ov, 4),
                    "toploc_overall_ms": round(t_ov, 4),
                    "toplocplus_overall_ms": round(p_ov, 4),
                    "speedup_toploc": round(spd_t, 3),
                    "speedup_toplocplus": round(spd_p, 3),
                    "baseline_NDCG@3": round(b3, 4),
                    "toploc_NDCG@3": round(t3, 4),
                    "toplocplus_NDCG@3": round(p3, 4),
                    "baseline_NDCG@10": round(b10, 4),
                    "toploc_NDCG@10": round(t10, 4),
                    "toplocplus_NDCG@10": round(p10, 4),
                    "baseline_MRR@10": round(bmrr, 4),
                    "toploc_MRR@10": round(tmrr, 4),
                    "toplocplus_MRR@10": round(pmrr, 4),
                    "toplocplus_refreshes": p_refr,
                }
            )

        write_csv()  # flush after every (H, alpha) block

print("=" * 140)
print(
    f"\nH grid: {H_GRID}   |   alpha grid: {ALPHA_GRID}   |   "
    f"k={RETRIEVE_K}   |   threads={NUM_THREADS}   |   "
    f"warmup={BATCH_WARMUP_RUNS} timed={BATCH_TIMED_RUNS}"
)
print(
    "Times = FOLLOW-UP per-query (ms). spd_T / spd_T+ = speedup over baseline. "
    "refr = TopLoc+ cache refreshes."
)

# ----- grid-search winner (paper-style: best effectiveness, then fastest) -----
if rows:
    best_t = max(rows, key=lambda r: (r["toploc_NDCG@10"], -r["toploc_fu_ms"]))
    best_p = max(
        rows, key=lambda r: (r["toplocplus_NDCG@10"], -r["toplocplus_fu_ms"])
    )
    print(
        f"\nBest TopLoc  : H={best_t['H']} nprobe={best_t['nprobe']}  "
        f"NDCG@10={best_t['toploc_NDCG@10']}  "
        f"{best_t['toploc_fu_ms']} ms  ({best_t['speedup_toploc']}x)"
    )
    print(
        f"Best TopLoc+ : H={best_p['H']} alpha={best_p['alpha']} nprobe={best_p['nprobe']}  "
        f"NDCG@10={best_p['toplocplus_NDCG@10']}  "
        f"{best_p['toplocplus_fu_ms']} ms  ({best_p['speedup_toplocplus']}x)"
    )

if out_csv:
    print(f"\nCSV written: {out_csv}  ({len(rows)} rows)")
    print(
        "All three methods: follow-up + overall ms-per-query, speedups, "
        "and NDCG@3 / NDCG@10 / MRR@10 for each (H, alpha, nprobe) combination."
    )
