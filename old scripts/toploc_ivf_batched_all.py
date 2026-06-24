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

USE_MMAP = os.environ.get("MMAP", "0") == "1"

# ================= PARAMETER GRID =================
H_VALUES = [512, 1024, 4096, 8192]
NP_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

# How many timed batch runs to average (same as baseline file)
BATCH_WARMUP_RUNS = 2
BATCH_TIMED_RUNS = 5


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
                    emb = torch.nn.functional.normalize(
                        model(**tokens).last_hidden_state[:, 0, :], p=2, dim=1
                    )
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================= TOPLOC IVF SEARCH =================
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


def toploc_ivf_search_single(index, q_emb, cached_centroid_indices, nprobe, k):
    """Single query search — used for building the run dict (quality metrics)."""
    centroid_vecs = get_centroid_vectors(index.quantizer, cached_centroid_indices)
    use_ip = index.metric_type == faiss.METRIC_INNER_PRODUCT
    if use_ip:
        coarse = (centroid_vecs @ q_emb.T).squeeze(axis=1)
        top_local = np.argpartition(-coarse, min(nprobe, len(coarse) - 1))[:nprobe]
        top_local = top_local[np.argsort(-coarse[top_local])]
    else:
        coarse = ((centroid_vecs - q_emb) ** 2).sum(axis=1)
        top_local = np.argpartition(coarse, min(nprobe, len(coarse) - 1))[:nprobe]
        top_local = top_local[np.argsort(coarse[top_local])]
    sel_centroids = (
        np.asarray(cached_centroid_indices)[top_local].astype("int64").reshape(1, -1)
    )
    sel_coarse = coarse[top_local].astype("float32").reshape(1, -1)
    try:
        scores, indices = index.search_preassigned(q_emb, k, sel_centroids, sel_coarse)
    except TypeError:
        scores, indices = index.search_preassigned(q_emb, k, sel_centroids)
    return scores, indices


def toploc_ivf_search_batch(index, query_matrix, cached_ids_list, nprobe, k):
    """Batch search — used for timing only. Mirrors paper methodology."""
    N = query_matrix.shape[0]
    all_scores = np.full((N, k), -1e38, dtype="float32")
    all_indices = np.full((N, k), -1, dtype="int64")

    for i in range(N):
        q_emb = query_matrix[i : i + 1]
        cached_ids = cached_ids_list[i]
        s, idx = toploc_ivf_search_single(index, q_emb, cached_ids, nprobe, k)
        all_scores[i] = s[0]
        all_indices[i] = idx[0]

    return all_scores, all_indices


# ================= FLEXIBLE PARSERS =================
def split_flexible(line, expected):
    for sep in ("\t", ","):
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) == expected:
            return parts
    parts = line.split()
    if len(parts) == expected:
        return parts
    if expected == 4 and len(parts) > 4:
        return [parts[0], parts[1], " ".join(parts[2:-1]), parts[-1]]
    return None


# =================================================================
# LOAD EVERYTHING ONCE
# =================================================================
print(f"Model: {model_name} | Index: {index_type}")
print("=" * 60)
print("Loading shared resources (index, encoder, topics, qrels)...")

index_path = os.path.join(cache_dir, f"{index_type}_index.index")
if USE_MMAP:
    print("  Loading index with mmap...")
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)
try:
    ivf_index.make_direct_map()
except Exception:
    pass

print(
    f"  Index: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}"
)

ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())
print(f"  ID map: {len(id_map)} passages")

topics = {}
with open(os.path.join(DATASET_DIR, "topics.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = split_flexible(line, 2)
        if parts:
            topics[parts[0]] = parts[1]
if not topics:
    raise RuntimeError("Parsed 0 topics.")
print(f"  Topics: {len(topics)}")

qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = split_flexible(line, 4)
        if not parts:
            continue
        qid, _, pid, score = parts
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
print(f"  Qrels: {len(filtered_qrels)} turns")

conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)
print(f"  Conversations: {len(conversations)} ({len(topics)} turns)")

# Collect eval keys in stable order — same as baseline
eval_keys = [k for k in topics if k in filtered_qrels]
eval_queries = [topics[k] for k in eval_keys]
N = len(eval_queries)

print(f"  Encoding {N} queries (batch)...")
encode_batch = load_query_encoder(model_name)
query_matrix = encode_batch(eval_queries)  # shape (N, dim) — encoded ONCE
print(f"  Query matrix: {query_matrix.shape}")
print("Shared resources loaded.\n")


# =================================================================
# SINGLE EVAL FUNCTION — quality + batch timing
# =================================================================
def run_eval(H, NP):
    ivf_index.nprobe = NP
    h = min(H, ivf_index.nlist)
    k = 10

    # ── PHASE 1: quality metrics ─────────────────────────────────
    # Build centroid cache using Turn 0 of each conversation,
    # then collect run dict for ir_measures.
    # Also collect the per-query cached_ids in eval_keys order
    # so we can reuse them for batch timing in Phase 2.

    conv_cache = {}  # conv_id → cached centroid ids
    run = defaultdict(dict)
    query_cached = {}  # turn_key → cached_ids for that query

    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        if q0_key not in filtered_qrels:
            continue

        # Turn 0 — full search to build cache
        q0_idx = eval_keys.index(q0_key) if q0_key in eval_keys else None
        q0_emb = (
            query_matrix[q0_idx : q0_idx + 1]
            if q0_idx is not None
            else encode_batch([topics[q0_key]])
        )

        _, c0_indices = ivf_index.quantizer.search(q0_emb, h)
        conv_cache[conv_id] = c0_indices[0].astype("int64")
        query_cached[q0_key] = None  # Turn 0 uses full search

        scores, indices = base_index.search(q0_emb, k)
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            pid = id_map.get(str(idx))
            if pid is not None:
                run[q0_key][pid] = float(score)

        # Turns 1+ — restricted search
        for turn_key in turns[1:]:
            if turn_key not in filtered_qrels:
                continue
            t_idx = eval_keys.index(turn_key) if turn_key in eval_keys else None
            q_emb = (
                query_matrix[t_idx : t_idx + 1]
                if t_idx is not None
                else encode_batch([topics[turn_key]])
            )

            query_cached[turn_key] = conv_cache[conv_id]
            scores, indices = toploc_ivf_search_single(
                ivf_index, q_emb, conv_cache[conv_id], NP, k
            )
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                pid = id_map.get(str(idx))
                if pid is not None:
                    run[turn_key][pid] = float(score)

    measures = [nDCG @ 3, nDCG @ k, RR @ k]
    results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

    # ── PHASE 2: batch timing (paper methodology) ────────────────
    # Build ordered list of cached_ids matching eval_keys order.
    # Turn 0 queries use full base_index.search — same as paper baseline.
    follow_up_keys = [k for k in eval_keys if query_cached.get(k) is not None]
    follow_up_matrix = np.vstack(
        [
            query_matrix[eval_keys.index(k) : eval_keys.index(k) + 1]
            for k in follow_up_keys
        ]
    )
    follow_up_caches = [query_cached[k] for k in follow_up_keys]
    Nf = len(follow_up_keys)

    # Warmup
    for _ in range(BATCH_WARMUP_RUNS):
        toploc_ivf_search_batch(ivf_index, follow_up_matrix, follow_up_caches, NP, k)

    # Timed runs
    batch_times = []
    for _ in range(BATCH_TIMED_RUNS):
        t0 = time.perf_counter()
        toploc_ivf_search_batch(ivf_index, follow_up_matrix, follow_up_caches, NP, k)
        batch_times.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(batch_times)) / Nf if Nf > 0 else float("nan")

    return {
        "H": h,
        "NP": NP,
        "ndcg3": results[nDCG @ 3],
        "ndcg10": results[nDCG @ k],
        "mrr10": results[RR @ k],
        "avg_time_ms": avg_ms,
    }


# =================================================================
# PARAMETER SWEEP
# =================================================================
total_runs = len(H_VALUES) * len(NP_VALUES)
print(f"Starting sweep: {len(H_VALUES)} H x {len(NP_VALUES)} NP = {total_runs} runs\n")

all_results = []
run_num = 0

for H in H_VALUES:
    for NP in NP_VALUES:
        run_num += 1
        print(f"[{run_num}/{total_runs}] H={H}, NP={NP} ...", end=" ", flush=True)
        r = run_eval(H, NP)
        all_results.append(r)
        print(
            f"NDCG@10={r['ndcg10']:.4f}  MRR@10={r['mrr10']:.4f}  "
            f"Time={r['avg_time_ms']:.2f}ms"
        )

# =================================================================
# RESULTS TABLE
# =================================================================
print("\n" + "=" * 75)
print(f"TOPLOC-IVF SWEEP RESULTS ({index_type.upper()}, {model_name})")
print("=" * 75)
print(
    f"{'H':>6}  {'NP':>4}  {'NDCG@3':>8}  {'NDCG@10':>8}  "
    f"{'MRR@10':>8}  {'Time(ms)':>10}"
)
print("-" * 75)
for r in all_results:
    print(
        f"{r['H']:>6}  {r['NP']:>4}  {r['ndcg3']:>8.4f}  "
        f"{r['ndcg10']:>8.4f}  {r['mrr10']:>8.4f}  "
        f"{r['avg_time_ms']:>10.2f}"
    )

best = max(all_results, key=lambda x: x["ndcg10"])
print("=" * 75)
print(
    f"BEST NDCG@10: {best['ndcg10']:.4f}  "
    f"->  H={best['H']}, NP={best['NP']}, "
    f"Time={best['avg_time_ms']:.2f}ms"
)
print("=" * 75)
