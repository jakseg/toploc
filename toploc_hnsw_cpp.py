#!/usr/bin/env python3
"""
TopLoc-HNSW evaluation — C++ level-0 search module.

This is the C++ analogue of toploc_hnsw_pure_python.py:
  - q0 of each conversation: normal FAISS HNSW search with higher efSearch.
  - cache q0's best result(s) as privileged entry point(s).
  - follow-up turns: call a pybind11 C++ module that performs HNSW level-0
    beam search from those cached entry point(s).

The evaluation mirrors Jakob's baseline and Ahmad's IVF timing style:
  - build run dict first, untimed, for metrics with ir_measures
  - measure latency separately with warmup + timed sweeps
  - report first-turn, follow-up, and overall latency

Build first:
  mkdir -p build && cd build
  cmake ..
  make -j
  cd ..

If the built .so stays in build/, run with:
  PYTHONPATH=build python -u toploc_hnsw_cpp.py snowflake --max-turns 20

Run examples:
  PYTHONPATH=build python -u toploc_hnsw_cpp.py snowflake --ef-search 64 --up 2 --entry-points 1 --max-turns 20
  PYTHONPATH=build python -u toploc_hnsw_cpp.py snowflake --ef-search 64 --up 2 --entry-points 1
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import faiss
import ir_measures
from ir_measures import nDCG, RR

try:
    from toploc_hnsw_search import toploc_hnsw_level0_search_ptr
except ImportError as exc:
    raise ImportError(
        "Could not import toploc_hnsw_search. Build the C++ module first and set PYTHONPATH.\n"
        "Example:\n"
        "  mkdir -p build && cd build && cmake .. && make -j && cd ..\n"
        "  PYTHONPATH=build python -u toploc_hnsw_cpp.py snowflake --max-turns 20"
    ) from exc


CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc2/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc2/Datasets/conversational/CAST2019/topics"
)
CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

USE_MMAP = os.environ.get("MMAP", "0") == "1"
BATCH_WARMUP_RUNS = int(os.environ.get("BATCH_WARMUP_RUNS", 2))
BATCH_TIMED_RUNS = int(os.environ.get("BATCH_TIMED_RUNS", 5))


def faiss_index_ptr(index):
    """Return raw C++ pointer from a FAISS SWIG Python object."""
    try:
        return int(index.this)
    except Exception:
        return int(faiss.downcast_index(index).this)


def load_query_encoder(model_name):
    """Return a function: list[str] -> np.ndarray of shape (N, dim)."""
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

    if model_name == "dragon":
        import torch
        from transformers import AutoModel, AutoTokenizer

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
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch

    raise ValueError(f"Unknown model: {model_name}")


def load_precomputed_query_embeddings(model_name, keys):
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    emb_path = os.path.join(DATASET_DIR, f"topics_{model_name}_embeddings.parquet")
    if not os.path.exists(emb_path):
        return None

    table = pq.read_table(emb_path)
    cols = table.column_names
    id_col = next((c for c in ("id", "qid", "turn_id", "topic_id", "tid") if c in cols), None)
    emb_col = next((c for c in ("embedding", "embeddings", "vector", "emb") if c in cols), None)
    if id_col is None or emb_col is None:
        return None

    emb_map = {
        str(i): e
        for i, e in zip(table.column(id_col).to_pylist(), table.column(emb_col).to_pylist())
    }
    missing = [k for k in keys if k not in emb_map]
    if missing:
        return None

    matrix = np.ascontiguousarray([emb_map[k] for k in keys], dtype="float32")
    faiss.normalize_L2(matrix)
    print(f"  Loaded precomputed query embeddings {matrix.shape} from {os.path.basename(emb_path)}")
    return matrix


def load_topics(path):
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                topics[parts[0].strip()] = parts[1].strip()
    return topics


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
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
    return qrels


def turn_sort_key(turn_key):
    numbers = re.findall(r"\d+", turn_key)
    return tuple(int(x) for x in numbers) if numbers else (10**9,)


def load_hnsw_level0_graph(index):
    hnsw = index.hnsw
    offsets = faiss.vector_to_array(hnsw.offsets).astype("int64", copy=False)
    neighbors = faiss.vector_to_array(hnsw.neighbors).astype("int64", copy=False)
    degree0 = int(hnsw.nb_neighbors(0))
    return offsets, neighbors, degree0


def cpp_level0_search(index_ptr, graph, q_embs, entry_points, k, ef_search):
    offsets, neighbors, degree0 = graph
    q_c = np.ascontiguousarray(q_embs, dtype="float32")
    ep_c = np.ascontiguousarray(entry_points, dtype="int64")
    off_c = np.ascontiguousarray(offsets, dtype="int64")
    nei_c = np.ascontiguousarray(neighbors, dtype="int64")
    return toploc_hnsw_level0_search_ptr(
        index_ptr, q_c, ep_c, off_c, nei_c, degree0, k, ef_search
    )


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


def per_query_line(stats, n):
    lo, md, mn = stats
    if not n:
        return "n/a"
    return f"{lo / n:8.3f}  /  {md / n:8.3f}  /  {mn / n:8.3f}  ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["snowflake", "dragon"], nargs="?", default="snowflake")
    parser.add_argument("--ef-search", type=int, default=int(os.environ.get("EF_SEARCH", 64)))
    parser.add_argument("--up", type=int, default=int(os.environ.get("UP", 2)))
    parser.add_argument("--entry-points", type=int, default=int(os.environ.get("ENTRY_POINTS", 1)))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=0, help="Debug limit. 0 means all judged turns.")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)
    model_name = args.model
    cache_dir = CACHE_DIRS[model_name]
    index_path = os.path.join(cache_dir, "hnsw_index.index")
    ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
    topics_path = os.path.join(DATASET_DIR, "topics.tsv")
    qrels_path = os.path.join(DATASET_DIR, "qrels.qrel")
    k = args.k

    print(f"Evaluating TopLoc-HNSW C++ for: {model_name}")
    print(f"FAISS threads: {faiss.omp_get_max_threads()}")
    print(f"Index path: {index_path}")

    if USE_MMAP:
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(index_path)
    if not hasattr(index, "hnsw"):
        raise TypeError("Loaded index is not an HNSW index.")
    index.hnsw.efSearch = args.ef_search
    index_ptr = faiss_index_ptr(index)
    print(f"Loaded HNSW index: ntotal={index.ntotal:,}, dim={index.d}, mmap={USE_MMAP}")
    print(f"Base efSearch={args.ef_search}, q0 efSearch={args.ef_search * args.up}")

    print("Loading HNSW level-0 graph arrays...")
    graph = load_hnsw_level0_graph(index)
    print(f"Level-0 degree slots: {graph[2]}")

    id_array = np.load(ids_path, allow_pickle=True)
    id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
    indexed_pids = set(id_map.values())
    print(f"ID map loaded: {len(id_map):,} passages")

    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    filtered_qrels = {
        qid: {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
        for qid, pid_scores in qrels.items()
    }
    filtered_qrels = {qid: vals for qid, vals in filtered_qrels.items() if vals}
    if not topics:
        raise RuntimeError("Parsed 0 topics.")
    if not filtered_qrels:
        raise RuntimeError("No qrels survived filtering against indexed passage ids.")

    conversations = defaultdict(list)
    for turn_key in topics:
        conversations[turn_key.split("_")[0]].append(turn_key)
    for conv_id in conversations:
        conversations[conv_id].sort(key=turn_sort_key)

    # Same eval set style as baseline: only judged turns with relevant pids in index.
    eval_keys_all = [k0 for k0 in topics if k0 in filtered_qrels]
    if args.max_turns and args.max_turns > 0:
        eval_key_set = set(eval_keys_all[: args.max_turns])
    else:
        eval_key_set = set(eval_keys_all)

    print(f"Topics loaded: {len(topics):,} turns")
    print(f"Conversations: {len(conversations):,}")
    print(f"Turns with relevant passages in index: {len(filtered_qrels):,}/{len(qrels):,}")

    # Collect valid turns by conversation after max-turns filter.
    conv_valid = {}
    eval_keys = []
    eval_queries = []
    for conv_id, turns in conversations.items():
        valid = [t for t in turns if t in filtered_qrels and t in eval_key_set]
        if not valid:
            continue
        conv_valid[conv_id] = valid
        for t in valid:
            eval_keys.append(t)
            eval_queries.append(topics[t])

    N = len(eval_keys)
    print(f"Eval set: {N} turns across {len(conv_valid)} conversations")

    print(f"\nObtaining query embeddings for {N} queries...")
    query_matrix = load_precomputed_query_embeddings(model_name, eval_keys)
    if query_matrix is None:
        print(f"Loading {model_name} query encoder...")
        encode_batch = load_query_encoder(model_name)
        t0 = time.perf_counter()
        query_matrix = encode_batch(eval_queries)
        enc_ms = (time.perf_counter() - t0) * 1000
        print(f"Query encoding done in {enc_ms:.1f} ms total ({enc_ms / max(N, 1):.2f} ms/query)")

    emb_lookup = {k0: query_matrix[i : i + 1] for i, k0 in enumerate(eval_keys)}

    # ================= BUILD RUN DICT, UNTIMED =================
    print("\nBuilding run dict (untimed)...")
    run = defaultdict(dict)
    conv_q0_emb = {}
    conv_entry_points = {}
    conv_fu_embs = {}
    conv_fu_keys = {}
    visited_counts = []

    old_ef = index.hnsw.efSearch
    for conv_i, (conv_id, valid_turns) in enumerate(conv_valid.items(), start=1):
        q0_key = valid_turns[0]
        q0_emb = emb_lookup[q0_key]

        index.hnsw.efSearch = args.ef_search * args.up
        q0_scores, q0_indices = index.search(q0_emb, max(k, args.entry_points))
        index.hnsw.efSearch = old_ef

        eps = np.asarray([int(x) for x in q0_indices[0][: args.entry_points] if int(x) >= 0], dtype="int64")
        if len(eps) == 0:
            continue

        conv_q0_emb[conv_id] = q0_emb
        conv_entry_points[conv_id] = eps

        for idx, score in zip(q0_indices[0][:k], q0_scores[0][:k]):
            if idx >= 0 and id_map.get(str(idx)):
                run[q0_key][id_map[str(idx)]] = float(score)

        followup_keys = valid_turns[1:]
        if followup_keys:
            fu_embs = np.vstack([emb_lookup[t] for t in followup_keys]).astype("float32")
            conv_fu_embs[conv_id] = fu_embs
            conv_fu_keys[conv_id] = followup_keys

            # Shared entry points for all follow-ups in this conversation.
            scores, indices, visited = cpp_level0_search(
                index_ptr, graph, fu_embs, eps, k, args.ef_search
            )
            visited_counts.extend([int(x) for x in np.asarray(visited)])
            for row, turn_key in enumerate(followup_keys):
                for idx, score in zip(indices[row], scores[row]):
                    if idx >= 0 and id_map.get(str(idx)):
                        run[turn_key][id_map[str(idx)]] = float(score)

        if conv_i % 5 == 0:
            print(f"  run build progress: conv {conv_i}/{len(conv_valid)}", flush=True)

    index.hnsw.efSearch = old_ef

    first_n = len(conv_q0_emb)
    followup_n = sum(len(v) for v in conv_fu_keys.values())
    print(f"DEBUG: I have {len(run)} turns in my run dict.")

    measures = [nDCG @ 3, nDCG @ k, RR @ k]
    results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

    # ================= TIMING =================
    def timed_sweep():
        first_total_ms = 0.0
        followup_total_ms = 0.0
        old = index.hnsw.efSearch
        for conv_id in conv_q0_emb:
            q0_emb = conv_q0_emb[conv_id]
            index.hnsw.efSearch = args.ef_search * args.up
            t0 = time.perf_counter()
            index.search(q0_emb, max(k, args.entry_points))
            first_total_ms += (time.perf_counter() - t0) * 1000
            index.hnsw.efSearch = old

            if conv_id in conv_fu_embs:
                t0 = time.perf_counter()
                cpp_level0_search(
                    index_ptr,
                    graph,
                    conv_fu_embs[conv_id],
                    conv_entry_points[conv_id],
                    k,
                    args.ef_search,
                )
                followup_total_ms += (time.perf_counter() - t0) * 1000
        index.hnsw.efSearch = old
        return first_total_ms, followup_total_ms

    print(
        f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed "
        "per-conversation C++ sweeps..."
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

    f_min, f_med, f_mean = latency_stats(first_times)
    u_min, u_med, u_mean = latency_stats(followup_times)
    overall_times = [f + u for f, u in zip(first_times, followup_times)]
    o_min, o_med, o_mean = latency_stats(overall_times)

    print("\n" + "=" * 70)
    print(f"TOPLOC-HNSW C++ EVALUATION RESULTS ({model_name})")
    print("=" * 70)
    print(
        f"Turns evaluated: {first_n + followup_n}  "
        f"({first_n} first-turn, {followup_n} follow-up, {len(conv_q0_emb)} conversations)"
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
    print(f"  overall    total:     {o_min:8.2f}  /  {o_med:8.2f}  /  {o_mean:8.2f}  ms")
    print(f"  overall    per query: {per_query_line((o_min, o_med, o_mean), first_n + followup_n)}")
    print(f"Base efSearch:            {args.ef_search}")
    print(f"q0 efSearch:              {args.ef_search * args.up}")
    print(f"Cached entry points/conv: {args.entry_points}")
    if visited_counts:
        print(f"Avg visited nodes:        {np.mean(visited_counts):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
