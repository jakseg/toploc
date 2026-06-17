#!/usr/bin/env python3
"""
TopLoc-HNSW evaluation — PURE PYTHON, no C++ module needed.


Logic:
  - q0 of each conversation: normal FAISS HNSW search with higher efSearch
    (ef_search * up). The best q0 result(s) become privileged entry point(s).
  - follow-up turns: custom Python level-0 HNSW beam search starts from the
    cached q0 entry point(s).
  - evaluation metrics are computed with ir_measures: nDCG@3, nDCG@10, RR@10.
  - latency is measured separately from correctness, using the same
    per-conversation timing style as the baseline:
      first query searched alone, follow-ups handled per conversation.

Important:
  FAISS Python does not expose a public API to search HNSW from a custom entry
  point. Therefore the follow-up search is implemented in Python. This is useful
  to test correctness and TopLoc logic, but real latency should later be tested
  with the C++/native implementation.

Run examples:
    python -u toploc_hnsw_pure_python.py snowflake
    EF_SEARCH=64 UP=2 ENTRY_POINTS=1 python -u toploc_hnsw_pure_python.py snowflake
    MMAP=1 python -u toploc_hnsw_pure_python.py snowflake --max-turns 20
"""

import argparse
import heapq
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import faiss
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

USE_MMAP = os.environ.get("MMAP", "0") == "1"
BATCH_WARMUP_RUNS = int(os.environ.get("BATCH_WARMUP_RUNS", 2))
BATCH_TIMED_RUNS = int(os.environ.get("BATCH_TIMED_RUNS", 5))


# ================= QUERY ENCODER (Batched) =================
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


# ================= PRECOMPUTED QUERY EMBEDDINGS (optional) =================
def load_precomputed_query_embeddings(model_name, keys):
    """Try to load precomputed topic embeddings aligned to `keys`.

    Returns an (N, dim) float32 matrix aligned to keys on full success,
    otherwise None.
    """
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
        print(
            f"  WARN: {os.path.basename(emb_path)} has columns {cols}; "
            "could not find id/embedding columns — encoding instead."
        )
        return None

    emb_map = {
        str(i): e
        for i, e in zip(table.column(id_col).to_pylist(), table.column(emb_col).to_pylist())
    }
    missing = [k for k in keys if k not in emb_map]
    if missing:
        print(
            f"  WARN: {len(missing)}/{len(keys)} eval turns missing from "
            f"{os.path.basename(emb_path)} (e.g. {missing[:3]}) — encoding instead."
        )
        return None

    matrix = np.ascontiguousarray([emb_map[k] for k in keys], dtype="float32")
    faiss.normalize_L2(matrix)
    print(f"  Loaded precomputed query embeddings {matrix.shape} from {os.path.basename(emb_path)}")
    return matrix


# ================= FILE LOADERS =================
def load_topics(path):
    """Load topics.tsv. In this project it is comma-separated: turn_id,query."""
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
    """Load qrels.qrel. In this project it is comma-separated: qid,iter,pid,score."""
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
    """Sort keys like '31_1', '31_2', ... in conversation order."""
    numbers = re.findall(r"\d+", turn_key)
    return tuple(int(x) for x in numbers) if numbers else (10**9,)


# ================= HNSW GRAPH ACCESS =================
def load_hnsw_level0_graph(index):
    """Copy FAISS HNSW graph arrays once instead of on every neighbor lookup."""
    hnsw = index.hnsw
    offsets = faiss.vector_to_array(hnsw.offsets).astype("int64", copy=False)
    neighbors = faiss.vector_to_array(hnsw.neighbors).astype("int64", copy=False)
    degree0 = int(hnsw.nb_neighbors(0))
    return offsets, neighbors, degree0


def level0_neighbors(graph, node_id):
    offsets, neighbors, degree0 = graph
    start = int(offsets[int(node_id)])
    block = neighbors[start : start + degree0]
    return block[block >= 0]


def reconstruct_batch_safe(index, node_ids):
    """Return vectors for node_ids using batched FAISS reconstruction when possible."""
    ids = np.asarray(node_ids, dtype="int64")
    try:
        return index.reconstruct_batch(ids).astype("float32")
    except Exception:
        vecs = np.empty((len(ids), index.d), dtype="float32")
        for i, node_id in enumerate(ids):
            index.reconstruct(int(node_id), vecs[i])
        return vecs


# ================= TOPLOC-HNSW LEVEL-0 SEARCH =================
def toploc_hnsw_level0_search(index, graph, q_emb, entry_points, k=10, ef_search=64):
    """Pure Python HNSW level-0 beam search from custom entry points.

    Parameters:
        q_emb: shape (1, dim) or (dim,)
        entry_points: cached q0 result ids used as privileged entry points

    Returns:
        scores: shape (1, k)
        indices: shape (1, k)
        visited_count: number of level-0 nodes visited by the Python beam search
    """
    query_vec = q_emb.reshape(-1).astype("float32")

    candidates = []  # (-score, node_id), best score pops first
    results = []     # (score, node_id), worst current result is results[0]
    visited = set()

    def add_candidates(node_ids):
        new_ids = []
        for node_id in node_ids:
            node_id = int(node_id)
            if node_id >= 0 and node_id not in visited:
                visited.add(node_id)
                new_ids.append(node_id)

        if not new_ids:
            return

        vecs = reconstruct_batch_safe(index, new_ids)
        scores = vecs @ query_vec

        for node_id, score in zip(new_ids, scores):
            score = float(score)
            heapq.heappush(candidates, (-score, node_id))
            if len(results) < ef_search:
                heapq.heappush(results, (score, node_id))
            elif score > results[0][0]:
                heapq.heapreplace(results, (score, node_id))

    add_candidates(entry_points)

    while candidates:
        neg_score, current = heapq.heappop(candidates)
        current_score = -neg_score

        if len(results) >= ef_search and current_score < results[0][0]:
            break

        add_candidates(level0_neighbors(graph, current))

    top = sorted(results, key=lambda x: x[0], reverse=True)[:k]

    scores = np.full((1, k), -np.inf, dtype="float32")
    indices = np.full((1, k), -1, dtype="int64")
    for rank, (score, node_id) in enumerate(top):
        scores[0, rank] = score
        indices[0, rank] = node_id

    return scores, indices, len(visited)


# ================= QLR STEP 1 — QUERY LOG INDEX (I_Q) =================
def build_query_log_index(log_emb, m=32, ef_construction=500, ef_search=64):
    """Build I_Q, the query-log index: an HNSW graph over the historical query
    vectors Q_L. A new incoming query is searched here first, to find similar
    past queries (Algorithm 1, line 1).

    log_emb: (n_log, d) float32, L2-normalised historical query vectors.

    Metric is METRIC_INNER_PRODUCT, matching the document index. Because the
    vectors are L2-normalised, an inner-product search returns cosine
    similarities directly — so the top-1 score is the routing similarity s, on
    the same [-1, 1] scale as the threshold th and the reference s_max used in
    later steps.

    HNSW knobs — none of these is the number of vectors; the index holds every
    vector passed to add():
        m               graph degree (edges per node; level 0 gets 2*m). Paper: 32.
        ef_construction build-time candidate list — graph quality vs build time. Paper: 500.
        ef_search       query-time beam width — recall vs speed. Paper sweeps 10–200.
    """
    d = log_emb.shape[1]
    iq = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    iq.hnsw.efConstruction = ef_construction
    iq.add(np.ascontiguousarray(log_emb, dtype="float32"))
    iq.hnsw.efSearch = ef_search
    return iq


# ================= QLR STEP 2 — LOOKUP TABLE (EP) + s_max =================
def build_lookup_table(index_d, log_emb, k_ep=10, ep_build_ef_search=512, smax_percentile=75):
    """Build EP (the lookup table) and the reference similarity s_max.

    For each historical query in Q_L, search the document index I_D for its
    k_ep nearest documents. Those document node ids are the precomputed entry 
    points reused at query time as the seed set C — so at query time we skip 
    the costly top-down HNSW traversal and jump straight into the right neighbourhood.

    log_emb: (n_log, d) float32, L2-normalised historical query vectors.

    Returns:
        ep_table: (n_log, k_ep) int64 — document node ids per log query.
        s_max:    float — the smax_percentile-th percentile of each log query's
                  top-1 similarity to a document; calibrates the adaptive
                  efSearch in the search phase.

    k_ep is how many neighbours we store (paper: 10). ep_build_ef_search is the
    one-time offline build-quality knob (beam width, must be >= k_ep); higher =
    better stored neighbours. Not a paper value — the paper only sweeps the
    query-time efSearch over [10, 200].
    """
    old_ef = index_d.hnsw.efSearch
    index_d.hnsw.efSearch = max(ep_build_ef_search, k_ep)
    ep_scores, ep_ids = index_d.search(
        np.ascontiguousarray(log_emb, dtype="float32"), k_ep
    )
    index_d.hnsw.efSearch = old_ef

    ep_table = ep_ids.astype("int64")                       # (n_log, k_ep) doc node ids
    s_max = float(np.percentile(ep_scores[:, 0], smax_percentile))
    return ep_table, s_max


# ================= QLR STEP 3 — ROUTING (Algorithm 1) =================
def adaptive_ef_search(s, s_max, th, ef_default, ef_min):
    """Shrink the beam width as routing confidence s grows (Alg. 1, lines 6-9).
        s >= s_max -> ef_min       (very confident: seeds are basically on target)
        s == th    -> ef_default   (barely routed: search as widely as usual)
        between    -> linear interpolation.
    The mechanism + the (s_max - s)/(s_max - th) factor are from the paper.
    ef_min is our choice (the paper interpolates down to a minimum but doesn't pin it).
    """
    if s_max <= th:                              # degenerate calibration
        return ef_min
    frac = (s_max - s) / (s_max - th)            # 0 at s_max, 1 at th
    frac = min(max(frac, 0.0), 1.0)              # clamp: s may exceed s_max
    return int(round(ef_min + (ef_default - ef_min) * frac))


def route(q, iq, ep_table, index_d, graph, s_max,
          k=10, k_prime=10, th=0.5, ef_default=100, ef_min=10):
    """Route one query (Alg. 1). Returns (scores, indices, visited, routed, s);
    visited is None on the fallback path (native FAISS gives no count).

    ef_default is the upper bound of the adaptive ef' and the efSearch used on
    the fallback path; set it to the HNSW baseline operating point you compare
    against (the efSearch sweep includes 512).
    """
    q = np.ascontiguousarray(q, dtype="float32").reshape(1, -1)

    # 1-2. Closest historical queries. I_Q is inner-product on normalised
    #      vectors, so the returned score IS the cosine similarity s.
    sims, log_idx = iq.search(q, k_prime)
    s = float(sims[0, 0])
    matched = [int(i) for i in log_idx[0] if i >= 0]

    # 3-4. Not similar enough -> plain HNSW search on the document index.
    if not matched or s < th:
        old_ef = index_d.hnsw.efSearch
        index_d.hnsw.efSearch = ef_default
        f_scores, f_idx = index_d.search(q, k)
        index_d.hnsw.efSearch = old_ef
        return f_scores[0], f_idx[0], None, False, s

    # 5. Candidate set C = union of the matched log queries' precomputed docs.
    C = np.unique(ep_table[matched].ravel())
    C = C[C >= 0]

    # 6-9 / 10. Adaptive beam width, then seeded level-0 search from C.
    ef = max(adaptive_ef_search(s, s_max, th, ef_default, ef_min), k)
    scores, idx, visited = toploc_hnsw_level0_search(
        index_d, graph, q, entry_points=C, k=k, ef_search=ef
    )
    return scores[0], idx[0], visited, True, s


# ================= UTILS =================
def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


def per_query_line(stats, n):
    lo, md, mn = stats
    if not n:
        return "n/a"
    return f"{lo / n:8.3f}  /  {md / n:8.3f}  /  {mn / n:8.3f}  ms"


def add_to_run(run, turn_key, scores_row, indices_row, id_map):
    for idx, score in zip(indices_row, scores_row):
        if int(idx) < 0:
            continue
        pid = id_map.get(str(int(idx)))
        if pid is not None:
            run[turn_key][pid] = float(score)


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["snowflake", "dragon"], nargs="?", default="snowflake")
    parser.add_argument("--ef-search", type=int, default=int(os.environ.get("EF_SEARCH", 64)))
    parser.add_argument("--up", type=int, default=int(os.environ.get("UP", 2)))
    parser.add_argument("--entry-points", type=int, default=int(os.environ.get("ENTRY_POINTS", 1)))
    parser.add_argument("--k", type=int, default=int(os.environ.get("K", 10)))
    parser.add_argument("--max-turns", type=int, default=0, help="Debug limit. 0 means all eval turns.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", os.cpu_count() or 1)))
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)

    model_name = args.model
    cache_dir = CACHE_DIRS[model_name]
    index_path = os.path.join(cache_dir, "hnsw_index.index")
    ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
    topics_path = os.path.join(DATASET_DIR, "topics.tsv")
    qrels_path = os.path.join(DATASET_DIR, "qrels.qrel")

    print(f"Evaluating TopLoc-HNSW PURE PYTHON for: {model_name}", flush=True)
    print(f"FAISS threads: {faiss.omp_get_max_threads()}", flush=True)
    print(f"Index path: {index_path}", flush=True)

    if not os.path.exists(index_path):
        print(f"ERROR: HNSW index not found: {index_path}", flush=True)
        print(f"Run first: python create_index.py {model_name} hnsw", flush=True)
        sys.exit(1)

    if USE_MMAP:
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(index_path)

    if not hasattr(index, "hnsw"):
        raise TypeError("Loaded index is not an HNSW index.")

    index.hnsw.efSearch = args.ef_search
    print(
        f"Loaded HNSW index: ntotal={index.ntotal:,}, dim={index.d}, mmap={USE_MMAP}",
        flush=True,
    )
    print(
        f"Base efSearch={args.ef_search}, q0 efSearch={args.ef_search * args.up}, "
        f"entry_points={args.entry_points}",
        flush=True,
    )

    print("Loading HNSW level-0 graph arrays...", flush=True)
    graph = load_hnsw_level0_graph(index)
    print(f"Level-0 degree slots: {graph[2]}", flush=True)

    id_array = np.load(ids_path, allow_pickle=True)
    id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
    indexed_pids = set(id_map.values())
    print(f"ID map loaded: {len(id_map):,} passages", flush=True)

    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    if not topics:
        raise RuntimeError("Parsed 0 topics. Check topics.tsv delimiter/format.")

    filtered_qrels = {}
    for turn_key, pid_scores in qrels.items():
        valid_pids = {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
        if valid_pids:
            filtered_qrels[turn_key] = valid_pids

    if not filtered_qrels:
        raise RuntimeError("No qrels survived filtering against indexed passage ids.")

    # Stable evaluation order: topics order, only judged turns.
    eval_keys = [k for k in topics if k in filtered_qrels]
    if args.max_turns:
        eval_keys = eval_keys[: args.max_turns]
    eval_queries = [topics[k] for k in eval_keys]
    N = len(eval_keys)

    print(f"Topics loaded: {len(topics):,} turns", flush=True)
    print(f"Turns with relevant passages in index: {len(filtered_qrels):,}/{len(qrels):,}", flush=True)
    print(f"Eval turns used: {N:,}", flush=True)

    # ================= OBTAIN QUERY EMBEDDINGS =================
    print(f"\nObtaining query embeddings for {N} queries...", flush=True)
    query_matrix = load_precomputed_query_embeddings(model_name, eval_keys)
    if query_matrix is None:
        print(f"Loading {model_name} query encoder...", flush=True)
        encode_batch = load_query_encoder(model_name)
        print(f"Batch-encoding {N} queries...", flush=True)
        t0 = time.perf_counter()
        query_matrix = encode_batch(eval_queries)
        enc_ms = (time.perf_counter() - t0) * 1000
        print(
            f"Query encoding done in {enc_ms:.1f} ms total "
            f"({enc_ms / N:.2f} ms/query) — shape={query_matrix.shape}",
            flush=True,
        )

    # Group encoded rows by conversation, preserving eval order.
    conv_rows = defaultdict(list)
    for row, key in enumerate(eval_keys):
        conv_rows[key.split("_")[0]].append(row)
    for conv_id in conv_rows:
        conv_rows[conv_id].sort(key=lambda r: turn_sort_key(eval_keys[r]))

    first_rows = [rows[0] for rows in conv_rows.values()]
    followup_rows_per_conv = [rows[1:] for rows in conv_rows.values()]
    first_n = len(first_rows)
    followup_n = sum(len(rows) for rows in followup_rows_per_conv)
    print(
        f"\nEval set: {N} turns ({first_n} first-turn, {followup_n} follow-up) "
        f"across {len(conv_rows)} conversations",
        flush=True,
    )

    k = args.k

    # ================= BUILD RUN (correctness, untimed) =================
    # For TopLoc-HNSW, q0 must be searched first to get the privileged entry
    # point. Follow-ups use the same cached entry point(s).
    print("\nBuilding run dict with TopLoc-HNSW logic (untimed)...", flush=True)
    run = defaultdict(dict)
    conv_entry_points = {}
    conv_q0_emb = {}
    conv_followup_embs = {}
    conv_followup_keys = {}
    conv_visited_counts = defaultdict(list)

    old_ef = index.hnsw.efSearch

    for conv_i, (conv_id, rows) in enumerate(conv_rows.items(), start=1):
        q0_row = rows[0]
        q0_key = eval_keys[q0_row]
        q0_emb = query_matrix[q0_row : q0_row + 1]
        conv_q0_emb[conv_id] = q0_emb

        # q0: full/native HNSW with upscaled efSearch.
        index.hnsw.efSearch = args.ef_search * args.up
        q0_scores, q0_indices = index.search(q0_emb, max(k, args.entry_points))
        index.hnsw.efSearch = old_ef

        entry_points = [int(x) for x in q0_indices[0][: args.entry_points] if int(x) >= 0]
        if not entry_points:
            continue
        conv_entry_points[conv_id] = entry_points
        add_to_run(run, q0_key, q0_scores[0][:k], q0_indices[0][:k], id_map)

        fu_rows = rows[1:]
        if fu_rows:
            fu_embs = np.ascontiguousarray(query_matrix[fu_rows], dtype="float32")
            fu_keys = [eval_keys[r] for r in fu_rows]
            conv_followup_embs[conv_id] = fu_embs
            conv_followup_keys[conv_id] = fu_keys

            # The pure Python search handles one query at a time, but the
            # follow-up embeddings are stored as the per-conversation batch.
            for local_i, turn_key in enumerate(fu_keys):
                scores, indices, visited = toploc_hnsw_level0_search(
                    index=index,
                    graph=graph,
                    q_emb=fu_embs[local_i : local_i + 1],
                    entry_points=entry_points,
                    k=k,
                    ef_search=args.ef_search,
                )
                add_to_run(run, turn_key, scores[0], indices[0], id_map)
                conv_visited_counts[conv_id].append(visited)

        if conv_i % 5 == 0 or conv_i == len(conv_rows):
            print(f"  built run for conv {conv_i}/{len(conv_rows)}", flush=True)

    index.hnsw.efSearch = old_ef

    print(f"DEBUG: I have {len(run)} turns in my run dict.", flush=True)

    measures = [nDCG @ 3, nDCG @ k, RR @ k]
    results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

    # ================= TIMING: PER-CONVERSATION SWEEP =================
    def timed_sweep():
        """One full pass: q0 single + follow-up batch per conversation.

        In this pure Python version, follow-up embeddings are batched per
        conversation, but the custom level-0 search is still called once per
        follow-up query because FAISS Python has no custom-entry-point batch API.
        """
        first_total_ms, followup_total_ms = 0.0, 0.0
        visited_counts = []

        for conv_id, rows in conv_rows.items():
            if conv_id not in conv_entry_points:
                continue

            q0_emb = conv_q0_emb[conv_id]
            old = index.hnsw.efSearch
            index.hnsw.efSearch = args.ef_search * args.up
            t0 = time.perf_counter()
            index.search(q0_emb, max(k, args.entry_points))
            first_total_ms += (time.perf_counter() - t0) * 1000
            index.hnsw.efSearch = old

            if conv_id in conv_followup_embs:
                fu_embs = conv_followup_embs[conv_id]
                entry_points = conv_entry_points[conv_id]
                t0 = time.perf_counter()
                for i in range(fu_embs.shape[0]):
                    _, _, visited = toploc_hnsw_level0_search(
                        index=index,
                        graph=graph,
                        q_emb=fu_embs[i : i + 1],
                        entry_points=entry_points,
                        k=k,
                        ef_search=args.ef_search,
                    )
                    visited_counts.append(visited)
                followup_total_ms += (time.perf_counter() - t0) * 1000

        return first_total_ms, followup_total_ms, visited_counts

    print(
        f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed "
        "per-conversation sweeps...",
        flush=True,
    )
    for _ in range(BATCH_WARMUP_RUNS):
        timed_sweep()

    first_times, followup_times, all_visited = [], [], []
    for run_i in range(BATCH_TIMED_RUNS):
        f_ms, u_ms, visited = timed_sweep()
        first_times.append(f_ms)
        followup_times.append(u_ms)
        all_visited.extend(visited)
        f_pq = f_ms / first_n if first_n else float("nan")
        u_pq = u_ms / followup_n if followup_n else float("nan")
        print(
            f"  run {run_i + 1}: first-turn total={f_ms:7.2f} ms (per-query={f_pq:.3f}) | "
            f"follow-up total={u_ms:7.2f} ms (per-query={u_pq:.3f})",
            flush=True,
        )

    f_min, f_med, f_mean = latency_stats(first_times)
    u_min, u_med, u_mean = latency_stats(followup_times)
    overall_times = [f + u for f, u in zip(first_times, followup_times)]
    o_min, o_med, o_mean = latency_stats(overall_times)

    print("\n" + "=" * 70)
    print(f"TOPLOC-HNSW PURE PYTHON RESULTS ({model_name})")
    print("=" * 70)
    print(
        f"Turns evaluated: {N}  ({first_n} first-turn, {followup_n} follow-up, "
        f"{len(conv_rows)} conversations)"
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
    print(
        f"  overall    per query: {per_query_line((o_min, o_med, o_mean), N)}"
        "   <- compare to paper Time"
    )
    print()
    print(f"Base efSearch:            {args.ef_search}")
    print(f"q0 efSearch:              {args.ef_search * args.up}")
    print(f"Cached entry points/conv: {args.entry_points}")
    if all_visited:
        print(f"Avg visited nodes:        {np.mean(all_visited):.1f}")
    print("=" * 70)
    print("NOTE: Follow-up search uses Python level-0 beam search.")
    print("It tests TopLoc-HNSW correctness, but real latency needs the C++/native version.")


if __name__ == "__main__":
    main()
