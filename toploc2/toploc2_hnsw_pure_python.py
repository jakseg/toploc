#!/usr/bin/env python3
"""
QLR (Query Log Router) evaluation — PURE PYTHON, no C++ module needed.

Implements "HNSW Graph Meets Query Logs" (the toploc2 paper) on top of a
standard HNSW document index. Algorithm 1, per query:
  - search a query-log index I_Q for the closest historical queries; let s be
    the similarity to the closest one.
  - if s < th: fall back to a plain HNSW search on the document index.
  - else: gather the precomputed document neighbours of the matched log queries
    (lookup table EP) into a seed set C, shrink efSearch adaptively from s, and
    run a level-0 beam search on the document index seeded from C.

The seeded beam search reuses the TopLoc-HNSW level-0 kernel (FAISS exposes no
API to seed HNSW entry points). Metrics via ir_measures (nDCG@3/10, RR@10) plus
Accuracy@10 vs exhaustive search (the paper's headline metric — needs a ground
truth from compute_groundtruth.py). Latency is split into routed vs fallback.

Modes:
  --mode qlr (default)   route every query through I_Q (Algorithm 1).
  --mode baseline        plain HNSW, no routing — the apples-to-apples comparison.
  --sweep                build EP/s_max (and I_Q per PCA dim) once, then sweep the
                         paper grid: th, k', ef-search, PCA on/off -> CSV + table.
  --pca-dim N            reduce the I_Q query vectors to N dims (paper dim/4).

dragon vs snowflake: dragon is raw dot-product (not L2-normalised — like the
paper's Contriever), so its th is on the dot-product scale and is calibrated from
the data when unset; snowflake is cosine with th in [0.3, 0.7].

Run examples:
    python -u toploc2_hnsw_pure_python.py snowflake --dataset msmarco-on-cast
    # plain-HNSW baseline curve (sweep ef-search):
    MMAP=1 python -u toploc2_hnsw_pure_python.py snowflake --dataset msmarco-on-cast \
        --mode baseline --sweep
    # full QLR sweep (th x k' x ef x PCA), capped log for a quick run:
    MMAP=1 python -u toploc2_hnsw_pure_python.py snowflake --dataset msmarco-on-cast \
        --sweep --log-limit 100000
"""

import argparse
import glob
import heapq
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
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
    # dragon points at the preserved L2-NORMALISED (cosine) build. The un-normalised
    # dot-product indexes at .../dragon/ break HNSW/IVF graph search (~0 recall);
    # the correct normalised indexes are in the _cosine_old/ subfolder. EP/I_Q/GT
    # caches are written here too. After a clean rebuild with the fixed
    # create_index.py (normalize_vecs=True) this can go back to just "dragon".
    "dragon": os.path.join(CACHE_BASE, "dragon", "_cosine_old"),
}

# msmarco (QLR / toploc2): the collection HNSW index lives in its own cache; the
# dev queries (test set), train queries (Q_L = historical log) and dev qrels live
# under MSMARCO_BASE. All query embeddings are parquet shards (id + embedding).
MSMARCO_CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "msmarco", "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "msmarco", "dragon"),
}
MSMARCO_BASE = os.environ.get(
    "MSMARCO_BASE", "/home/toploc2/Datasets/conversational/CAST2019/msmarco"
)
# dev queries (test set), per encoder. snowflake was precomputed (1024-d); the
# dragon set (768-d) is produced once by encode_msmarco_dev_dragon.py into
# dev_query_dragon/ (no precomputed dragon dev embeddings exist on the server).
MSMARCO_DEV_QUERY_DIRS = {
    "snowflake": os.path.join(MSMARCO_BASE, "msmarco_embeddings", "dev_query"),
    "dragon": os.path.join(MSMARCO_BASE, "msmarco_embeddings", "dev_query_dragon"),
}
MSMARCO_TRAIN_QUERY_DIR = os.path.join(MSMARCO_BASE, "msmarco_embeddings", "train_query")
MSMARCO_QRELS = os.path.join(MSMARCO_BASE, "qrels.dev.small.tsv")

# Full msmarco train-query log (~808k rows = the paper's |Q_L|), separate from
# the 367k train_query subset above. Used by --dataset msmarco-on-cast.
MSMARCO_FULL_LOG_DIRS = {
    "snowflake": "/home/toploc2/Datasets/conversational/msmarco/snowflake",
    "dragon": "/home/toploc2/Datasets/conversational/msmarco/dragon",
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


def load_qrels(path, pid_prefix=""):
    """Load qrels: `qid iter pid score`. Delimiter-flexible — CAST uses commas
    (`31_1,0,CAR_..,1`), msmarco uses tabs (`174249\t0\t1092925\t1`).

    pid_prefix is prepended to every pid. Used by msmarco-on-cast: the msmarco
    qrels list bare numeric pids (`1092925`), but the CAST2019 index stores the
    same passage as `MARCO_1092925`, so mapping `1092925` -> `MARCO_1092925`
    aligns the qrels with the indexed pid space (the run side already emits the
    `MARCO_<n>` form via id_map, so both sides match)."""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"[\t,]", line.strip())
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
                qrels[qid][pid_prefix + pid] = score
    return qrels


def load_parquet_embeddings(emb_dir, normalize=True):
    """Load all *.parquet shards in emb_dir → (ids, (N,d) float32).

    Used for the msmarco query-embedding dirs (dev_query = test, train_query =
    Q_L). Each shard has an `id` (string) and `embedding` (list<float>) column;
    the shards are concatenated in filename order.

    normalize: L2-normalise the vectors (cosine scale). True for snowflake
    (arctic-embed is trained for cosine). MUST be False for dragon — dragon-plus
    is trained for raw dot product, and the dragon document index is built
    un-normalised (see create_index.py: `normalize_vecs = model_name != "dragon"`).
    Normalising dragon here would put s / s_max / I_Q on a different scale than
    the dragon I_D, breaking the th comparison.
    """
    files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {emb_dir}")
    ids, mats = [], []
    for f in files:
        t = pq.read_table(f, columns=["id", "embedding"])
        ids.extend(str(x) for x in t.column("id").to_pylist())
        d = len(t.column("embedding")[0].as_py())
        try:
            flat = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
            mats.append(flat.reshape(-1, d).astype("float32"))
        except Exception:
            mats.append(np.array(t.column("embedding").to_pylist(), dtype="float32"))
    emb = np.ascontiguousarray(np.vstack(mats), dtype="float32")
    if normalize:
        faiss.normalize_L2(emb)
    return ids, emb


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


# ================= LEVEL-0 SEARCH VIA FAISS (real C++ latency) =================
def faiss_level0_search(index, graph, q_emb, entry_points, k=10, ef_search=64):
    """Seeded level-0 HNSW beam search via FAISS' built-in `search_level_0`.

    Drop-in replacement for `toploc_hnsw_level0_search` (same signature + return
    shape), but the beam search runs inside FAISS' compiled C++ instead of the
    pure-Python loop -> REAL latency. `graph` is unused (kept for parity).

    Recipe (verified identical top-k vs the Python beam): `search_type=2` enqueues
    all entry points (= seed the beam from C); the seed distances are NEGATED inner
    products because FAISS HNSW is internally "smaller = better". Assumes
    METRIC_INNER_PRODUCT (every index in this repo). Returns (scores, indices,
    visited) with visited=None — FAISS exposes no visited count.
    """
    q = np.ascontiguousarray(q_emb, dtype="float32").reshape(1, -1)
    C = np.asarray(entry_points, dtype="int64")
    C = np.ascontiguousarray(C[C >= 0].astype("int32"))
    nprobe = int(C.shape[0])
    if nprobe == 0:
        return (np.full((1, k), -np.inf, "float32"),
                np.full((1, k), -1, "int64"), None)
    seed_vecs = reconstruct_batch_safe(index, C.astype("int64"))
    nearest_d = np.ascontiguousarray((-(seed_vecs @ q[0])).astype("float32"))
    D = np.empty((1, k), dtype="float32")
    I = np.empty((1, k), dtype="int64")
    old_ef = index.hnsw.efSearch
    index.hnsw.efSearch = int(ef_search)
    try:
        args = (1, faiss.swig_ptr(q), k,
                faiss.swig_ptr(C), faiss.swig_ptr(nearest_d),
                faiss.swig_ptr(D), faiss.swig_ptr(I), nprobe, 2)
        try:
            index.search_level_0(*args)
        except TypeError:          # some builds want the params arg explicitly
            index.search_level_0(*args, None)
    finally:
        index.hnsw.efSearch = old_ef
    return D, I, None


def faiss_level0_search_batch(index, queries, entry_point_lists, ef_search, k=10):
    """One `search_level_0` call over m queries -> FAISS allocates its
    VisitedTable(ntotal) ONCE for the whole batch instead of once per query. This
    is the paper-faithful way to TIME the seeded search: the per-query cost with
    the scratch buffer amortized, single-threaded (the paper's "average query
    latency"). The single-query `faiss_level0_search` re-allocates that table on
    every call -> a benchmark artifact (~8-18 ms flat on the 38.6M index), NOT the
    algorithm. See the latency notes in CLAUDE.md.

    All m queries are searched at the SAME ef_search, so the caller groups routed
    queries by their adaptive ef' first. Seed sets C have variable length; shorter
    ones are padded up to the batch max with -1 (FAISS' "no entry" marker, which the
    beam skips) so the batch has a uniform nprobe WITHOUT polluting results with
    duplicate seeds. Results are IDENTICAL to looping `faiss_level0_search`
    (asserted in test_qlr_pipeline).

    queries: (m, d) float32. entry_point_lists: list of m int arrays (the seed set
    C per query). Returns (D, I) of shape (m, k); rows with an empty C are blanked.
    """
    m = len(entry_point_lists)
    if m == 0:
        return np.empty((0, k), "float32"), np.empty((0, k), "int64")
    Q = np.ascontiguousarray(queries, dtype="float32").reshape(m, -1)
    cleaned = [np.asarray(c, dtype="int64") for c in entry_point_lists]
    cleaned = [c[c >= 0] for c in cleaned]
    nprobe = max((len(c) for c in cleaned), default=0)
    D = np.full((m, k), -np.inf, dtype="float32")
    I = np.full((m, k), -1, dtype="int64")
    if nprobe == 0:
        return D, I
    empty = [i for i, c in enumerate(cleaned) if len(c) == 0]
    nearest = np.full((m, nprobe), -1, dtype="int32")       # -1 = padding, skipped by the beam
    for i, c in enumerate(cleaned):
        nearest[i, :len(c)] = c.astype("int32")
    # seed distances for the whole batch at once: negated inner products (FAISS HNSW
    # is "smaller = better"; assumes METRIC_INNER_PRODUCT). Padded (-1) slots get
    # +inf so they are never selected even by a build that does not skip them.
    recon = np.where(nearest.reshape(-1) < 0, 0, nearest.reshape(-1)).astype("int64")
    vecs = reconstruct_batch_safe(index, recon).reshape(m, nprobe, -1)
    nearest_d = (-(np.einsum("mpd,md->mp", vecs, Q))).astype("float32")
    nearest_d[nearest < 0] = np.inf
    nearest_d = np.ascontiguousarray(nearest_d)
    nearest = np.ascontiguousarray(nearest)
    old_ef = index.hnsw.efSearch
    index.hnsw.efSearch = int(ef_search)
    try:
        args = (m, faiss.swig_ptr(Q), k,
                faiss.swig_ptr(nearest), faiss.swig_ptr(nearest_d),
                faiss.swig_ptr(D), faiss.swig_ptr(I), nprobe, 2)
        try:
            index.search_level_0(*args)
        except TypeError:                               # some builds want the params arg
            index.search_level_0(*args, None)
    finally:
        index.hnsw.efSearch = old_ef
    for i in empty:
        D[i] = -np.inf
        I[i] = -1
    return D, I


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
          k=10, k_prime=10, th=0.5, ef_default=100, ef_min=10,
          pca=None, log_emb_full=None, level0_backend="python"):
    """Route one query (Alg. 1). Returns (scores, indices, visited, routed, s);
    visited is None on the fallback path (native FAISS gives no count).

    ef_default is the upper bound of the adaptive ef' and the efSearch used on
    the fallback path; set it to the HNSW baseline operating point you compare
    against (the efSearch sweep includes 512).

    PCA (optional optimisation, paper §"Practical Considerations"): when `pca` is
    given, I_Q lives in the reduced space, so the incoming query is projected
    (and L2-normalised) before the I_Q candidate search. But the routing
    similarity `s` is then RECOMPUTED in the full-dimensional native space
    (against the matched log vector in `log_emb_full`) so it stays on the same
    scale as `th` and `s_max` — the PCA-space score is not. The seeded beam
    search always runs on the full-dim document index with the full-dim query.
    """
    q_full = np.ascontiguousarray(q, dtype="float32").reshape(1, -1)

    # 1-2. Closest historical queries via I_Q.
    if pca is not None:
        q_iq = pca_apply(pca, q_full)
        faiss.normalize_L2(q_iq)
    else:
        # No PCA: I_Q is inner-product on the native query vectors, so the
        # returned score IS s directly (cosine for snowflake, raw dot for dragon).
        q_iq = q_full
    sims, log_idx = iq.search(q_iq, k_prime)
    matched = [int(i) for i in log_idx[0] if i >= 0]

    if pca is not None and matched:
        # Recompute s in the full-dim native space so it is comparable to th/s_max.
        s = float(np.dot(q_full[0], log_emb_full[matched[0]]))
    else:
        s = float(sims[0, 0])

    # 3-4. Not similar enough -> plain HNSW search on the document index.
    if not matched or s < th:
        old_ef = index_d.hnsw.efSearch
        index_d.hnsw.efSearch = ef_default
        f_scores, f_idx = index_d.search(q_full, k)
        index_d.hnsw.efSearch = old_ef
        return f_scores[0], f_idx[0], None, False, s

    # 5. Candidate set C = union of the matched log queries' precomputed docs.
    C = np.unique(ep_table[matched].ravel())
    C = C[C >= 0]

    # 6-9 / 10. Adaptive beam width, then seeded level-0 search from C. The beam
    # search backend is swappable: the pure-Python loop (validates correctness) or
    # FAISS' built-in search_level_0 (real C++ latency, identical top-k).
    ef = max(adaptive_ef_search(s, s_max, th, ef_default, ef_min), k)
    level0_fn = faiss_level0_search if level0_backend == "faiss" else toploc_hnsw_level0_search
    scores, idx, visited = level0_fn(
        index_d, graph, q_full, entry_points=C, k=k, ef_search=ef
    )
    return scores[0], idx[0], visited, True, s


# ================= QLR STEP 4a — QUERY LOG SOURCE (placeholder) =================
def build_query_log(query_matrix, source="self"):
    """Return the historical query log Q_L as an (n_log, dim) normalised matrix.

    PLACEHOLDER — the real historical log (a held-out / external query set) is
    not wired up yet. With source="self" we reuse the evaluation queries
    themselves so the whole QLR pipeline runs end-to-end. This is degenerate
    (every query then has a near-identical match in the log, so everything
    routes) and is only for plumbing/sanity checks, not real numbers. Wire a
    real log in here later.
    """
    if source == "self":
        print(
            "  WARN: using EVAL queries as the query log (placeholder Q_L). "
            "Wire a real historical log here later."
        )
        return np.ascontiguousarray(query_matrix, dtype="float32")
    raise NotImplementedError(f"query-log source '{source}' not wired up yet")


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


# ================= PCA ON I_Q (optional) =================
def pca_apply(pca, x):
    """Project x with a trained faiss.PCAMatrix (handles the apply/apply_py rename)."""
    x = np.ascontiguousarray(x, dtype="float32")
    try:
        return pca.apply(x)
    except AttributeError:
        return pca.apply_py(x)


def build_pca(log_emb_full, pca_dim):
    """Fit a PCA projection (dim -> pca_dim) on the query log Q_L (paper: dim/4)."""
    d = log_emb_full.shape[1]
    pca = faiss.PCAMatrix(d, pca_dim)
    pca.train(np.ascontiguousarray(log_emb_full, dtype="float32"))
    return pca


def build_iq_for_pca(log_emb_full, pca):
    """Build I_Q in the PCA-reduced space (projected + L2-normalised log vectors).
    I_Q here only does coarse 'have I seen a similar query?' matching; the routing
    similarity s is recomputed full-dim in route()."""
    log_proj = pca_apply(pca, log_emb_full)
    faiss.normalize_L2(log_proj)
    return build_query_log_index(log_proj)


# ================= QLR ARTIFACT CACHE (EP + I_Q) =================
# EP (the ~80-min |Q_L| document searches) and I_Q (the HNSW over Q_L) are the
# only expensive build-time artifacts, and they depend solely on the document
# index and the query log — not on the th/k'/ef/PCA sweep. So build each once and
# cache it to disk; later runs load it in seconds instead of rebuilding.
#
# EP additionally stores FAISS document NODE ids, which are tied to one specific
# hnsw_index.index. If that index is ever rebuilt the ids go stale silently, so
# the EP cache records the index file's mtime+size and is rebuilt on a mismatch.
def _index_signature(index_path):
    st = os.stat(index_path)
    return int(st.st_mtime), int(st.st_size)


def ep_cache_path(cache_dir, model, dataset, k_ep, ep_build_ef_search, log_limit):
    return os.path.join(
        cache_dir,
        f"ep_{model}_{dataset}_kep{k_ep}_ef{ep_build_ef_search}_log{log_limit}.npz",
    )


def load_or_build_ep(index, log_emb, cache_dir, model, dataset, index_path,
                     k_ep=10, ep_build_ef_search=512, smax_percentile=75,
                     log_limit=0, rebuild=False):
    """Return (ep_table, s_max), loading a cached EP when one matches this
    (model, dataset, k_ep, ep_build_ef_search, log_limit) AND the document index
    is unchanged; otherwise run the ~80-min build and cache the result. The
    mtime+size guard catches a rebuilt hnsw_index.index (whose node ids, stored
    in ep_table, would otherwise be silently stale). log_limit=0 = full log."""
    path = ep_cache_path(cache_dir, model, dataset, k_ep, ep_build_ef_search, log_limit)
    mtime, size = _index_signature(index_path)
    if not rebuild and os.path.exists(path):
        d = np.load(path)
        if int(d["id_mtime"]) == mtime and int(d["id_size"]) == size:
            ep_table = d["ep_table"]
            s_max = float(d["s_max"])
            print(f"  EP loaded from cache: {os.path.basename(path)} "
                  f"({len(ep_table):,} log queries, s_max={s_max:.4f})", flush=True)
            return ep_table, s_max
        print(f"  EP cache ignored (document index changed since it was built) "
              f"-> rebuilding {os.path.basename(path)}", flush=True)
    ep_table, s_max = build_lookup_table(
        index, log_emb, k_ep=k_ep, ep_build_ef_search=ep_build_ef_search,
        smax_percentile=smax_percentile)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, ep_table=ep_table, s_max=np.float64(s_max),
             id_mtime=np.int64(mtime), id_size=np.int64(size))
    print(f"  EP saved to cache: {os.path.basename(path)}", flush=True)
    return ep_table, s_max


def iq_cache_path(cache_dir, model, dataset, log_limit, pca_dim):
    return os.path.join(
        cache_dir, f"iq_{model}_{dataset}_log{log_limit}_pca{pca_dim}.index")


def load_or_build_iq(log_emb, cache_dir, model, dataset, log_limit, pca_dim,
                     rebuild=False):
    """Return (iq, pca). Cache the HNSW query-log index (faiss.write_index) and,
    for pca>0, the trained PCA matrix next to it. Keyed by
    (model, dataset, log_limit, pca_dim) — I_Q is built over the query log only,
    so it needs no document-index guard."""
    path = iq_cache_path(cache_dir, model, dataset, log_limit, pca_dim)
    pca_path = path[: -len(".index")] + ".pca"
    has_pca = bool(pca_dim and pca_dim > 0)
    if (not rebuild and os.path.exists(path)
            and (not has_pca or os.path.exists(pca_path))):
        iq = faiss.read_index(path)
        pca = faiss.read_VectorTransform(pca_path) if has_pca else None
        print(f"  I_Q loaded from cache: {os.path.basename(path)}", flush=True)
        return iq, pca
    if has_pca:
        pca = build_pca(log_emb, pca_dim)
        iq = build_iq_for_pca(log_emb, pca)
    else:
        pca = None
        iq = build_query_log_index(log_emb)
    os.makedirs(cache_dir, exist_ok=True)
    faiss.write_index(iq, path)
    if pca is not None:
        faiss.write_VectorTransform(pca, pca_path)
    print(f"  I_Q saved to cache: {os.path.basename(path)}", flush=True)
    return iq, pca


# ================= GROUND TRUTH / ACCURACY@10 =================
def groundtruth_path(cache_dir, model_name, dataset, k):
    """Where compute_groundtruth.py writes the exact top-k for the dev queries."""
    return os.path.join(cache_dir, f"groundtruth_dev_top{k}_{model_name}_{dataset}.json")


def load_groundtruth(path):
    """Load exact top-k pids per dev-query id (dict key -> [pid, ...]); None if absent.
    Produced by compute_groundtruth.py in the same MARCO_<n> pid space as the run."""
    if not os.path.exists(path):
        return None
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def accuracy_at_k(run, gt, eval_keys, k):
    """Paper metric: mean fraction of the true top-k retrieved (vs exhaustive GT).
    Returns nan if no ground truth is available."""
    if not gt:
        return float("nan")
    accs = []
    for key in eval_keys:
        true = gt.get(key)
        if not true:
            continue
        true_set = set(true[:k])
        got = set(run.get(key, {}).keys())
        accs.append(len(got & true_set) / float(k))
    return float(np.mean(accs)) if accs else float("nan")


def compute_metrics(run, filtered_qrels, gt, eval_keys, k):
    """nDCG@3, nDCG@k, RR@k (qrels-based relevance) + Accuracy@k (vs exhaustive)."""
    measures = [nDCG @ 3, nDCG @ k, RR @ k]
    res = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))
    return {
        "NDCG@3": float(res[nDCG @ 3]),
        f"NDCG@{k}": float(res[nDCG @ k]),
        f"MRR@{k}": float(res[RR @ k]),
        f"Accuracy@{k}": accuracy_at_k(run, gt, eval_keys, k),
    }


# ================= RUN BUILDERS =================
def build_run_baseline(index, query_matrix, eval_keys, id_map, k, ef_search):
    """Plain HNSW search (no routing) at a given efSearch — the apples-to-apples
    baseline the paper compares QLR against. Returns (run, total_ms)."""
    old_ef = index.hnsw.efSearch
    index.hnsw.efSearch = ef_search
    t0 = time.perf_counter()
    scores, idxs = index.search(np.ascontiguousarray(query_matrix, dtype="float32"), k)
    total_ms = (time.perf_counter() - t0) * 1000
    index.hnsw.efSearch = old_ef
    run = defaultdict(dict)
    for row, key in enumerate(eval_keys):
        add_to_run(run, key, scores[row], idxs[row], id_map)
    return run, total_ms


def build_run_qlr(index, graph, query_matrix, eval_keys, id_map, iq, ep_table, s_max,
                  k, k_prime, th, ef_default, ef_min, pca=None, log_emb_full=None,
                  level0_backend="python"):
    """Route every eval query (Alg. 1) and assemble the run dict, timing routed vs
    fallback. Returns (run, route_flags, sims, visited_counts, routed_ms, fallback_ms).
    level0_backend selects the routed beam search engine ('python' or 'faiss')."""
    run = defaultdict(dict)
    route_flags, sims, visited_counts = [], [], []
    routed_ms = fallback_ms = 0.0
    for row, key in enumerate(eval_keys):
        t0 = time.perf_counter()
        scores_row, idx_row, visited, routed, s = route(
            query_matrix[row:row + 1], iq, ep_table, index, graph, s_max,
            k=k, k_prime=k_prime, th=th, ef_default=ef_default, ef_min=ef_min,
            pca=pca, log_emb_full=log_emb_full, level0_backend=level0_backend,
        )
        dt = (time.perf_counter() - t0) * 1000
        add_to_run(run, key, scores_row, idx_row, id_map)
        route_flags.append(routed)
        sims.append(s)
        if routed:
            routed_ms += dt
        else:
            fallback_ms += dt
        if visited is not None:
            visited_counts.append(visited)
    return run, route_flags, sims, visited_counts, routed_ms, fallback_ms


def measure_qlr_latency_faithful(index, query_matrix, eval_keys, id_map, iq, ep_table, s_max,
                                 k, k_prime, th, ef_default, ef_min, pca=None, log_emb_full=None):
    """Paper-faithful QLR latency: the per-query cost with the FAISS per-call scratch
    (VisitedTable(ntotal)) AMORTIZED, single-threaded — i.e. the paper's "average
    query latency". Every FAISS op is BATCHED so a search *call* allocates its
    ntotal-sized table ~once per phase instead of once per query (the single-query
    build_run_qlr/route path re-allocates it every query -> the flat ~18 ms
    artifact; see CLAUDE.md). Set faiss threads to 1 for a paper-comparable number.

    Phases (each timed): (1) routing = PCA projection + I_Q search + s recompute,
    batched over all queries; (2) fallback = one batched plain index.search at
    ef_default over the un-routed queries; (3) seeded = routed queries grouped by
    their adaptive ef', one batched search_level_0 per group. The candidate-set
    union (Alg. 1 line 5) is Python glue and left UNtimed (a compiled impl amortizes
    it); the seed-vector reconstruct IS timed (real per-query work).

    Returns (run, route_flags, sims, timing, routed_n). The run's top-k is identical
    to build_run_qlr's (asserted in test_qlr_pipeline), so metrics come from it too.
    timing has per-query ms: latency_ms_per_q (total) + qlr_route/seeded/fallback.
    """
    N = len(eval_keys)
    Q = np.ascontiguousarray(query_matrix, dtype="float32")

    # ---- Phase 1: routing (PCA projection + I_Q search + s), BATCHED ----
    t0 = time.perf_counter()
    if pca is not None:
        Qiq = np.ascontiguousarray(pca_apply(pca, Q), dtype="float32")
        faiss.normalize_L2(Qiq)
    else:
        Qiq = Q
    sims_iq, logidx = iq.search(Qiq, k_prime)
    if pca is not None:
        first = logidx[:, 0]
        s = np.full(N, -np.inf, dtype="float64")
        valid = first >= 0
        s[valid] = np.einsum("nd,nd->n", Q[valid].astype("float64"),
                             log_emb_full[first[valid]].astype("float64"))
    else:
        s = sims_iq[:, 0].astype("float64")
    route_ms = (time.perf_counter() - t0) * 1000

    routed_mask = (logidx[:, 0] >= 0) & (s >= th)
    routed_idx = np.nonzero(routed_mask)[0]
    fb_idx = np.nonzero(~routed_mask)[0]
    run = defaultdict(dict)

    # ---- Phase 2: fallback = one batched plain HNSW search at ef_default ----
    fallback_ms = 0.0
    if len(fb_idx):
        old_ef = index.hnsw.efSearch
        index.hnsw.efSearch = ef_default
        t0 = time.perf_counter()
        f_scores, f_idx = index.search(np.ascontiguousarray(Q[fb_idx]), k)
        fallback_ms = (time.perf_counter() - t0) * 1000
        index.hnsw.efSearch = old_ef
        for r, i in enumerate(fb_idx):
            add_to_run(run, eval_keys[i], f_scores[r], f_idx[r], id_map)

    # ---- Phase 3: seeded search, routed queries grouped by ef', BATCHED ----
    seeded_ms = 0.0
    if len(routed_idx):
        C_list, ef_of = [], np.empty(len(routed_idx), dtype="int64")
        for j, i in enumerate(routed_idx):                     # untimed glue (union of EP)
            matched = [int(x) for x in logidx[i] if x >= 0]
            C = np.unique(ep_table[matched].ravel())
            C_list.append(C[C >= 0])
            ef_of[j] = max(adaptive_ef_search(float(s[i]), s_max, th, ef_default, ef_min), k)
        for ef_val in np.unique(ef_of):                        # ef' mostly collapses -> few calls
            gsel = np.nonzero(ef_of == ef_val)[0]
            Qg = Q[routed_idx[gsel]]
            Cg = [C_list[j] for j in gsel]
            t0 = time.perf_counter()
            Dg, Ig = faiss_level0_search_batch(index, Qg, Cg, ef_search=int(ef_val), k=k)
            seeded_ms += (time.perf_counter() - t0) * 1000
            for r, j in enumerate(gsel):
                add_to_run(run, eval_keys[int(routed_idx[j])], Dg[r], Ig[r], id_map)

    routed_n = int(len(routed_idx))
    total_ms = route_ms + fallback_ms + seeded_ms
    timing = {
        "latency_ms_per_q": round(total_ms / N, 4) if N else float("nan"),
        "qlr_route_ms_per_q": round(route_ms / N, 4) if N else float("nan"),
        "qlr_seeded_ms_per_q": round(seeded_ms / routed_n, 4) if routed_n else float("nan"),
        "qlr_fallback_ms_per_q": round(fallback_ms / len(fb_idx), 4) if len(fb_idx) else float("nan"),
    }
    return run, routed_mask.tolist(), s.tolist(), timing, routed_n


def topk_match_rate(run_a, run_b, keys, k):
    """Mean |top-k(a) ∩ top-k(b)| / k over `keys` (1.0 = the two backends return
    identical result sets). Used to confirm the FAISS backend matches the Python
    beam search before trusting its latency."""
    if not keys:
        return float("nan")
    fracs = []
    for key in keys:
        a = set(sorted(run_a.get(key, {}), key=lambda p: -run_a[key][p])[:k])
        b = set(sorted(run_b.get(key, {}), key=lambda p: -run_b[key][p])[:k])
        if a or b:
            fracs.append(len(a & b) / max(k, 1))
    return float(np.mean(fracs)) if fracs else float("nan")


def routing_similarities(query_matrix, iq, k_prime, pca=None, log_emb_full=None):
    """Top-1 I_Q similarity s for every eval query (full-dim recompute under PCA).
    Used to calibrate a data-driven th on the dragon (raw dot-product) scale."""
    s_vals = []
    for row in range(query_matrix.shape[0]):
        q_full = np.ascontiguousarray(query_matrix[row:row + 1], dtype="float32")
        if pca is not None:
            q_iq = pca_apply(pca, q_full)
            faiss.normalize_L2(q_iq)
        else:
            q_iq = q_full
        sims, log_idx = iq.search(q_iq, k_prime)
        if pca is not None and log_idx[0, 0] >= 0:
            s_vals.append(float(np.dot(q_full[0], log_emb_full[int(log_idx[0, 0])])))
        else:
            s_vals.append(float(sims[0, 0]))
    return np.asarray(s_vals, dtype="float64")


# ================= SWEEP OUTPUT =================
SWEEP_COLUMNS = [
    "model", "dataset", "mode", "th", "k_prime", "ef", "pca",
    "route_rate", "NDCG@3", "NDCG@10", "MRR@10", "Accuracy@10",
    "avg_visited",
    # PAPER-FAITHFUL latency (batched, threads=1, VisitedTable amortized) — THE
    # comparison column: baseline = batched ms/q; qlr = route+seeded+fallback.
    "latency_ms_per_q", "qlr_route_ms_per_q", "qlr_seeded_ms_per_q", "qlr_fallback_ms_per_q",
    # single-query (indicative / the ~18 ms artifact) — only under --compare-backends.
    "routed_ms_per_q", "fallback_ms_per_q",
    "routed_ms_faiss_per_q", "speedup_routed", "topk_match",
]


def write_sweep_results(rows, out_path):
    """Write the sweep rows to CSV (same csv.DictWriter convention as
    combine_all3 / combine_base_top_hnsw; creates the parent dir if needed)
    and print a compact markdown table."""
    import csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in SWEEP_COLUMNS})
    print(f"\nCSV written: {out_path} ({len(rows)} rows)", flush=True)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if not np.isnan(v) else "nan"
        return str(v)

    print("\n| " + " | ".join(SWEEP_COLUMNS) + " |")
    print("|" + "|".join(["---"] * len(SWEEP_COLUMNS)) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c, "")) for c in SWEEP_COLUMNS) + " |")


# ================= MAIN (QLR STEPS 4b–4d) =================
def _parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["snowflake", "dragon"], nargs="?", default="snowflake")
    parser.add_argument("--k", type=int, default=int(os.environ.get("K", 10)))
    parser.add_argument("--k-prime", type=int, default=int(os.environ.get("K_PRIME", 10)),
                        help="k': log queries retrieved from I_Q per incoming query")
    parser.add_argument("--th", type=float,
                        default=(float(os.environ["TH"]) if os.environ.get("TH") else None),
                        help="similarity threshold; below it the query is not routed. "
                             "Unset -> 0.5 for snowflake (cosine); data-driven for dragon "
                             "(raw dot-product scale, 25th pct of the routing similarities).")
    parser.add_argument("--ef-default", type=int, default=int(os.environ.get("EF_DEFAULT", 100)),
                        help="upper bound of adaptive ef' and the fallback efSearch")
    parser.add_argument("--ef-min", type=int, default=int(os.environ.get("EF_MIN", 10)),
                        help="lower bound of adaptive ef' (confident routes)")
    parser.add_argument("--k-ep", type=int, default=int(os.environ.get("K_EP", 10)),
                        help="entry points stored per log query in EP")
    parser.add_argument("--pca-dim", type=int, default=int(os.environ.get("PCA_DIM", 0)),
                        help="reduce I_Q query vectors to this dim via PCA (0=off; paper dim/4)")
    parser.add_argument("--level0-backend", choices=["python", "faiss"],
                        default=os.environ.get("LEVEL0_BACKEND", "python"),
                        help="routed beam search engine: 'python' (validates "
                             "correctness) or 'faiss' (search_level_0, real C++ latency)")
    parser.add_argument("--compare-backends", action="store_true",
                        default=os.environ.get("COMPARE_BACKENDS", "") == "1",
                        help="in --sweep: also run the FAISS backend and add "
                             "routed_ms_faiss_per_q / speedup_routed / topk_match columns")
    parser.add_argument("--mode", choices=["qlr", "baseline", "both"], default=os.environ.get("MODE", "qlr"),
                        help="qlr (routed, default), baseline (plain HNSW), or both "
                             "(baseline + qlr in ONE process so the index loads once)")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep hyperparameters, building EP/s_max (and I_Q per PCA) once")
    parser.add_argument("--th-list", default=os.environ.get("TH_LIST", ""),
                        help="comma th values for --sweep (default: paper grid / dragon pct grid)")
    parser.add_argument("--kprime-list", default=os.environ.get("KPRIME_LIST", ""),
                        help="comma k' values for --sweep (default: 10,20)")
    parser.add_argument("--ef-list", default=os.environ.get("EF_LIST", ""),
                        help="comma ef values for --sweep (ef_default / baseline efSearch; default 10..200)")
    parser.add_argument("--pca-list", default=os.environ.get("PCA_LIST", ""),
                        help="comma PCA dims for --sweep (default: 0,dim/4 to see with vs without)")
    parser.add_argument("--out", default=os.environ.get("OUT", ""),
                        help="results CSV path for --sweep (default: auto-named in cwd)")
    parser.add_argument("--log-source", default=os.environ.get("LOG_SOURCE", "self"),
                        help="where Q_L comes from for cast2019 (placeholder; 'self')")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "cast2019"),
                        choices=["cast2019", "msmarco", "msmarco-on-cast"],
                        help="cast2019 (default); msmarco (own QLR collection + train log); "
                             "or msmarco-on-cast (CAST2019 index as I_D — it contains the "
                             "msmarco passages as MARCO_<n> — with the msmarco log/dev/qrels)")
    parser.add_argument("--log-limit", type=int, default=int(os.environ.get("LOG_LIMIT", 0)),
                        help="cap |Q_L| for quick runs (0 = full log)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="ignore any cached EP/I_Q artifacts and rebuild them "
                             "from scratch, then overwrite the cache")
    parser.add_argument("--max-turns", type=int, default=0, help="Debug limit. 0 means all eval turns.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", os.cpu_count() or 1)))
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)

    model_name = args.model
    dataset = args.dataset
    # Both models use cosine (L2-normalised) now: dragon-plus HNSW/IVF navigation
    # requires normalised vectors — raw inner product breaks the graph (the paper
    # L2-normalises Dragon before indexing; see create_index.py). Queries must be
    # on the same cosine scale as the doc index so s, s_max and th are consistent.
    normalize = True
    # msmarco-on-cast searches the CAST2019 index (which already contains every
    # msmarco passage as MARCO_<n>, plus the CAR passages) using the msmarco
    # log/dev/qrels. Both msmarco modes share the parquet query loaders and the
    # msmarco qrels; they differ only in (a) which HNSW index is I_D and (b)
    # whether qrels pids need the MARCO_ prefix to match the indexed pids.
    msmarco_queries = dataset in ("msmarco", "msmarco-on-cast")
    on_cast = dataset == "msmarco-on-cast"
    cache_dir = (MSMARCO_CACHE_DIRS if dataset == "msmarco" else CACHE_DIRS)[model_name]
    index_path = os.path.join(cache_dir, "hnsw_index.index")
    ids_path = os.path.join(cache_dir, "hnsw_ids.npy")

    print(f"Evaluating QLR (Query Log Router) PURE PYTHON for: {model_name} [{dataset}]", flush=True)
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
    print(
        f"Loaded HNSW index: ntotal={index.ntotal:,}, dim={index.d}, mmap={USE_MMAP}",
        flush=True,
    )

    print("Loading HNSW level-0 graph arrays...", flush=True)
    graph = load_hnsw_level0_graph(index)
    print(f"Level-0 degree slots: {graph[2]}", flush=True)

    id_array = np.load(ids_path, allow_pickle=True)
    id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
    indexed_pids = set(id_map.values())
    print(f"ID map loaded: {len(id_map):,} passages", flush=True)

    # ---- qrels (delimiter-flexible: CAST commas / msmarco tabs) ----
    qrels_path = MSMARCO_QRELS if msmarco_queries else os.path.join(DATASET_DIR, "qrels.qrel")
    qrels = load_qrels(qrels_path, pid_prefix="MARCO_" if on_cast else "")
    filtered_qrels = {}
    for turn_key, pid_scores in qrels.items():
        valid_pids = {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
        if valid_pids:
            filtered_qrels[turn_key] = valid_pids
    if not filtered_qrels:
        raise RuntimeError("No qrels survived filtering against indexed passage ids.")

    # ---- eval queries (test set) + their embeddings ----
    if msmarco_queries:
        # dev queries: precomputed embeddings in parquet shards, keyed by query id.
        dev_dir = MSMARCO_DEV_QUERY_DIRS[model_name]
        print(f"\nLoading msmarco dev-query embeddings from {dev_dir}...", flush=True)
        q_ids, q_emb = load_parquet_embeddings(dev_dir, normalize=normalize)
        id2row = {qid: i for i, qid in enumerate(q_ids)}
        eval_keys = [qid for qid in q_ids if qid in filtered_qrels]
        if args.max_turns:
            eval_keys = eval_keys[: args.max_turns]
        query_matrix = np.ascontiguousarray(
            q_emb[[id2row[k] for k in eval_keys]], dtype="float32"
        )
        N = len(eval_keys)
        print(f"Dev queries with qrels in index: {N:,} (of {len(q_ids):,} embedded)", flush=True)
    else:
        topics = load_topics(os.path.join(DATASET_DIR, "topics.tsv"))
        if not topics:
            raise RuntimeError("Parsed 0 topics. Check topics.tsv delimiter/format.")
        eval_keys = [k for k in topics if k in filtered_qrels]
        if args.max_turns:
            eval_keys = eval_keys[: args.max_turns]
        eval_queries = [topics[k] for k in eval_keys]
        N = len(eval_keys)
        print(f"Topics: {len(topics):,} | eval turns: {N:,}", flush=True)
        print(f"\nObtaining query embeddings for {N} queries...", flush=True)
        query_matrix = load_precomputed_query_embeddings(model_name, eval_keys)
        if query_matrix is None:
            print(f"Loading {model_name} query encoder...", flush=True)
            encode_batch = load_query_encoder(model_name)
            t0 = time.perf_counter()
            query_matrix = encode_batch(eval_queries)
            enc_ms = (time.perf_counter() - t0) * 1000
            print(f"Query encoding done in {enc_ms:.1f} ms ({enc_ms / N:.2f} ms/query)", flush=True)

    if N == 0:
        raise RuntimeError("No eval queries after aligning topics/qrels/embeddings.")

    k = args.k

    # ---- ground truth for Accuracy@k (exhaustive top-k from compute_groundtruth.py) ----
    gt_path = groundtruth_path(cache_dir, model_name, dataset, k)
    gt = load_groundtruth(gt_path)
    if gt is not None:
        n_gt = sum(1 for key in eval_keys if gt.get(key))
        print(f"Ground truth for Accuracy@{k}: {n_gt:,}/{N:,} eval queries "
              f"({os.path.basename(gt_path)})", flush=True)
    else:
        print(f"No ground truth at {gt_path} -> Accuracy@{k} will be nan "
              f"(run compute_groundtruth.py to enable it)", flush=True)

    def make_row(mode, th, k_prime, ef, pca_dim, metrics, route_rate, avg_visited,
                 routed_ms, fallback_ms, routed_n, fallback_n, timing=None):
        row = {
            "model": model_name, "dataset": dataset, "mode": mode,
            "th": "" if th is None else round(float(th), 4),
            "k_prime": "" if k_prime is None else k_prime,
            "ef": ef, "pca": pca_dim, "route_rate": round(route_rate, 4),
            "NDCG@3": round(metrics.get("NDCG@3", float("nan")), 4),
            "NDCG@10": round(metrics.get(f"NDCG@{k}", float("nan")), 4),
            "MRR@10": round(metrics.get(f"MRR@{k}", float("nan")), 4),
            "Accuracy@10": round(metrics.get(f"Accuracy@{k}", float("nan")), 4),
            "avg_visited": round(avg_visited, 2) if avg_visited == avg_visited else float("nan"),
            "routed_ms_per_q": round(routed_ms / routed_n, 4) if routed_n else float("nan"),
            "fallback_ms_per_q": round(fallback_ms / fallback_n, 4) if fallback_n else float("nan"),
        }
        if timing:
            row.update(timing)
        return row

    # Default output mirrors combine_all3 / combine_base_top_hnsw: results/raw/<kind>/
    # with the run params + mmap flag in the name. --out still overrides.
    out_path = args.out or os.path.join(
        "results", "raw", "qlr",
        f"qlr_{model_name}_{dataset}_{args.mode}_log{args.log_limit}_mmap{int(USE_MMAP)}.csv")

    # ================= BASELINE: plain HNSW (the apples-to-apples comparison) =================
    # --mode both runs baseline AND qlr in this same process, so the giant index is
    # loaded only ONCE; the rows are merged into a single CSV at the end.
    baseline_rows = []
    if args.mode in ("baseline", "both"):
        ef_list_b = (_parse_int_list(args.ef_list) if args.ef_list else
                     (list(range(10, 201, 10)) if args.sweep else [args.ef_default]))
        print(f"\n[baseline] plain HNSW search (no routing), efSearch in {ef_list_b}", flush=True)
        for ef in ef_list_b:
            run, total_ms = build_run_baseline(index, query_matrix, eval_keys, id_map, k, ef)
            metrics = compute_metrics(run, filtered_qrels, gt, eval_keys, k)
            row = make_row("baseline", None, None, ef, 0, metrics,
                           route_rate=0.0, avg_visited=float("nan"),
                           routed_ms=0.0, fallback_ms=total_ms, routed_n=0, fallback_n=N,
                           timing={"latency_ms_per_q": round(total_ms / N, 4)})
            baseline_rows.append(row)
            print(f"  ef={ef:4d}  NDCG@{k}={row['NDCG@10']:.4f}  MRR@{k}={row['MRR@10']:.4f}  "
                  f"Acc@{k}={row['Accuracy@10']:.4f}  {total_ms / N:.3f} ms/q", flush=True)
        if args.mode == "baseline":
            if args.sweep:
                write_sweep_results(baseline_rows, out_path)
            print("\nNOTE: pure-Python; batch latency is indicative only. The metrics "
                  "(NDCG/MRR/Accuracy) are real and directly comparable to the QLR run.")
            return
        # mode == "both": the index stays loaded — fall through to QLR, merge at the end.

    # ================= QLR: build EP/s_max once, then route =================
    # Q_L: msmarco uses the train-query split (the historical log); cast2019 uses
    # the placeholder (--log-source self reuses the eval queries).
    print("\nBuilding Query Log Router...", flush=True)
    if msmarco_queries:
        log_dir = MSMARCO_FULL_LOG_DIRS[model_name] if on_cast else MSMARCO_TRAIN_QUERY_DIR
        print(f"Loading msmarco train-query log from {log_dir}...", flush=True)
        _, log_emb = load_parquet_embeddings(log_dir, normalize=normalize)
    else:
        log_emb = build_query_log(query_matrix, source=args.log_source)
    if args.log_limit:
        log_emb = np.ascontiguousarray(log_emb[: args.log_limit], dtype="float32")
        print(f"  Q_L capped to {len(log_emb):,} (via --log-limit)", flush=True)

    # EP + s_max depend only on k_ep (fixed) and the doc index — build them ONCE
    # and reuse across the whole th/k'/ef/PCA sweep (the |Q_L| doc searches are
    # the expensive part), and cache them to disk so later runs skip the ~80-min
    # build entirely. I_Q is (re)built/cached per distinct PCA dim below.
    t0 = time.perf_counter()
    ep_table, s_max = load_or_build_ep(
        index, log_emb, cache_dir, model_name, dataset, index_path,
        k_ep=args.k_ep, log_limit=args.log_limit, rebuild=args.rebuild_cache)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Q_L size={len(log_emb):,} | k_ep={args.k_ep} | s_max={s_max:.4f} | "
          f"EP ready in {build_ms:.1f} ms", flush=True)

    dim = log_emb.shape[1]
    pca_list = (_parse_int_list(args.pca_list) if args.pca_list else
                ([0, dim // 4] if args.sweep else [args.pca_dim]))
    kprime_list = (_parse_int_list(args.kprime_list) if args.kprime_list else
                   ([10, 20] if args.sweep else [args.k_prime]))
    ef_list = (_parse_int_list(args.ef_list) if args.ef_list else
               (list(range(10, 201, 10)) if args.sweep else [args.ef_default]))

    rows = []
    # carried out of the loop for the single-config detailed report below
    iq = pca = sims = visited_counts = metrics = None
    s_max_eff = s_max            # dragon overrides this per PCA dim (see below)
    th_used = kprime_used = ef_used = None
    for pca_dim in pca_list:
        if pca_dim and pca_dim > 0:
            print(f"\n[I_Q] PCA {dim} -> {pca_dim}", flush=True)
        iq, pca = load_or_build_iq(
            log_emb, cache_dir, model_name, dataset, args.log_limit, pca_dim,
            rebuild=args.rebuild_cache)

        # Resolve th AND the adaptive-ef anchor s_max. dragon is asymmetric (query
        # vs context encoder): the routing similarity s (query->log-query) lives on a
        # HIGHER scale than the paper's query->doc s_max, which then sits BELOW the
        # entire s-distribution -> the degenerate guard (s_max<=th) in
        # adaptive_ef_search pins ef' to ef_min and the adaptive engine NEVER engages
        # (all routed queries get the min beam regardless of confidence). Fix: for
        # dragon derive BOTH th and s_max from the routing-similarity distribution
        # (the same scale as s), so th < s_max and s spans [th, s_max]. snowflake is
        # symmetric -> keep the paper's fixed th and query->doc s_max (already work).
        if model_name == "dragon":
            s_vals = routing_similarities(query_matrix, iq, args.k_prime, pca, log_emb)
            # p75 = the paper's s_max percentile, now on the routing-sim scale. The th
            # sweep uses percentiles <= 70, so th < s_max holds by construction.
            s_max_eff = float(np.percentile(s_vals, 75))
            print(f"  routing similarities s: min={s_vals.min():.4f} "
                  f"p25={np.percentile(s_vals, 25):.4f} median={np.median(s_vals):.4f} "
                  f"max={s_vals.max():.4f} | adaptive s_max<-p75={s_max_eff:.4f} "
                  f"(paper query->doc s_max was {s_max:.4f})", flush=True)
        else:
            s_vals, s_max_eff = None, s_max

        if args.sweep:
            th_list = (_parse_float_list(args.th_list) if args.th_list else
                       ([round(float(np.percentile(s_vals, p)), 4) for p in (10, 25, 40, 55, 70)]
                        if model_name == "dragon" else [0.3, 0.4, 0.5, 0.6, 0.7]))
        else:
            th_resolved = (args.th if args.th is not None else
                           (float(np.percentile(s_vals, 25)) if model_name == "dragon" else 0.5))
            th_list = [th_resolved]

        for th in th_list:
            for k_prime in kprime_list:
                for ef in ef_list:
                    # Primary = paper-faithful batched latency (VisitedTable amortized,
                    # threads=1). Its run's top-k is identical to the single-query path
                    # (test-validated), so metrics come from it. --compare-backends
                    # ADDS the single-query python/faiss passes (visited counts, the
                    # ~18 ms artifact column, a topk cross-check) — use on a small grid.
                    run, route_flags, sims, timing, routed_n = measure_qlr_latency_faithful(
                        index, query_matrix, eval_keys, id_map, iq, ep_table, s_max_eff,
                        k=k, k_prime=k_prime, th=th, ef_default=ef, ef_min=args.ef_min,
                        pca=pca, log_emb_full=log_emb,
                    )
                    metrics = compute_metrics(run, filtered_qrels, gt, eval_keys, k)
                    row = make_row("qlr", th, k_prime, ef, pca_dim, metrics,
                                   route_rate=routed_n / N, avg_visited=float("nan"),
                                   routed_ms=0.0, fallback_ms=0.0, routed_n=0, fallback_n=0,
                                   timing=timing)
                    extra = ""
                    if args.compare_backends:
                        (run_py, py_flags, _s, vis_py,
                         r_ms_py, f_ms_py) = build_run_qlr(
                            index, graph, query_matrix, eval_keys, id_map, iq, ep_table, s_max_eff,
                            k=k, k_prime=k_prime, th=th, ef_default=ef, ef_min=args.ef_min,
                            pca=pca, log_emb_full=log_emb, level0_backend="python",
                        )
                        (run_fs, _ff, _fs, _fv, r_ms_fs, _ffb) = build_run_qlr(
                            index, graph, query_matrix, eval_keys, id_map, iq, ep_table, s_max_eff,
                            k=k, k_prime=k_prime, th=th, ef_default=ef, ef_min=args.ef_min,
                            pca=pca, log_emb_full=log_emb, level0_backend="faiss",
                        )
                        py_routed_n = int(sum(py_flags))
                        py_fb_n = N - py_routed_n
                        routed_keys = [eval_keys[i] for i, r in enumerate(route_flags) if r]
                        row["avg_visited"] = round(float(np.mean(vis_py)), 2) if vis_py else float("nan")
                        row["routed_ms_per_q"] = round(r_ms_py / py_routed_n, 4) if py_routed_n else float("nan")
                        row["fallback_ms_per_q"] = round(f_ms_py / py_fb_n, 4) if py_fb_n else float("nan")
                        row["routed_ms_faiss_per_q"] = round(r_ms_fs / py_routed_n, 4) if py_routed_n else float("nan")
                        row["topk_match"] = round(topk_match_rate(run, run_fs, routed_keys, k), 4)
                        extra = (f" | seeded={timing['qlr_seeded_ms_per_q']:.3f} "
                                 f"faiss1q={row['routed_ms_faiss_per_q']:.3f} "
                                 f"visited={row['avg_visited']:.0f} match={row['topk_match']:.3f}")
                    rows.append(row)
                    th_used, kprime_used, ef_used = th, k_prime, ef
                    print(f"  pca={pca_dim:>4} th={th:<8} k'={k_prime} ef={ef:<4} | "
                          f"route={routed_n / N:5.1%} NDCG@{k}={rows[-1]['NDCG@10']:.4f} "
                          f"MRR@{k}={rows[-1]['MRR@10']:.4f} Acc@{k}={rows[-1]['Accuracy@10']:.4f} "
                          f"lat={timing['latency_ms_per_q']:.3f}ms{extra}", flush=True)

    if args.sweep:
        write_sweep_results(baseline_rows + rows, out_path)
        print("\nNOTE: latency_ms_per_q = PAPER-FAITHFUL per-query latency (batched, "
              "VisitedTable amortized, threads=1) — compare baseline vs qlr at MATCHED "
              "Accuracy@10 (read off the speedup). qlr_route/seeded/fallback_ms_per_q "
              "decompose it. NDCG/MRR/Accuracy are real. With --compare-backends the "
              "single-query columns are added: routed_ms_faiss_per_q (the ~18 ms "
              "artifact), routed_ms_per_q (python beam), avg_visited, topk_match "
              "(faithful-vs-single cross-check, ~1.0).")
        return

    # ================= single-config detailed latency (warmup + timed sweeps) =================
    th, k_prime, ef_default = th_used, kprime_used, ef_used

    def timed_sweep():
        """One full pass over all eval queries; split routed vs fallback time."""
        routed_ms, fallback_ms = 0.0, 0.0
        routed_cnt, fallback_cnt = 0, 0
        for row in range(N):
            t0 = time.perf_counter()
            _, _, _, routed, _ = route(
                query_matrix[row:row + 1], iq, ep_table, index, graph, s_max_eff,
                k=k, k_prime=k_prime, th=th, ef_default=ef_default, ef_min=args.ef_min,
                pca=pca, log_emb_full=log_emb, level0_backend=args.level0_backend,
            )
            dt = (time.perf_counter() - t0) * 1000
            if routed:
                routed_ms += dt
                routed_cnt += 1
            else:
                fallback_ms += dt
                fallback_cnt += 1
        return routed_ms, fallback_ms, routed_cnt, fallback_cnt

    print(f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed per-query sweeps...",
          flush=True)
    for _ in range(BATCH_WARMUP_RUNS):
        timed_sweep()
    routed_times, fallback_times = [], []
    routed_n = fallback_n = 0
    for run_i in range(BATCH_TIMED_RUNS):
        r_ms, f_ms, routed_n, fallback_n = timed_sweep()
        routed_times.append(r_ms)
        fallback_times.append(f_ms)
        r_pq = r_ms / routed_n if routed_n else float("nan")
        f_pq = f_ms / fallback_n if fallback_n else float("nan")
        print(f"  run {run_i + 1}: routed total={r_ms:7.2f} ms (per-query={r_pq:.3f}) | "
              f"fallback total={f_ms:7.2f} ms (per-query={f_pq:.3f})", flush=True)

    r_min, r_med, r_mean = latency_stats(routed_times)
    f_min, f_med, f_mean = latency_stats(fallback_times)
    overall_times = [r + f for r, f in zip(routed_times, fallback_times)]
    o_min, o_med, o_mean = latency_stats(overall_times)

    print("\n" + "=" * 70)
    print(f"QLR PURE PYTHON RESULTS ({model_name} [{dataset}])")
    print("=" * 70)
    print(f"Turns evaluated: {N}  ({routed_n} routed, {fallback_n} fallback)")
    print(f"Route rate:      {routed_n / N:.1%}  (s>=th)")
    print(f"Avg similarity:  {np.mean(sims):.4f}  (s_max={s_max_eff:.4f}, th={th:.4f})")
    print(f"NDCG@3:  {metrics['NDCG@3']:.4f}")
    print(f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
    print(f"MRR@{k}:  {metrics[f'MRR@{k}']:.4f}")
    print(f"Accuracy@{k}: {metrics[f'Accuracy@{k}']:.4f}   (vs exhaustive; nan if no ground truth)")
    print()
    print(f"Latency, summed over queries (min / median / mean over {BATCH_TIMED_RUNS} sweeps):")
    print(f"  routed   total:     {r_min:8.2f}  /  {r_med:8.2f}  /  {r_mean:8.2f}  ms")
    print(f"  routed   per query: {per_query_line((r_min, r_med, r_mean), routed_n)}")
    print(f"  fallback total:     {f_min:8.2f}  /  {f_med:8.2f}  /  {f_mean:8.2f}  ms")
    print(f"  fallback per query: {per_query_line((f_min, f_med, f_mean), fallback_n)}")
    print(f"  overall  total:     {o_min:8.2f}  /  {o_med:8.2f}  /  {o_mean:8.2f}  ms")
    print(f"  overall  per query: {per_query_line((o_min, o_med, o_mean), N)}   <- compare to paper Time")
    print()
    pca_dim = pca_list[0]
    print(f"th={th:.4f}  k'={k_prime}  k_ep={args.k_ep}  ef'∈[{args.ef_min},{ef_default}]  pca={pca_dim}")
    if visited_counts:
        print(f"Avg visited nodes (routed): {np.mean(visited_counts):.1f}")
    if args.out:
        write_sweep_results(baseline_rows + rows, out_path)
    print("=" * 70)
    print("NOTE: routed search uses the pure-Python level-0 beam search; real")
    print("latency needs the C++ kernel. NDCG/MRR/Accuracy are real numbers.")


if __name__ == "__main__":
    main()
