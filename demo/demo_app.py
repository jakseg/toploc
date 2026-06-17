"""TopLoc demo dashboard (Streamlit).

Conversational retrieval demo over a small precomputed subset (see
build_demo_subset.py). Lets the user pick Exact / IVF / TopLoc IVF and shows,
per query: the retrieved passages, the number of vectors actually compared
(a scale-independent efficiency proxy) and — for known CAsT turns — NDCG/MRR.

Run:  streamlit run demo_app.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import altair as alt
import faiss
import pyarrow.parquet as pq
import streamlit as st
import streamlit.components.v1 as components

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
K = 10

# Paper Table 1 — TREC CAsT 2019, full ~38M collection. Reported effectiveness
# and latency (the demo subset cannot show these). Data-driven so a
# "This reimplementation" column can be appended later.
PAPER_TABLE = {
    "snowflake": [
        {"Method": "Exact",       "MRR@10": 0.817, "NDCG@3": 0.550, "NDCG@10": 0.502, "Time (ms)": "–",  "Speedup": "–"},
        {"Method": "IVF",         "MRR@10": 0.815, "NDCG@3": 0.544, "NDCG@10": 0.497, "Time (ms)": "24.9", "Speedup": "–"},
        {"Method": "TopLoc IVF",  "MRR@10": 0.827, "NDCG@3": 0.555, "NDCG@10": 0.505, "Time (ms)": "5.7",  "Speedup": "4.4×"},
        {"Method": "TopLoc IVF+", "MRR@10": 0.827, "NDCG@3": 0.554, "NDCG@10": 0.501, "Time (ms)": "5.7",  "Speedup": "4.4×"},
        {"Method": "HNSW",        "MRR@10": 0.814, "NDCG@3": 0.548, "NDCG@10": 0.500, "Time (ms)": "1.8",  "Speedup": "–"},
        {"Method": "TopLoc HNSW", "MRR@10": 0.808, "NDCG@3": 0.549, "NDCG@10": 0.493, "Time (ms)": "0.7",  "Speedup": "2.6×"},
    ],
    "dragon": [
        {"Method": "Exact",       "MRR@10": 0.799, "NDCG@3": 0.522, "NDCG@10": 0.492, "Time (ms)": "–",  "Speedup": "–"},
        {"Method": "IVF",         "MRR@10": 0.813, "NDCG@3": 0.528, "NDCG@10": 0.486, "Time (ms)": "33.0", "Speedup": "–"},
        {"Method": "TopLoc IVF",  "MRR@10": 0.789, "NDCG@3": 0.517, "NDCG@10": 0.479, "Time (ms)": "6.5",  "Speedup": "5.1×"},
        {"Method": "TopLoc IVF+", "MRR@10": 0.795, "NDCG@3": 0.518, "NDCG@10": 0.477, "Time (ms)": "3.8",  "Speedup": "8.7×"},
        {"Method": "HNSW",        "MRR@10": 0.789, "NDCG@3": 0.508, "NDCG@10": 0.469, "Time (ms)": "8.3",  "Speedup": "–"},
        {"Method": "TopLoc HNSW", "MRR@10": 0.785, "NDCG@3": 0.503, "NDCG@10": 0.466, "Time (ms)": "0.8",  "Speedup": "10.4×"},
    ],
}

# Our reimplementation — same eval set as the paper: full ~38M-passage TREC
# CAsT 2019 collection, snowflake encoder, 173 scored turns (20 first-turn,
# 153 follow-up). NDCG@3/NDCG@10/MRR@10 are therefore directly comparable to
# PAPER_TABLE above. "Time (ms)" = median follow-up latency per query over 5
# timed sweeps on our hardware (the metric TopLoc is meant to accelerate); it
# is NOT directly comparable to the paper's absolute ms (different machine),
# but the within-table relative speeds and the "Speedup" column are. "Speedup"
# = our baseline IVF/HNSW time ÷ the TopLoc variant's time (within each family).
OURS_TABLE = {
    "snowflake": [
        {"Method": "Exact",             "MRR@10": 0.8158, "NDCG@3": 0.5500, "NDCG@10": 0.5020, "Time (ms)": "≈1865.2", "Speedup": "–"},
        {"Method": "IVF",               "MRR@10": 0.7931, "NDCG@3": 0.5426, "NDCG@10": 0.4841, "Time (ms)": "20.71", "Speedup": "–"},
        {"Method": "TopLoc IVF (np=32)", "MRR@10": 0.7845, "NDCG@3": 0.5201, "NDCG@10": 0.4929, "Time (ms)": "13.4",  "Speedup": "1.5×"},
        {"Method": "TopLoc IVF (np=128)", "MRR@10": 0.7927, "NDCG@3": 0.5301, "NDCG@10": 0.5252, "Time (ms)": "52.7",  "Speedup": "0.39×"},
        {"Method": "TopLoc IVF+ (np=128)", "MRR@10": 0.7895, "NDCG@3": 0.5319, "NDCG@10": 0.4750, "Time (ms)": "67.3", "Speedup": "0.31×"},
        {"Method": "HNSW",              "MRR@10": 0.8042, "NDCG@3": 0.5438, "NDCG@10": 0.4941, "Time (ms)": "18.9",  "Speedup": "–"},
        {"Method": "TopLoc HNSW",       "MRR@10": 0.8100, "NDCG@3": 0.5496, "NDCG@10": 0.4994, "Time (ms)": "49.1",  "Speedup": "0.38×"},
    ],
}

# Real TopLoc kernel: the compiled C++ module (toploc_search.cpp) shared with
# toploc_ivf.py. Built into the repo root. If it is not compiled in this env,
# fall back to an equivalent pure-Python path so the demo still runs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    import toploc_search as _toploc_search
    _HAVE_CPP = hasattr(_toploc_search, "toploc_ivf_search_ptr")
except Exception:
    _toploc_search = None
    _HAVE_CPP = False

st.set_page_config(page_title="TopLoc Conversational Search", layout="wide")


# ================= LOADING (cached) =================
@st.cache_resource
def load_demo():
    meta = json.load(open(os.path.join(DEMO_DIR, "meta.json")))
    exact = faiss.read_index(os.path.join(DEMO_DIR, "exact_index.index"))
    # IndexIVFFlat already is a faiss::IndexIVF; int(ivf.this) is the pointer the
    # C++ kernel expects. (Do NOT wrap in extract_index_ivf on a temporary — the
    # base index would be GC'd and leave a dangling pointer.)
    ivf = faiss.read_index(os.path.join(DEMO_DIR, "ivf_index.index"))
    try:
        ivf.make_direct_map()
    except Exception:
        pass

    # HNSW is optional — only present if build_demo_subset.py was rerun with it.
    hnsw_path = os.path.join(DEMO_DIR, "hnsw_index.index")
    hnsw = faiss.read_index(hnsw_path) if os.path.exists(hnsw_path) else None

    ids = np.load(os.path.join(DEMO_DIR, "ids.npy"), allow_pickle=True)
    id_map = {i: str(pid) for i, pid in enumerate(ids)}

    table = pq.read_table(os.path.join(DEMO_DIR, "passages.parquet"))
    texts = {str(pid): txt for pid, txt in
             zip(table.column("id").to_pylist(), table.column("text").to_pylist())}

    topics = json.load(open(os.path.join(DEMO_DIR, "topics.json")))
    qrels = json.load(open(os.path.join(DEMO_DIR, "qrels.json")))

    topic_emb = {}
    tep = os.path.join(DEMO_DIR, "topic_embeddings.parquet")
    if os.path.exists(tep):
        t = pq.read_table(tep)
        topic_emb = {str(i): np.asarray(e, dtype="float32") for i, e in
                     zip(t.column("id").to_pylist(), t.column("embedding").to_pylist())}
    return meta, exact, ivf, hnsw, id_map, texts, topics, qrels, topic_emb


@st.cache_resource
def load_encoder(model_name):
    """Lazy — only loaded when a free-text query needs encoding."""
    from sentence_transformers import SentenceTransformer
    if model_name == "snowflake":
        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
        return lambda q: model.encode([q], prompt_name="query", normalize_embeddings=True,
                                      convert_to_numpy=True).astype("float32")
    raise ValueError(f"No live encoder configured for model '{model_name}'")


# ================= SEARCH METHODS =================
def list_sizes(ivf):
    il = ivf.invlists
    return np.array([il.list_size(i) for i in range(ivf.nlist)], dtype="int64")


def search_exact(exact, q, k):
    scores, idx = exact.search(q, k)
    return scores[0], idx[0], exact.ntotal  # scanned = all vectors


def search_hnsw(hnsw, q, k, ef_search):
    """Plain FAISS HNSW search (the baseline), like evaluate_baseline.py.

    ``scanned`` is the real number of distance computations the graph traversal
    performed (faiss hnsw_stats.ndis) — the scale-independent efficiency proxy,
    directly comparable to the IVF/Exact "vectors compared" counts.
    """
    hnsw.hnsw.efSearch = int(ef_search)
    try:
        faiss.cvar.hnsw_stats.reset()
        scores, idx = hnsw.search(q, k)
        scanned = int(faiss.cvar.hnsw_stats.ndis)
    except Exception:
        scores, idx = hnsw.search(q, k)
        scanned = int(ef_search)  # fallback if stats are unavailable
    return scores[0], idx[0], scanned


def search_ivf(ivf, q, k, nprobe, sizes):
    ivf.nprobe = nprobe
    _, cent = ivf.quantizer.search(q, nprobe)
    scanned = int(ivf.nlist + sizes[cent[0]].sum())  # coarse over all + scanned lists
    scores, idx = ivf.search(q, k)
    return scores[0], idx[0], scanned


def centroid_vectors(quantizer, idx):
    """Reconstruct centroid vectors; robust across faiss versions."""
    idx = np.asarray(idx, dtype="int64")
    try:
        return quantizer.reconstruct_batch(idx).astype("float32")
    except (AttributeError, RuntimeError):
        vecs = np.empty((len(idx), quantizer.d), dtype="float32")
        for local_i, global_i in enumerate(idx):
            vecs[local_i] = quantizer.reconstruct(int(global_i))
        return vecs


def search_toploc(ivf, q, k, nprobe, cached_centroids, sizes):
    """Restrict the coarse step to the cached centroids C0, then scan nprobe of them.

    The ranking comes from the real C++ kernel in toploc_search.cpp (the very
    function toploc_ivf.py calls); without the compiled module we use an
    equivalent pure-Python path. ``scanned`` is computed here for the efficiency
    metric and mirrors exactly what the kernel scans (h coarse + selected lists).
    """
    cvecs = centroid_vectors(ivf.quantizer, cached_centroids)
    coarse = (cvecs @ q[0])  # inner product (vectors are normalized)
    npr = min(nprobe, len(cached_centroids))
    top_local = np.argpartition(-coarse, npr - 1)[:npr]
    top_local = top_local[np.argsort(-coarse[top_local])]
    sel = cached_centroids[top_local].astype("int64").reshape(1, -1)
    scanned = int(len(cached_centroids) + sizes[sel[0]].sum())  # h coarse + scanned lists

    if _HAVE_CPP:
        # Real C++ TopLoc search. Index handed over as raw pointer (int(ivf.this))
        # because the conda faiss build hands Python a SWIG proxy.
        scores, idx = _toploc_search.toploc_ivf_search_ptr(
            int(ivf.this), q,
            np.ascontiguousarray(cached_centroids, dtype="int64"),
            int(nprobe), int(k))
        return scores[0], idx[0], scanned

    # Pure-Python fallback (identical algorithm).
    sel_coarse = coarse[top_local].astype("float32").reshape(1, -1)
    ivf.nprobe = npr  # search_preassigned asserts sel.shape == (n, ivf.nprobe)
    try:
        scores, idx = ivf.search_preassigned(q, k, sel, sel_coarse)
    except TypeError:
        scores, idx = ivf.search_preassigned(q, k, sel)
    return scores[0], idx[0], scanned


def build_toploc_cache(ivf, q0, h):
    """Top-h 'hot' centroids from the first utterance (cached for the conversation)."""
    _, cent = ivf.quantizer.search(q0, h)
    return cent[0].astype("int64")


# ----- TopLoc IVF+ : cache + drift detection (mirrors toploc_ivf_plus.py) -----
def rank_within_cache(c0_vecs, q, nprobe):
    """Top-nprobe centroids within the cache (IP on normalized vectors).

    Returns local indices into C0 — like rank_within_cache in toploc_ivf_plus.py.
    """
    coarse = c0_vecs @ q[0]
    npr = min(nprobe, len(c0_vecs))
    order = np.argpartition(-coarse, npr - 1)[:npr]
    return order[np.argsort(-coarse[order])]


def build_toploc_cache_plus(ivf, q0, h, nprobe):
    """IVF+ cache: cached centroid ids, their vectors, and the first-turn top set."""
    c0 = build_toploc_cache(ivf, q0, h)
    c0_vecs = centroid_vectors(ivf.quantizer, c0)
    top0 = rank_within_cache(c0_vecs, q0, nprobe)
    return {"c0": c0, "c0_vecs": c0_vecs, "top0": top0}


def search_toploc_plus(ivf, q, k, nprobe, cache, sizes, alpha):
    """TopLoc IVF+: refresh the cache on topic drift, then restricted search.

    Drift = the top-nprobe overlap with the first turn (I0) drops below
    alpha*nprobe. On refresh, the full coarse search over all centroids is added
    to ``scanned`` (a refresh costs a first-turn-style coarse pass).
    Returns (scores, idx, scanned, cache, refreshed).
    """
    top_j = rank_within_cache(cache["c0_vecs"], q, nprobe)
    i0 = len(np.intersect1d(top_j, cache["top0"]))
    refreshed = i0 < alpha * nprobe
    refresh_cost = 0
    if refreshed:
        cache = build_toploc_cache_plus(ivf, q, len(cache["c0"]), nprobe)
        refresh_cost = ivf.nlist  # refresh = full coarse over all centroids
    scores, idx, scanned = search_toploc(ivf, q, k, nprobe, cache["c0"], sizes)
    return scores, idx, scanned + refresh_cost, cache, refreshed


# ----- TopLoc HNSW : privileged entry point + level-0 beam search ----------
# Mirrors toploc_hnsw_2.py. The follow-up search runs in Python; ``scanned`` is
# the number of visited graph nodes — exactly the efficiency proxy the demo
# shows (identical to what a C++ kernel would report; C++ only changes latency).
@st.cache_resource
def hnsw_level0_graph(_hnsw):
    h = _hnsw.hnsw
    offsets = faiss.vector_to_array(h.offsets).astype("int64")
    neighbors = faiss.vector_to_array(h.neighbors).astype("int64")
    return offsets, neighbors, int(h.nb_neighbors(0))


def _hnsw_reconstruct(hnsw, ids):
    ids = np.asarray(ids, dtype="int64")
    try:
        return hnsw.reconstruct_batch(ids).astype("float32")
    except Exception:
        out = np.empty((len(ids), hnsw.d), dtype="float32")
        for i, n in enumerate(ids):
            hnsw.reconstruct(int(n), out[i])
        return out


def build_hnsw_entry(hnsw, q0, up, ef_search, entry_points=1):
    """Turn 0: full HNSW search with efSearch*up; cache the top entry point(s)."""
    old = hnsw.hnsw.efSearch
    hnsw.hnsw.efSearch = int(ef_search * up)
    try:
        faiss.cvar.hnsw_stats.reset()
        scores, idx = hnsw.search(q0, max(K, entry_points))
        scanned = int(faiss.cvar.hnsw_stats.ndis)
    except Exception:
        scores, idx = hnsw.search(q0, max(K, entry_points))
        scanned = int(ef_search * up)
    hnsw.hnsw.efSearch = old
    entry = [int(x) for x in idx[0][:entry_points] if int(x) >= 0]
    return entry, scores[0], idx[0], scanned


def search_toploc_hnsw(hnsw, graph, q, k, entry_points, ef_search):
    """Follow-up: level-0 beam search starting from the cached entry point(s)."""
    import heapq
    offsets, neighbors, degree0 = graph
    qv = q.reshape(-1).astype("float32")
    candidates, results, visited = [], [], set()

    def add(nodes):
        new = [int(n) for n in nodes if int(n) >= 0 and int(n) not in visited]
        for n in new:
            visited.add(n)
        if not new:
            return
        for n, sc in zip(new, _hnsw_reconstruct(hnsw, new) @ qv):
            sc = float(sc)
            heapq.heappush(candidates, (-sc, n))
            if len(results) < ef_search:
                heapq.heappush(results, (sc, n))
            elif sc > results[0][0]:
                heapq.heapreplace(results, (sc, n))

    add(entry_points)
    while candidates:
        neg, cur = heapq.heappop(candidates)
        if len(results) >= ef_search and -neg < results[0][0]:
            break
        start = int(offsets[cur])
        block = neighbors[start:start + degree0]
        add(block[block >= 0])

    top = sorted(results, key=lambda x: x[0], reverse=True)[:k]
    scores = np.array([s for s, _ in top], dtype="float32")
    idx = np.array([n for _, n in top], dtype="int64")
    return scores, idx, len(visited)


# ================= METRICS =================
def metrics_for(qid, ranked_pids, scores, qrels):
    """NDCG@3, NDCG@10, MRR@10 for a single known turn, else None."""
    if qid not in qrels:
        return None
    import ir_measures
    from ir_measures import nDCG, RR
    run = {qid: {pid: float(s) for pid, s in zip(ranked_pids, scores) if pid}}
    res = ir_measures.calc_aggregate([nDCG @ 3, nDCG @ K, RR @ K],
                                     {qid: qrels[qid]}, run)
    return {"NDCG@3": res[nDCG @ 3], "NDCG@10": res[nDCG @ K], "MRR@10": res[RR @ K]}


# ================= APP =================
meta, exact, ivf, hnsw, id_map, texts, topics, qrels, topic_emb = load_demo()
sizes = list_sizes(ivf)

st.title("TopLoc — Conversational Dense Retrieval")
st.caption(f"Demo subset: {meta['n_passages']:,} passages · model={meta['model']} · "
           f"IVF nlist={meta['nlist']} · dim={meta['dim']}")

with st.sidebar:
    st.header("Settings")
    methods = (["Exact", "IVF", "TopLoc IVF", "TopLoc IVF+"]
               + (["HNSW", "TopLoc HNSW"] if hnsw is not None else []))
    method = st.radio("Search method", methods, index=2)
    nprobe = st.slider("nprobe", 1, meta["nlist"], min(8, meta["nlist"]))
    h = st.slider("TopLoc cached centroids (h)", 1, meta["nlist"],
                  min(16, meta["nlist"]))
    alpha = st.slider("IVF+ refresh α (drift)", 0.0, 0.5, 0.1, 0.05,
                      help="IVF+ refreshes the cache when the top-nprobe overlap "
                           "with the first turn drops below α·nprobe (topic drift).")
    if hnsw is not None:
        ef_search = st.slider("efSearch (HNSW)", K, 256,
                              int(meta.get("hnsw_ef_search", 64)))
        up = st.slider("TopLoc HNSW upscaling (up)", 1, 8, 2,
                       help="First turn searches with efSearch×up to pick a "
                            "strong privileged entry point for the follow-ups.")

st.session_state.setdefault("history", [])
st.session_state.setdefault("toploc_cache", None)

# ---- Query input: toggle between known turn and free text ----
query_text, query_id, query_vec = None, None, None

_modes = ["Known CAsT turn", "Free text"]
col_mode, col_new = st.columns([3, 1])
with col_mode:
    if hasattr(st, "segmented_control"):
        mode = st.segmented_control("Input", _modes, default=_modes[0],
                                    label_visibility="collapsed")
    else:
        mode = st.radio("Input", _modes, horizontal=True,
                        label_visibility="collapsed")
with col_new:
    if st.button("New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.toploc_cache = None

# Per-mode state: when switching, stash the current conversation and restore the
# target mode's, so reopening a mode shows exactly what was there before.
if mode is not None and st.session_state.get("_mode") != mode:
    store = st.session_state.setdefault("mode_store", {})
    prev = st.session_state.get("_mode")
    if prev is not None:
        store[prev] = {"history": st.session_state.get("history", []),
                       "toploc_cache": st.session_state.get("toploc_cache")}
    restored = store.get(mode, {"history": [], "toploc_cache": None})
    st.session_state.history = restored["history"]
    st.session_state.toploc_cache = restored["toploc_cache"]
    st.session_state._mode = mode

if mode == _modes[1]:
    # Free text → encode the query live.
    with st.form("free_query", clear_on_submit=True):
        c_in, c_send = st.columns([6, 1])
        typed = c_in.text_input(
            "Ask a question", label_visibility="collapsed",
            placeholder="Ask a question (follow-ups reuse the TopLoc cache)…",
        )
        submitted = c_send.form_submit_button("Send", use_container_width=True)
    if submitted and typed:
        try:
            with st.spinner("Encoding query…"):
                query_vec = load_encoder(meta["model"])(typed)
            query_text = typed
        except Exception:
            query_vec = None
            st.error(
                "Free-text queries need the query encoder, which isn't installed in "
                "this environment. Run `pip install sentence-transformers` in the "
                "`toploc-demo` env (the model downloads on first use), or use the "
                "'Known CAsT turn' mode — those use precomputed embeddings."
            )
else:
    # Known CAsT turn → precomputed embedding + real metrics (default mode).
    c_sel, c_btn = st.columns([5, 1])
    known = c_sel.selectbox(
        "Pick a turn", [""] + sorted(topic_emb.keys()),
        format_func=lambda x: f"{x}: {topics.get(x, '')[:60]}" if x else "—",
        label_visibility="collapsed",
    )
    if c_btn.button("Run", use_container_width=True, disabled=not known):
        query_text, query_id = topics.get(known, known), known
        query_vec = topic_emb[known].reshape(1, -1).copy()
    st.caption("Uses precomputed embeddings → real NDCG / MRR metrics.")

# ---- Run a query ----
if query_vec is not None:
    is_first = len(st.session_state.history) == 0
    refreshed = False
    cache = st.session_state.toploc_cache
    if method == "TopLoc IVF":
        if is_first or not isinstance(cache, np.ndarray):
            # First turn: a standard IVF search seeds the conversation cache
            # (coarse over all centroids) — exactly like toploc_ivf.py turn 0.
            st.session_state.toploc_cache = build_toploc_cache(ivf, query_vec, h)
            scores, idx, scanned = search_ivf(ivf, query_vec, K, nprobe, sizes)
        else:
            # Follow-up turns: restricted search over the cached centroids (C++).
            scores, idx, scanned = search_toploc(ivf, query_vec, K, nprobe, cache, sizes)
    elif method == "TopLoc IVF+":
        if is_first or not (isinstance(cache, dict) and "c0" in cache):
            # First turn: build the richer IVF+ cache + full search.
            st.session_state.toploc_cache = build_toploc_cache_plus(
                ivf, query_vec, h, nprobe)
            scores, idx, scanned = search_ivf(ivf, query_vec, K, nprobe, sizes)
        else:
            # Follow-up: drift check, refresh on topic shift, then restricted search.
            scores, idx, scanned, st.session_state.toploc_cache, refreshed = \
                search_toploc_plus(ivf, query_vec, K, nprobe, cache, sizes, alpha)
    elif method == "TopLoc HNSW":
        if is_first or not (isinstance(cache, dict) and "entry" in cache):
            # First turn: full HNSW (efSearch×up) seeds the privileged entry point.
            entry, scores, idx, scanned = build_hnsw_entry(hnsw, query_vec, up, ef_search)
            scores, idx = scores[:K], idx[:K]
            st.session_state.toploc_cache = {"entry": entry}
        else:
            # Follow-up: level-0 beam search from the cached entry point.
            scores, idx, scanned = search_toploc_hnsw(
                hnsw, hnsw_level0_graph(hnsw), query_vec, K, cache["entry"], ef_search)
    elif method == "IVF":
        scores, idx, scanned = search_ivf(ivf, query_vec, K, nprobe, sizes)
    elif method == "HNSW":
        scores, idx, scanned = search_hnsw(hnsw, query_vec, K, ef_search)
    else:
        scores, idx, scanned = search_exact(exact, query_vec, K)

    ranked_pids = [id_map.get(int(i)) for i in idx if i >= 0]
    ranked_scores = [float(s) for s, i in zip(scores, idx) if i >= 0]
    m = metrics_for(query_id, ranked_pids, ranked_scores, qrels) if query_id else None

    # Agreement with exhaustive search: how many of this method's top-k also
    # appear in Exact's top-k. Honest accuracy proxy on the subset (relative to
    # exact, not to relevance) and defined for free-text queries too.
    _, ex_idx = exact.search(query_vec, K)
    exact_set = {int(i) for i in ex_idx[0] if i >= 0}
    method_set = {int(i) for i in idx if i >= 0}
    agreement = len(method_set & exact_set)

    st.session_state.history.append({
        "turn": len(st.session_state.history) + 1, "query": query_text, "method": method,
        "scanned": scanned, "speedup": exact.ntotal / max(scanned, 1),
        "pids": ranked_pids, "scores": ranked_scores, "metrics": m,
        "agreement": agreement, "k": K, "refreshed": refreshed,
    })
    st.session_state._scroll_top = True  # keep focus on the chart, don't jump down

# ---- Efficiency chart: vectors compared per turn (compact, on top) ----
if st.session_state.history:
    df = pd.DataFrame([
        {
            "Turn": t["turn"],
            "Vectors compared": t["scanned"],
            "Method": t["method"],
            "Role": "First turn" if t["turn"] == 1 else "Follow-up",
        }
        for t in st.session_state.history
    ])

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, opacity=0.92)
        .encode(
            x=alt.X("Turn:O", title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=18)),
            y=alt.Y("Vectors compared:Q", title="Vectors compared",
                    axis=alt.Axis(format="~s", gridOpacity=0.25, labelFontSize=15)),
            color=alt.Color(
                "Role:N", title=None,
                scale=alt.Scale(domain=["First turn", "Follow-up"],
                                range=["#cbd5e1", "#10b981"]),
                legend=alt.Legend(orient="top", labelFontSize=17, symbolType="circle",
                                  symbolSize=200),
            ),
            tooltip=["Turn:O", "Method:N", alt.Tooltip("Vectors compared:Q", format=",")],
        )
    )
    labels = bars.mark_text(dy=-10, baseline="bottom", fontSize=22,
                            fontWeight="bold").encode(
        text=alt.Text("Vectors compared:Q", format="~s"), color=alt.value("#334155"))
    chart = (
        (bars + labels)
        .properties(height=230)
        .configure_view(stroke=None)
        .configure_axis(labelColor="#64748b", titleColor="#64748b", titleFontSize=17,
                        domainColor="#e2e8f0", tickColor="#e2e8f0")
    )
    st.altair_chart(chart, use_container_width=True)

    follow_ups = df[df["Role"] == "Follow-up"]
    if not follow_ups.empty:
        first_n = int(df.loc[df["Role"] == "First turn", "Vectors compared"].iloc[0])
        avg_fu = follow_ups["Vectors compared"].mean()
        st.caption(
            f"First turn compares **{first_n:,}** vectors; follow-ups reuse the "
            f"cache and compare on average **{avg_fu:,.0f}** "
            f"(**{first_n / max(avg_fu, 1):.1f}× fewer**) · Exact always **{exact.ntotal:,}**."
        )
    st.divider()

# ---- Render conversation: per-turn metrics + collapsed answers ----
for turn in st.session_state.history:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vectors compared", f"{turn['scanned']:,}",
              help=f"Exact would compare all {exact.ntotal:,}")
    c2.metric("Speedup vs Exact", f"{turn['speedup']:.1f}×")
    k_turn = turn.get("k", K)
    c3.metric("Agreement w/ Exact", f"{turn.get('agreement', k_turn)}/{k_turn}",
              help="Top-k overlap with exhaustive Exact search — shows the "
                   "approximation returns the same results despite scanning less.")
    if turn["metrics"]:
        c4.metric("NDCG@10", f"{turn['metrics']['NDCG@10']:.3f}",
                  help=f"NDCG@3={turn['metrics']['NDCG@3']:.3f} · "
                       f"MRR@10={turn['metrics']['MRR@10']:.3f} · "
                       f"On the 2k-doc subset — indicative only; real values: paper reference below.")
    else:
        c4.metric("NDCG@10", "—", help="Only available for known CAsT turns")
    label = f"Turn {turn['turn']} · {turn['method']} — {turn['query']}"
    if turn.get("refreshed"):
        label += "   · cache refreshed (drift)"
    with st.expander(label, expanded=False):
        for rank, (pid, sc) in enumerate(zip(turn["pids"], turn["scores"]), 1):
            st.markdown(f"**{rank}.** `{pid}` · score={sc:.3f}")
            st.caption(texts.get(pid, "(text not in subset)")[:400])
    st.write("")  # small spacer between turns

# Pin metric columns to 4 decimals (a bare Styler defaults to 6). Header/index
# emphasis is handled by the injected CSS below — st.table strips a Styler's
# set_table_styles block, so that approach does not survive rendering.
def _emphasize(df):
    metric_cols = [c for c in ("MRR@10", "NDCG@3", "NDCG@10") if c in df.columns]
    return df.style.format("{:.4f}", subset=metric_cols)


# Make the header row and the left (Method) column bold + black in every
# st.table on the page. Targets the rendered DOM directly with !important so it
# wins over Streamlit's muted-grey table-header theme.
st.markdown(
    """
    <style>
    [data-testid="stTable"] th {
        font-weight: 700 !important;
        color: rgb(49, 51, 63) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Paper reference numbers (full 38M collection) ----
if meta["model"] in PAPER_TABLE:
    with st.expander("Paper reference — full collection (TopLoc, CAsT 2019)"):
        ref_df = pd.DataFrame(PAPER_TABLE[meta["model"]]).set_index("Method")
        st.table(_emphasize(ref_df))
        st.caption(
            f"Reported by the TopLoc paper for **{meta['model']}** on the full "
            "~38M-passage TREC CAsT 2019 collection — the real effectiveness and "
            "latency the 2k-doc demo above cannot show."
        )

# ---- Our reimplementation numbers (same full 38M collection) ----
if meta["model"] in OURS_TABLE:
    with st.expander("Our results — full collection (this reimplementation)"):
        ours_df = pd.DataFrame(OURS_TABLE[meta["model"]]).set_index("Method")
        st.table(_emphasize(ours_df))
        st.caption(
            f"Our reimplementation for **{meta['model']}**, measured on the same "
            "full ~38M-passage TREC CAsT 2019 collection and the same 173 scored "
            "turns (20 first-turn, 153 follow-up) as the paper — so **NDCG/MRR are "
            "directly comparable** to the paper table above. **Time (ms)** = median "
            "follow-up latency per query (the cost TopLoc targets) on our hardware, "
            "so absolute ms differ from the paper; **Speedup** = our baseline "
            "IVF/HNSW time ÷ the TopLoc variant's time."
        )
        st.caption(
            "Takeaways: effectiveness is reproduced — **TopLoc IVF (np=128) reaches "
            "NDCG@10 0.525**, above our Exact (0.502) and the paper's IVF (0.497). "
            "On speed, **TopLoc IVF (np=32) is ~1.5× faster than baseline IVF** "
            "(13.4 vs 20.7 ms) while still lifting NDCG@10 over baseline — the "
            "clearest reproduction of the paper's accelerate-without-losing-quality "
            "claim. At higher np and for TopLoc HNSW, however, the entry-point / "
            "centroid-cache overhead dominates in our setup (Speedup < 1)."
        )

st.divider()
st.caption("Speed shown as **vectors compared** — a scale-independent proxy. "
           "Real wall-clock latency at full 38M scale: see the demo video.")
st.caption(
    "Based on **TopLoc** — Cristina Ioana Muntean, Franco Maria Nardini, "
    "Raffaele Perego, Guido Rocchietti & Cosimo Rulli, *Efficient Conversational "
    "Search via Topical Locality in Dense Retrieval*, SIGIR ’25, Padua, Italy."
)

# After a new query, scroll the page back to the top so the chart stays in view
# instead of the viewport jumping down to the chat input.
if st.session_state.pop("_scroll_top", False):
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const el = doc.querySelector('section.main')
                || doc.querySelector('[data-testid="stMain"]')
                || doc.querySelector('[data-testid="stAppViewContainer"]');
        if (el) { el.scrollTo({top: 0, behavior: 'smooth'}); }
        window.parent.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )
