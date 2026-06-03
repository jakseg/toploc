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
import faiss
import pyarrow.parquet as pq
import streamlit as st

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
K = 10

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
    return meta, exact, ivf, id_map, texts, topics, qrels, topic_emb


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
meta, exact, ivf, id_map, texts, topics, qrels, topic_emb = load_demo()
sizes = list_sizes(ivf)

st.title("🔎 TopLoc — Conversational Dense Retrieval")
st.caption(f"Demo subset: {meta['n_passages']:,} passages · model={meta['model']} · "
           f"IVF nlist={meta['nlist']} · dim={meta['dim']}")

with st.sidebar:
    st.header("Settings")
    method = st.radio("Search method", ["Exact", "IVF", "TopLoc IVF"], index=2)
    nprobe = st.slider("nprobe", 1, meta["nlist"], min(8, meta["nlist"]))
    h = st.slider("TopLoc cached centroids (h)", 1, meta["nlist"],
                  min(16, meta["nlist"]))
    st.divider()
    if st.button("🆕 New conversation"):
        st.session_state.history = []
        st.session_state.toploc_cache = None
    st.markdown("**Known CAsT turn** (uses precomputed embedding + metrics):")
    known = st.selectbox("Pick a turn", [""] + sorted(topic_emb.keys()),
                         format_func=lambda x: f"{x}: {topics.get(x, '')[:50]}" if x else "—")

st.session_state.setdefault("history", [])
st.session_state.setdefault("toploc_cache", None)

# ---- Query input: known turn (button) or free text (chat) ----
query_text, query_id, query_vec = None, None, None
if known:
    if st.sidebar.button("Run selected turn"):
        query_text, query_id = topics.get(known, known), known
        query_vec = topic_emb[known].reshape(1, -1).copy()
typed = st.chat_input("Ask a question (follow-ups reuse the TopLoc cache)…")
if typed:
    query_text = typed
    query_vec = load_encoder(meta["model"])(typed)

# ---- Run a query ----
if query_vec is not None:
    is_first = len(st.session_state.history) == 0
    if method == "TopLoc IVF":
        if is_first or st.session_state.toploc_cache is None:
            # First turn: a standard IVF search seeds the conversation cache
            # (coarse over all centroids) — exactly like toploc_ivf.py turn 0.
            st.session_state.toploc_cache = build_toploc_cache(ivf, query_vec, h)
            scores, idx, scanned = search_ivf(ivf, query_vec, K, nprobe, sizes)
        else:
            # Follow-up turns: restricted search over the cached centroids (C++).
            scores, idx, scanned = search_toploc(ivf, query_vec, K, nprobe,
                                                 st.session_state.toploc_cache, sizes)
    elif method == "IVF":
        scores, idx, scanned = search_ivf(ivf, query_vec, K, nprobe, sizes)
    else:
        scores, idx, scanned = search_exact(exact, query_vec, K)

    ranked_pids = [id_map.get(int(i)) for i in idx if i >= 0]
    ranked_scores = [float(s) for s, i in zip(scores, idx) if i >= 0]
    m = metrics_for(query_id, ranked_pids, ranked_scores, qrels) if query_id else None

    st.session_state.history.append({
        "turn": len(st.session_state.history) + 1, "query": query_text, "method": method,
        "scanned": scanned, "speedup": exact.ntotal / max(scanned, 1),
        "pids": ranked_pids, "scores": ranked_scores, "metrics": m,
    })

# ---- Render conversation ----
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(f"**Turn {turn['turn']}** · *{turn['method']}* — {turn['query']}")
    with st.chat_message("assistant"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Vectors compared", f"{turn['scanned']:,}",
                  help=f"Exact would compare all {exact.ntotal:,}")
        c2.metric("Speedup vs Exact", f"{turn['speedup']:.1f}×")
        if turn["metrics"]:
            c3.metric("NDCG@10", f"{turn['metrics']['NDCG@10']:.3f}",
                      help=f"NDCG@3={turn['metrics']['NDCG@3']:.3f} · "
                           f"MRR@10={turn['metrics']['MRR@10']:.3f}")
        else:
            c3.metric("NDCG@10", "—", help="Only available for known CAsT turns")
        for rank, (pid, sc) in enumerate(zip(turn["pids"], turn["scores"]), 1):
            st.markdown(f"**{rank}.** `{pid}` · score={sc:.3f}")
            st.caption(texts.get(pid, "(text not in subset)")[:400])

st.divider()
st.caption("Speed shown as **vectors compared** — a scale-independent proxy. "
           "Real wall-clock latency at full 38M scale: see the demo video.")
