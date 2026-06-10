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

st.title("TopLoc — Conversational Dense Retrieval")
st.caption(f"Demo subset: {meta['n_passages']:,} passages · model={meta['model']} · "
           f"IVF nlist={meta['nlist']} · dim={meta['dim']}")

with st.sidebar:
    st.header("Settings")
    method = st.radio("Search method", ["Exact", "IVF", "TopLoc IVF"], index=2)
    nprobe = st.slider("nprobe", 1, meta["nlist"], min(8, meta["nlist"]))
    h = st.slider("TopLoc cached centroids (h)", 1, meta["nlist"],
                  min(16, meta["nlist"]))

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
    with st.expander(
        f"Turn {turn['turn']} · {turn['method']} — {turn['query']}",
        expanded=False,
    ):
        for rank, (pid, sc) in enumerate(zip(turn["pids"], turn["scores"]), 1):
            st.markdown(f"**{rank}.** `{pid}` · score={sc:.3f}")
            st.caption(texts.get(pid, "(text not in subset)")[:400])
    st.write("")  # small spacer between turns

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
