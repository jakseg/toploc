#!/usr/bin/env python3
"""
Synthetic regression test for the QLR pipeline — runs locally, NO server / no
real data needed. Builds tiny in-memory FAISS indexes and temp parquet/qrels
files to exercise: the loaders, adaptive ef, the seeded level-0 beam search,
routing (fallback + routed + PCA), and the metrics (incl. Accuracy@10).

Run (needs faiss + pyarrow + ir_measures, e.g. the toploc-demo conda env):
    python test_qlr_pipeline.py
"""

import os
import sys
import json
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss

import toploc2.toploc2_hnsw_pure_python as qlr


# ---------- tiny fixtures ----------
def make_doc_index(n=200, d=16, seed=0, normalize=True):
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((n, d)).astype("float32")
    if normalize:
        faiss.normalize_L2(docs)
    index = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.add(docs)
    index.hnsw.efSearch = 64
    graph = qlr.load_hnsw_level0_graph(index)
    id_map = {str(i): f"doc{i}" for i in range(n)}
    return index, graph, docs, id_map


def brute_force_topk(docs, q, k):
    scores = docs @ np.asarray(q, dtype="float32").reshape(-1)
    return list(np.argsort(-scores)[:k])


# ---------- tests ----------
def test_adaptive_ef_search():
    ef_d, ef_m, smax, th = 100, 10, 0.8, 0.4
    assert qlr.adaptive_ef_search(0.8, smax, th, ef_d, ef_m) == ef_m       # s == s_max
    assert qlr.adaptive_ef_search(0.95, smax, th, ef_d, ef_m) == ef_m      # s > s_max (clamp)
    assert qlr.adaptive_ef_search(0.4, smax, th, ef_d, ef_m) == ef_d       # s == th
    mid = qlr.adaptive_ef_search(0.6, smax, th, ef_d, ef_m)
    assert ef_m < mid < ef_d, mid
    # monotone non-increasing in s
    vals = [qlr.adaptive_ef_search(s, smax, th, ef_d, ef_m) for s in np.linspace(th, smax, 9)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), vals
    # degenerate calibration (s_max <= th) -> ef_min
    assert qlr.adaptive_ef_search(0.5, 0.3, 0.4, ef_d, ef_m) == ef_m


def test_load_qrels(tmp):
    # CAST-style commas
    p1 = os.path.join(tmp, "cast.qrel")
    with open(p1, "w") as f:
        f.write("31_1,0,CAR_abc,1\n31_1,0,CAR_def,0\n31_2,0,CAR_xyz,2\n")
    q1 = qlr.load_qrels(p1)
    assert q1["31_1"] == {"CAR_abc": 1}, q1           # score 0 dropped
    assert q1["31_2"] == {"CAR_xyz": 2}
    # msmarco-style tabs + pid_prefix
    p2 = os.path.join(tmp, "ms.tsv")
    with open(p2, "w") as f:
        f.write("174249\t0\t1092925\t1\n174249\t0\t9\t1\n")
    q2 = qlr.load_qrels(p2, pid_prefix="MARCO_")
    assert q2["174249"] == {"MARCO_1092925": 1, "MARCO_9": 1}, q2


def test_load_parquet_embeddings(tmp):
    d, n = 8, 12
    rng = np.random.default_rng(1)
    emb = (rng.standard_normal((n, d)) * 5).astype("float32")  # non-unit norms
    ids = [f"q{i}" for i in range(n)]
    path = os.path.join(tmp, "shard.part0.parquet")
    pq.write_table(pa.table({"id": ids, "embedding": [row.tolist() for row in emb]}), path)

    rids, rnorm = qlr.load_parquet_embeddings(tmp, normalize=True)
    assert rids == ids
    assert np.allclose(np.linalg.norm(rnorm, axis=1), 1.0, atol=1e-5)  # cosine scale

    _, rraw = qlr.load_parquet_embeddings(tmp, normalize=False)
    assert np.allclose(rraw, emb, atol=1e-4)                            # raw (dragon) scale
    assert not np.allclose(np.linalg.norm(rraw, axis=1), 1.0)


def test_build_iq_and_lookup():
    index, graph, docs, _ = make_doc_index()
    rng = np.random.default_rng(2)
    log = rng.standard_normal((40, docs.shape[1])).astype("float32")
    faiss.normalize_L2(log)
    iq = qlr.build_query_log_index(log)
    assert iq.ntotal == 40 and iq.d == docs.shape[1]
    ep, s_max = qlr.build_lookup_table(index, log, k_ep=10)
    assert ep.shape == (40, 10)
    assert ep.dtype == np.int64
    assert -1.0 <= s_max <= 1.0, s_max
    assert (ep >= 0).all()


def test_level0_search_exact():
    index, graph, docs, _ = make_doc_index(n=120, d=12, seed=3)
    n = docs.shape[0]
    q = docs[5] + 0.01 * np.random.default_rng(0).standard_normal(docs.shape[1]).astype("float32")
    # Seed every node + ef_search >= n  ->  the beam search must return the EXACT top-k.
    scores, idx, visited = qlr.toploc_hnsw_level0_search(
        index, graph, q, entry_points=np.arange(n), k=10, ef_search=n)
    got = [i for i in idx[0].tolist() if i >= 0]
    assert got == brute_force_topk(docs, q, 10), (got, brute_force_topk(docs, q, 10))
    assert visited == n


def test_level0_search_seeded_runs():
    index, graph, docs, _ = make_doc_index(n=150, seed=4)
    q = docs[7]
    scores, idx, visited = qlr.toploc_hnsw_level0_search(
        index, graph, q, entry_points=[7], k=10, ef_search=64)
    got = [i for i in idx[0].tolist() if i >= 0]
    assert len(got) == 10 and len(set(got)) == 10
    assert 7 in got                       # seeded from the true answer's node
    assert 0 < visited <= 150


def test_faiss_level0_matches_python():
    """FAISS search_level_0 backend == the Python beam (same seeded level-0 search).
    Skips gracefully if the installed faiss build does not expose search_level_0."""
    probe = faiss.IndexHNSWFlat(4, 8, faiss.METRIC_INNER_PRODUCT)
    if not hasattr(probe, "search_level_0"):
        print("  (skipped: this faiss build has no search_level_0)")
        return
    index, graph, docs, _ = make_doc_index(n=120, d=12, seed=12)
    n = docs.shape[0]
    q = docs[5] + 0.01 * np.random.default_rng(0).standard_normal(docs.shape[1]).astype("float32")
    bf = brute_force_topk(docs, q, 10)
    # Seed every node + ef_search >= n -> both backends must return the EXACT top-k.
    sp, ip_, _ = qlr.toploc_hnsw_level0_search(index, graph, q, entry_points=np.arange(n), k=10, ef_search=n)
    sf, iff, vis = qlr.faiss_level0_search(index, graph, q, entry_points=np.arange(n), k=10, ef_search=n)
    assert [i for i in ip_[0].tolist() if i >= 0] == bf
    assert [i for i in iff[0].tolist() if i >= 0] == bf, (iff[0].tolist(), bf)
    assert vis is None                                  # FAISS gives no visited count
    assert np.allclose(sf[0], sp[0], atol=1e-4)         # same inner-product scores
    # empty seed set -> graceful empty result (matches the route() C-can-be-empty path)
    se, ie, _ = qlr.faiss_level0_search(index, graph, q, entry_points=np.array([], dtype=int), k=10)
    assert (ie[0] < 0).all()


def test_topk_match_rate():
    run_a = {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}
    run_b = {"q1": {"a": 3.0, "b": 2.0, "z": 0.5}}
    # top-3 {a,b,c} vs {a,b,z} -> intersection {a,b} = 2/3
    assert abs(qlr.topk_match_rate(run_a, run_b, ["q1"], 3) - (2 / 3)) < 1e-9
    assert qlr.topk_match_rate(run_a, run_a, ["q1"], 3) == 1.0       # identical
    assert np.isnan(qlr.topk_match_rate(run_a, run_b, [], 3))        # no keys


def test_route_fallback_and_routed():
    index, graph, docs, id_map = make_doc_index(n=200, d=16, seed=5)
    log = docs[:30].copy()                # log queries == some docs -> high routability
    iq = qlr.build_query_log_index(log)
    ep, s_max = qlr.build_lookup_table(index, log, k_ep=10)
    q = log[3]

    # th above 1.0 -> never routes -> fallback path
    sc, ix, visited, routed, s = qlr.route(q, iq, ep, index, graph, s_max,
                                           k=10, k_prime=10, th=1.01, ef_default=64, ef_min=10)
    assert routed is False and visited is None
    assert (ix >= 0).sum() == 10

    # low th -> routes -> seeded beam search, visited counted
    sc, ix, visited, routed, s = qlr.route(q, iq, ep, index, graph, s_max,
                                           k=10, k_prime=10, th=-1.0, ef_default=64, ef_min=10)
    assert routed is True and visited is not None and visited > 0
    assert (ix >= 0).sum() == 10
    assert s >= -1.0


def test_route_with_pca():
    index, graph, docs, id_map = make_doc_index(n=200, d=24, seed=6)
    log = docs[:60].copy()
    pca = qlr.build_pca(log, 6)
    proj = qlr.pca_apply(pca, log)
    assert proj.shape == (60, 6)
    iq = qlr.build_iq_for_pca(log, pca)
    ep, s_max = qlr.build_lookup_table(index, log, k_ep=10)
    q = log[10]
    sc, ix, visited, routed, s = qlr.route(q, iq, ep, index, graph, s_max,
                                           k=10, k_prime=10, th=-1.0, ef_default=64, ef_min=10,
                                           pca=pca, log_emb_full=log)
    assert routed is True and (ix >= 0).sum() == 10
    # s is recomputed full-dim against the matched log vector -> a real cosine
    assert -1.0001 <= s <= 1.0001, s


def test_accuracy_at_k():
    gt = {"q1": ["a", "b", "c", "d", "e"], "q2": ["m", "n", "o", "p", "q"]}
    run = {"q1": {"a": 1.0, "b": 0.9, "z": 0.5}, "q2": {"x": 1.0}}
    # q1: {a,b} found of 5 -> 0.4 ; q2: none of 5 -> 0.0  -> mean 0.2
    assert abs(qlr.accuracy_at_k(run, gt, ["q1", "q2"], 5) - 0.2) < 1e-9
    assert np.isnan(qlr.accuracy_at_k(run, None, ["q1"], 5))


def test_compute_metrics_and_groundtruth_roundtrip(tmp):
    qrels = {"q1": {"a": 1, "b": 1}}
    run = {"q1": {"a": 5.0, "x": 4.0, "b": 3.0}}
    gt = {"q1": ["a", "b", "c"]}
    m = qlr.compute_metrics(run, qrels, gt, ["q1"], 10)
    assert set(m) == {"NDCG@3", "NDCG@10", "MRR@10", "Accuracy@10"}
    assert 0.0 <= m["NDCG@10"] <= 1.0
    assert abs(m["MRR@10"] - 1.0) < 1e-9             # 'a' (relevant) ranked first
    assert abs(m["Accuracy@10"] - (2 / 10)) < 1e-9   # {a,b} of true top-3, /k=10

    # groundtruth json roundtrip via the path helper
    path = qlr.groundtruth_path(tmp, "snowflake", "msmarco-on-cast", 10)
    with open(path, "w") as f:
        json.dump(gt, f)
    assert qlr.load_groundtruth(path) == gt
    assert qlr.load_groundtruth(os.path.join(tmp, "nope.json")) is None


def test_build_run_baseline_and_qlr():
    """The two run builders main() loops over, end to end on synthetic data."""
    index, graph, docs, id_map = make_doc_index(n=200, d=16, seed=8)
    eval_keys = [f"q{i}" for i in range(10)]
    query_matrix = docs[:10].copy()                 # queries == first 10 docs
    # qrels: each query's relevant doc is itself (docN); gt: exact top-10.
    qrels = {f"q{i}": {f"doc{i}": 1} for i in range(10)}
    gt = {}
    for i in range(10):
        gt[f"q{i}"] = [f"doc{j}" for j in brute_force_topk(docs, query_matrix[i], 10)]

    # baseline
    run_b, total_ms = qlr.build_run_baseline(index, query_matrix, eval_keys, id_map, k=10, ef_search=64)
    mb = qlr.compute_metrics(run_b, qrels, gt, eval_keys, 10)
    assert total_ms >= 0.0
    assert mb["Accuracy@10"] > 0.5                  # plain HNSW should recover most true NN
    assert mb["MRR@10"] > 0.5                       # each query's own doc ranks high

    # QLR routed (log == the queries -> everything routes)
    log = query_matrix.copy()
    iq = qlr.build_query_log_index(log)
    ep, s_max = qlr.build_lookup_table(index, log, k_ep=10)
    run_q, flags, sims, visited, r_ms, f_ms = qlr.build_run_qlr(
        index, graph, query_matrix, eval_keys, id_map, iq, ep, s_max,
        k=10, k_prime=10, th=-1.0, ef_default=64, ef_min=10)
    assert all(flags)                               # th=-1 -> all routed
    assert len(visited) == 10 and r_ms >= 0.0 and f_ms == 0.0
    mq = qlr.compute_metrics(run_q, qrels, gt, eval_keys, 10)
    assert mq["Accuracy@10"] > 0.5
    assert set(mq) == {"NDCG@3", "NDCG@10", "MRR@10", "Accuracy@10"}


def test_parse_lists():
    assert qlr._parse_float_list("0.3,0.4,0.5") == [0.3, 0.4, 0.5]
    assert qlr._parse_int_list("10,20") == [10, 20]
    assert qlr._parse_int_list("") == []


def main():
    tmp = tempfile.mkdtemp(prefix="qlr_test_")
    tests = [
        ("adaptive_ef_search", lambda: test_adaptive_ef_search()),
        ("load_qrels", lambda: test_load_qrels(tmp)),
        ("load_parquet_embeddings", lambda: test_load_parquet_embeddings(tmp)),
        ("build_iq_and_lookup", lambda: test_build_iq_and_lookup()),
        ("level0_search_exact", lambda: test_level0_search_exact()),
        ("level0_search_seeded_runs", lambda: test_level0_search_seeded_runs()),
        ("faiss_level0_matches_python", lambda: test_faiss_level0_matches_python()),
        ("topk_match_rate", lambda: test_topk_match_rate()),
        ("route_fallback_and_routed", lambda: test_route_fallback_and_routed()),
        ("route_with_pca", lambda: test_route_with_pca()),
        ("accuracy_at_k", lambda: test_accuracy_at_k()),
        ("compute_metrics_and_groundtruth", lambda: test_compute_metrics_and_groundtruth_roundtrip(tmp)),
        ("build_run_baseline_and_qlr", lambda: test_build_run_baseline_and_qlr()),
        ("parse_lists", lambda: test_parse_lists()),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
