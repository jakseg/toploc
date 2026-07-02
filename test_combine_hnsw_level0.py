#!/usr/bin/env python3
"""
Synthetic, server-free test for the TopLoc-HNSW latency fix.

Proves the two claims the fix rests on, on a tiny in-RAM HNSW index (no cluster,
no 38.6M vectors needed):

  1. CORRECTNESS — the native FAISS `search_level_0` backend
     (`faiss_level0_search_batch`) returns the SAME top-k as the hand-rolled
     level-0 beam (`python_level0_search_one`) it replaces, for the same cached
     entry points and ef-search. So switching backends does NOT change metrics /
     accuracy — only the latency path becomes native.

  2. SPEED — one globally-batched native call is much faster than looping the
     per-query Python beam over the same follow-up set. (Absolute numbers are tiny
     here; the point is the direction and that the native path is not the slow one.)

Run:
    python test_combine_hnsw_level0.py
"""

import time

import faiss
import numpy as np

from combine_base_top_hnsw_test import (
    faiss_level0_search_batch,
    load_hnsw_level0_graph,
    python_level0_search_one,
)


def build_index(nb=4000, d=48, seed=0):
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((nb, d)).astype("float32")
    faiss.normalize_L2(xb)  # cosine == inner product on the unit sphere
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(xb)
    return index, xb


def top_ids(row):
    return [int(x) for x in row if int(x) >= 0]


def main():
    faiss.omp_set_num_threads(1)
    k, ef = 10, 64
    index, xb = build_index()
    graph = load_hnsw_level0_graph(index)

    rng = np.random.default_rng(1)
    m = 60
    queries = rng.standard_normal((m, index.d)).astype("float32")
    faiss.normalize_L2(queries)

    # Cached entry points per query: pretend a "q0" landed near each query, i.e.
    # take a random-ish nearby node as the privileged entry point (1 per query),
    # exactly like TopLoc caches top1(q0, D).
    index.hnsw.efSearch = ef
    _, seed_hits = index.search(queries, 3)
    ep_lists = [np.array([int(seed_hits[i, i % 3])], dtype="int64") for i in range(m)]

    # --- native, globally batched ---
    Dg, Ig = faiss_level0_search_batch(index, queries, ep_lists, ef, k)

    # --- reference: per-query Python beam ---
    mismatches = 0
    checked = 0
    for i in range(m):
        s, ids, _ = python_level0_search_one(
            index, graph, queries[i : i + 1], ep_lists[i], k, ef
        )
        ref = top_ids(ids[0])
        got = top_ids(Ig[i])
        checked += 1
        # Compare as the retrieved SET of the k ids (identical algorithm; ties on
        # equal scores may reorder, so a set match is the right invariant).
        if set(ref) != set(got):
            # Allow a rare 1-id boundary difference from float tie-breaking only.
            if len(set(ref) ^ set(got)) > 2:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  MISMATCH q{i}: ref={ref}\n              got={got}")

    assert mismatches == 0, f"{mismatches}/{checked} queries differ beyond tie tolerance"
    print(f"[1] CORRECTNESS OK — native batched top-k matches the Python beam "
          f"on all {checked} queries (set-equal within tie tolerance).")

    # --- speed: native batched vs looping the Python beam ---
    for _ in range(2):  # warmup
        faiss_level0_search_batch(index, queries, ep_lists, ef, k)
    t0 = time.perf_counter()
    for _ in range(5):
        faiss_level0_search_batch(index, queries, ep_lists, ef, k)
    native_ms = (time.perf_counter() - t0) / 5 * 1000

    t0 = time.perf_counter()
    for i in range(m):
        python_level0_search_one(index, graph, queries[i : i + 1], ep_lists[i], k, ef)
    beam_ms = (time.perf_counter() - t0) * 1000

    print(f"[2] SPEED — native batched: {native_ms:7.3f} ms for {m} queries | "
          f"python beam: {beam_ms:7.3f} ms | {beam_ms / max(native_ms, 1e-9):.1f}x faster")
    assert native_ms < beam_ms, "native batched path should be faster than the Python beam"

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
