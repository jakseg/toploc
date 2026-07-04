#!/usr/bin/env python3
"""Decide WHY dragon msmarco-on-cast retrieval returns disjoint results from the
exact groundtruth (HNSW -> CAR_ heavy, exact -> MARCO_, 0/15 overlap, sane
~35 dot-product scores). --stage search already proved: index is full (38.6M),
scores are real un-normalised dot products (not cosine), so it is NOT a partial
index and NOT a normalisation mismatch. Only three causes remain:

  (A) hnsw_ids.npy is out of order vs the vectors stored in the HNSW index
      -> right neighbours found, WRONG pid labels (CAR-heavy because 77% of the
         corpus is CAR_). The 'same-count reordering' the coverage stage warned
         it could not catch.
  (B) the HNSW index was built from DIFFERENT doc vectors than GT/parquet
      -> HNSW searches a different space; its top vectors are not the true NN.
  (C) the HNSW graph is broken -> misses the true (reachable) neighbours.

Method: run BOTH the HNSW index and the exact IndexFlatIP for a handful of dev
queries, compare top-1 SCORES and check the labels by RECONSTRUCTING vectors.

  - hnsw_top_score ~= exact_top_score, labels differ  -> the HNSW index contains
    the same high-scoring vectors, only the pid mapping disagrees. Confirm by
    reconstructing the HNSW top node and the exact top node: if the VECTORS match
    but hnsw_ids[n] != exact_ids[m] -> (A) id_map scramble.
  - hnsw_top_score << exact_top_score -> HNSW misses the true neighbour. Then
    reconstruct the vector the EXACT index calls the top: if that exact-space
    vector is absent from / unreachable in the HNSW index -> (B) wrong vectors or
    (C) broken graph.

RAM: full-loads the HNSW index (~129 GB) and MMAPs the exact IndexFlatIP (flat,
mmap is fine). Check `free -h` first; ~140-200 GB free is comfortable on pegasus.

Run from inside toploc2/:
    python diagnose_dragon_idmap.py            # 10 sample queries
    python diagnose_dragon_idmap.py --n 25
"""
import argparse
import os

import numpy as np

import toploc2_hnsw_pure_python as qlr

MODEL = "dragon"
DATASET = "msmarco-on-cast"
K = 10

cache_dir = qlr.CACHE_DIRS[MODEL]
hnsw_index_path = os.path.join(cache_dir, "hnsw_index.index")
hnsw_ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
exact_index_path = os.path.join(cache_dir, "exact_index.index")
exact_ids_path = os.path.join(cache_dir, "exact_ids.npy")


def cheap_id_file_check():
    """No index load: are hnsw_ids.npy and exact_ids.npy the same array?
    Equal -> both built in the same order (does not prove correctness, but the
    two id systems at least agree). Differ -> either a benign reorder or the bug;
    the reconstruct test below settles which."""
    print("=== 0. cheap id-file comparison (no index load) ===", flush=True)
    if not os.path.exists(exact_ids_path):
        print(f"  exact_ids.npy missing ({exact_ids_path}) -> skipping; the "
              f"reconstruct test still works via exact_index if present.", flush=True)
        return
    h = np.load(hnsw_ids_path, allow_pickle=True).astype(str)
    e = np.load(exact_ids_path, allow_pickle=True).astype(str)
    print(f"len(hnsw_ids)={len(h):,}  len(exact_ids)={len(e):,}", flush=True)
    if len(h) != len(e):
        print("  DIFFERENT LENGTHS -> the two id arrays cover different sets; "
              "this alone can desync the run's pid mapping.", flush=True)
        return
    if np.array_equal(h, e):
        print("  IDENTICAL element-wise -> hnsw_ids and exact_ids are the same "
              "order. If retrieval still disagrees, the vectors/graph differ "
              "(B/C), not the labels. (Does NOT prove hnsw_ids matches the HNSW "
              "vector order — reconstruct test below is decisive.)", flush=True)
    else:
        diff = np.where(h != e)[0]
        print(f"  DIFFER at {len(diff):,}/{len(h):,} positions (first: "
              f"{diff[:5].tolist()}).", flush=True)
        for i in diff[:5]:
            print(f"    node {i}: hnsw_ids={h[i]!r}  exact_ids={e[i]!r}", flush=True)
        print("  -> the two id files are NOT aligned. If exact_ids is the trusted "
              "one (GT hits qrels), a scrambled hnsw_ids is the prime suspect for "
              "(A). The reconstruct test confirms which array matches the vectors.",
              flush=True)


def stage_search():
    import faiss

    print("\n=== 1. load indexes ===", flush=True)
    print(f"Full-loading HNSW {hnsw_index_path} "
          f"({os.path.getsize(hnsw_index_path)/1e9:.1f} GB)...", flush=True)
    hnsw = faiss.read_index(hnsw_index_path)
    hnsw.hnsw.efSearch = 200
    h_ids = np.load(hnsw_ids_path, allow_pickle=True).astype(str)
    print(f"  HNSW ntotal={hnsw.ntotal:,} dim={hnsw.d}", flush=True)

    have_exact = os.path.exists(exact_index_path) and os.path.exists(exact_ids_path)
    if have_exact:
        print(f"MMAP exact {exact_index_path} "
              f"({os.path.getsize(exact_index_path)/1e9:.1f} GB)...", flush=True)
        exact = faiss.read_index(exact_index_path, faiss.IO_FLAG_MMAP)
        e_ids = np.load(exact_ids_path, allow_pickle=True).astype(str)
        print(f"  exact ntotal={exact.ntotal:,} dim={exact.d}", flush=True)
    else:
        exact = e_ids = None
        print("  exact_index.index NOT present -> falling back to the "
              "groundtruth JSON for the true pids (scores unavailable).", flush=True)

    # sample dev queries that have qrels (same selection as the run)
    qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
    indexed = set(h_ids.tolist())
    fq = {q: {p: s for p, s in d.items() if p in indexed} for q, d in qrels.items()}
    fq = {q: d for q, d in fq.items() if d}

    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=False)  # dragon
    row_of = {q: i for i, q in enumerate(q_ids)}
    eval_keys = [q for q in q_ids if q in fq]

    gt = qlr.load_groundtruth(qlr.groundtruth_path(cache_dir, MODEL, DATASET, K))

    args_n = ARGS.n
    sample = eval_keys[:args_n]
    qv = np.ascontiguousarray(q_emb[[row_of[k] for k in sample]], dtype="float32")

    hD, hI = hnsw.search(qv, K)
    if have_exact:
        eD, eI = exact.search(qv, K)

    print(f"\n=== 2. per-query: HNSW top-1 vs exact top-1 ({len(sample)} queries) ===",
          flush=True)
    score_gap = []
    vec_match_when_score_close = 0
    n_score_close = 0
    for r, qid in enumerate(sample):
        q = qv[r]
        h_node0 = int(hI[r, 0])
        h_s0 = float(hD[r, 0])
        h_lab0 = h_ids[h_node0]
        hnsw_labels = [h_ids[i] for i in hI[r] if i != -1]

        if have_exact:
            e_node0 = int(eI[r, 0])
            e_s0 = float(eD[r, 0])
            e_lab0 = e_ids[e_node0]
            exact_labels = [e_ids[i] for i in eI[r] if i != -1]
        else:
            e_s0 = float("nan")
            e_lab0 = (gt.get(qid) or ["?"])[0] if gt else "?"
            exact_labels = (gt.get(qid) or [])[:K] if gt else []

        gap = (h_s0 - e_s0) if have_exact else float("nan")
        score_close = have_exact and abs(gap) < 0.05 * max(abs(e_s0), 1e-6)
        overlap = len(set(hnsw_labels[:K]) & set(exact_labels[:K]))

        note = ""
        if have_exact and score_close:
            n_score_close += 1
            # scores match but labels disjoint -> same vectors, different labels?
            # reconstruct the two top vectors and compare directly.
            hv = hnsw.reconstruct(h_node0)
            ev = exact.reconstruct(e_node0)
            same_vec = bool(np.allclose(hv, ev, atol=1e-4))
            # also: what pid does the EXACT index give to the HNSW top vector?
            _, back = exact.search(hv[None].astype("float32"), 1)
            exact_label_of_hnsw_vec = e_ids[int(back[0, 0])]
            if not same_vec and exact_label_of_hnsw_vec != h_lab0:
                vec_match_when_score_close += 1
            note = (f" | same_top_vec={same_vec} "
                    f"exact_calls_hnsw_top='{exact_label_of_hnsw_vec}' "
                    f"vs hnsw_label='{h_lab0}'")

        if have_exact:
            score_gap.append(gap)
        print(f"  {qid:>10}  hnsw_s={h_s0:7.3f}  exact_s={e_s0:7.3f}  gap={gap:7.3f}"
              f"  top10_overlap={overlap}  hnsw_top='{h_lab0}'  exact_top='{e_lab0}'"
              f"{note}", flush=True)

    print("\n=== 3. verdict ===", flush=True)
    if have_exact:
        gaps = np.array(score_gap)
        print(f"mean |hnsw_s - exact_s| = {np.abs(gaps).mean():.3f}  "
              f"(exact_s scale ~35)", flush=True)
        if n_score_close and (n_score_close >= 0.6 * len(sample)):
            print("HNSW top scores ~= exact top scores -> the HNSW index CONTAINS "
                  "the true high-scoring neighbours; the search is fine. The pids "
                  "just come out wrong ->", flush=True)
            print("  (A) hnsw_ids.npy is MISALIGNED with the stored vectors. "
                  "The 'exact_calls_hnsw_top' column shows the pid the exact index "
                  "assigns to the very vector HNSW returned — if that != the HNSW "
                  "label, hnsw_ids is scrambled. FIX: rebuild/repair hnsw_ids so "
                  "node i maps to the pid of the i-th added vector.", flush=True)
        else:
            print("HNSW top scores are well BELOW exact -> HNSW does NOT return the "
                  "true nearest vectors ->", flush=True)
            print("  (B) index built from different doc vectors, or (C) broken "
                  "graph. Check `same_top_vec` and whether the exact-top vector is "
                  "reachable in HNSW; the 140 ms/q latency also points at (C).",
                  flush=True)
    else:
        print("No exact index -> compared HNSW labels to the GT JSON only. If "
              "top10_overlap is 0 across the board with sane scores, build/point "
              "to exact_index.index and re-run for the decisive score comparison.",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="sample query count")
    ap.add_argument("--skip-cheap", action="store_true")
    ARGS = ap.parse_args()
    if not ARGS.skip_cheap:
        cheap_id_file_check()
    stage_search()
