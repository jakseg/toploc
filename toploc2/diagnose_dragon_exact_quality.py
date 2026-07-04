#!/usr/bin/env python3
"""FINAL fork for the dragon msmarco-dev accuracy bug. We know: vectors+ids are
correct (verify_dragon_index_alignment: same_pos=True), norms are ~constant (~65,
no hub), the encoder is identical to the working CAsT path, and HNSW recall@10 vs
exact is 0% even at efSearch=4000 while CAsT retrieval works on the SAME index.
So it is a GRAPH-level reachability failure for the dev-query targets -- UNLESS
exact itself is a bad reference (dev embeddings not aligned with the doc space).

This script settles rebuild-vs-not by scoring BOTH exact and HNSW against the
REAL qrels (relevance), plus a reachability probe:

  - exact MRR@10 GOOD, HNSW ~0   -> exact is a valid reference; the right MARCO
      docs ARE in the index and brute-force finds them; HNSW's graph cannot reach
      them -> GRAPH CORRUPTION from the chaotic build -> CLEAN REBUILD fixes it.
  - exact MRR@10 also ~0         -> dev embeddings do not align with the doc
      embeddings (data/encoding), NOT the graph -> a rebuild will NOT help;
      re-encode / check the doc-side dragon embeddings.

Reachability probe: search HNSW with big k (1000) at efSearch=4000 and check what
fraction of exact's top-10 nodes appear at all. ~0% => those nodes are effectively
disconnected in the graph (corruption); high => merely ranked out (unlikely here).

RAM: full-loads HNSW (~129 GB) + MMAPs exact. Run from inside toploc2/:
    python diagnose_dragon_exact_quality.py --nq 100
"""
import argparse
import os

import numpy as np

import toploc2_hnsw_pure_python as qlr

MODEL = "dragon"
cache_dir = qlr.CACHE_DIRS[MODEL]
hnsw_index_path = os.path.join(cache_dir, "hnsw_index.index")
hnsw_ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
exact_index_path = os.path.join(cache_dir, "exact_index.index")


def mrr_recall(top_pids_per_q, rel_per_q, k=10):
    """Return (MRR@k, mean Recall@k, hit-rate) over queries."""
    mrr, rec, hit = [], [], []
    for pids, rel in zip(top_pids_per_q, rel_per_q):
        pids = pids[:k]
        rr = 0.0
        for r, p in enumerate(pids, 1):
            if p in rel:
                rr = 1.0 / r
                break
        mrr.append(rr)
        inter = len(set(pids) & rel)
        rec.append(inter / max(len(rel), 1))
        hit.append(1.0 if inter else 0.0)
    return np.mean(mrr), np.mean(rec), np.mean(hit)


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--ef", type=int, default=200)
    args = ap.parse_args()

    print("=== load ===", flush=True)
    hnsw = faiss.read_index(hnsw_index_path)
    exact = faiss.read_index(exact_index_path, faiss.IO_FLAG_MMAP)
    h_ids = np.load(hnsw_ids_path, allow_pickle=True).astype(str)
    id_map = {i: h_ids[i] for i in range(len(h_ids))}
    print(f"HNSW ntotal={hnsw.ntotal:,}  exact ntotal={exact.ntotal:,}", flush=True)

    qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
    indexed = set(h_ids.tolist())
    fq = {q: {p for p in d if p in indexed} for q, d in qrels.items()}
    fq = {q: d for q, d in fq.items() if d}
    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=False)
    row_of = {q: i for i, q in enumerate(q_ids)}
    eval_keys = [q for q in q_ids if q in fq][: args.nq]
    qv = np.ascontiguousarray(q_emb[[row_of[k] for k in eval_keys]], dtype="float32")
    rel_per_q = [fq[k] for k in eval_keys]
    print(f"scoring {len(eval_keys)} dev queries that have qrels in the index",
          flush=True)

    # exact vs qrels
    eD, eI = exact.search(qv, 10)
    exact_pids = [[id_map[int(x)] for x in row if x != -1] for row in eI]
    e_mrr, e_rec, e_hit = mrr_recall(exact_pids, rel_per_q)

    # hnsw vs qrels
    hnsw.hnsw.efSearch = args.ef
    hD, hI = hnsw.search(qv, 10)
    hnsw_pids = [[id_map[int(x)] for x in row if x != -1] for row in hI]
    h_mrr, h_rec, h_hit = mrr_recall(hnsw_pids, rel_per_q)

    # reachability: are exact's top-10 nodes anywhere in a big HNSW result?
    hnsw.hnsw.efSearch = 4000
    _, hI_big = hnsw.search(qv, 1000)
    reach = []
    for r in range(len(eval_keys)):
        et = set(int(x) for x in eI[r] if x != -1)
        got = set(int(x) for x in hI_big[r] if x != -1)
        reach.append(len(et & got) / max(len(et), 1))
    reach = float(np.mean(reach))

    print("\n=== results (vs qrels) ===", flush=True)
    print(f"EXACT  MRR@10={e_mrr:.3f}  Recall@10={e_rec:.3f}  hit-rate={e_hit:.3f}",
          flush=True)
    print(f"HNSW   MRR@10={h_mrr:.3f}  Recall@10={h_rec:.3f}  hit-rate={h_hit:.3f}  "
          f"(efSearch={args.ef})", flush=True)
    print(f"reachability: exact top-10 present in HNSW top-1000@ef4000 = "
          f"{100*reach:.1f}%", flush=True)

    print("\n=== verdict ===", flush=True)
    if e_hit >= 0.3 and h_hit < 0.3 * e_hit:
        print("EXACT retrieves relevant docs, HNSW does not -> exact is a valid "
              "reference; the correct MARCO docs are indexed but the HNSW GRAPH "
              "cannot reach them (reachability {:.0f}%). This is graph corruption "
              "from the abort/resume build. FIX = CLEAN REBUILD of the dragon HNSW "
              "(delete index+ids+checkpoint, one uninterrupted run).".format(100*reach),
              flush=True)
    elif e_hit < 0.15:
        print("EXACT is ALSO bad vs qrels -> the dev embeddings do not align with "
              "the doc embeddings in the index (data/encoding), NOT the graph. A "
              "rebuild will NOT help. Check: were the MARCO doc vectors in the CAST "
              "index encoded with the dragon CONTEXT encoder consistently with the "
              "dragon QUERY dev embeddings? Re-encode / verify the doc side.",
              flush=True)
    else:
        print("Mixed signal -> raise --nq and inspect the per-metric numbers above "
              "(exact quality is the deciding factor).", flush=True)


if __name__ == "__main__":
    main()
