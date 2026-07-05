#!/usr/bin/env python3
"""Test a COLLEAGUE'S hypothesis: FAISS's *batched* search() returns wrong results
on the dragon index while *single-query* search is correct -- so the eval logic was
fine, but every full run was poisoned by one batched index.search() call.

Why this is plausible HERE (see CLAUDE.md):
  - build_run_baseline() computes NDCG/MRR/Accuracy from ONE batched
    index.search(query_matrix, k) over all dev queries.
  - --threads defaults to os.cpu_count(), so unless the run used --threads 1 the
    batched HNSW search ran multi-threaded (queries parallelised across OMP threads,
    each with its own graph-traversal scratch / VisitedTable).
  - In diagnose_dragon_exact_quality.py BOTH exact and HNSW were searched BATCHED on
    the same qv: exact (IndexFlatIP = a thread-safe BLAS gemm) was GOOD, HNSW was 0.
    A per-thread-scratch race in the graph search path would corrupt HNSW batched but
    not Flat batched -- exactly this pattern -- and early termination from a clobbered
    VisitedTable also explains the ~20x-too-fast latency.

This script settles it by comparing THREE search paths on the SAME dev queries and
the SAME real dragon HNSW index, scoring each against qrels and against each other:

  A. batched, threads = cpu_count   (the suspect path: what the run used)
  B. batched, threads = 1           (batched but no cross-query parallelism)
  C. single-query loop, threads = 1 (the colleague's "correct" reference)

Read-off:
  - C good, A ~0                    -> CONFIRMED: batched multi-thread search is
        broken on this index. If B is ALSO good -> it is a THREADING RACE (fix:
        run search with threads=1, or in per-query / small batches, or upgrade/patch
        faiss). If B is ALSO bad -> the batched *call* itself is wrong regardless of
        threads (fix: loop single queries). Either way the INDEX IS FINE -- no
        rebuild, no normalisation, no geometry problem. Fix the eval's search call.
  - A == B == C (all good or all bad) -> the colleague's bug does NOT apply here;
        it really is the index/geometry -> back to diagnose_dragon_hnsw_sample.py.

exact (batched, cpu_count) is included as a sanity anchor: it should stay good
regardless, isolating the graph-search path as the culprit.

RAM: full-loads the ~129 GB dragon HNSW + mmaps exact. Run from inside toploc2/:
    python diagnose_dragon_batched_search.py --nq 300 --ef 200
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


def hitrate(top_pids_per_q, rel_per_q, k=10):
    hit = []
    for pids, rel in zip(top_pids_per_q, rel_per_q):
        hit.append(1.0 if (set(pids[:k]) & rel) else 0.0)
    return float(np.mean(hit))


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=300)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    ncpu = os.cpu_count() or 1

    print("=== load ===", flush=True)
    hnsw = faiss.read_index(hnsw_index_path)
    exact = faiss.read_index(exact_index_path, faiss.IO_FLAG_MMAP)
    h_ids = np.load(hnsw_ids_path, allow_pickle=True).astype(str)
    hnsw.hnsw.efSearch = args.ef
    print(f"HNSW ntotal={hnsw.ntotal:,} dim={hnsw.d} efSearch={args.ef} | cpu={ncpu}",
          flush=True)

    qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
    indexed = set(h_ids.tolist())
    fq = {q: {p for p in d if p in indexed} for q, d in qrels.items()}
    fq = {q: d for q, d in fq.items() if d}
    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=False)
    row_of = {q: i for i, q in enumerate(q_ids)}
    eval_keys = [q for q in q_ids if q in fq][: args.nq]
    qv = np.ascontiguousarray(q_emb[[row_of[k] for k in eval_keys]], dtype="float32")
    rel = [fq[k] for k in eval_keys]
    print(f"scoring {len(eval_keys)} dev queries with qrels\n", flush=True)

    def to_pids(I):
        return [[h_ids[int(x)] for x in row if x != -1] for row in I]

    # C. single-query loop, threads=1 (the reference the colleague trusts)
    faiss.omp_set_num_threads(1)
    I_loop = np.full((len(qv), args.k), -1, dtype="int64")
    for r in range(len(qv)):
        _, i1 = hnsw.search(qv[r:r + 1], args.k)
        I_loop[r] = i1[0]
    pids_loop = to_pids(I_loop)

    # B. batched, threads=1
    faiss.omp_set_num_threads(1)
    _, I_b1 = hnsw.search(qv, args.k)
    pids_b1 = to_pids(I_b1)

    # A. batched, threads=cpu_count (what a default run used)
    faiss.omp_set_num_threads(ncpu)
    _, I_bN = hnsw.search(qv, args.k)
    pids_bN = to_pids(I_bN)

    # exact sanity anchor (batched, cpu_count)
    _, I_ex = exact.search(qv, args.k)
    pids_ex = to_pids(I_ex)

    def overlap(a, b):
        return float(np.mean([len(set(x[:args.k]) & set(y[:args.k])) / args.k
                              for x, y in zip(a, b)]))

    print("=== top-%d agreement vs the single-query loop (C) ===" % args.k, flush=True)
    print(f"  A batched(threads={ncpu}) vs C loop : {100*overlap(pids_bN, pids_loop):5.1f}%",
          flush=True)
    print(f"  B batched(threads=1)      vs C loop : {100*overlap(pids_b1, pids_loop):5.1f}%",
          flush=True)

    print("\n=== hit-rate@%d vs qrels (higher = correct retrieval) ===" % args.k,
          flush=True)
    print(f"  A batched(threads={ncpu}) : {hitrate(pids_bN, rel, args.k):.3f}", flush=True)
    print(f"  B batched(threads=1)      : {hitrate(pids_b1, rel, args.k):.3f}", flush=True)
    print(f"  C single-query loop       : {hitrate(pids_loop, rel, args.k):.3f}", flush=True)
    print(f"  exact (anchor, batched)   : {hitrate(pids_ex, rel, args.k):.3f}", flush=True)

    print("\n=== verdict ===", flush=True)
    hA = hitrate(pids_bN, rel, args.k)
    hB = hitrate(pids_b1, rel, args.k)
    hC = hitrate(pids_loop, rel, args.k)
    if hC > 0.3 and hA < 0.3 * max(hC, 1e-9):
        if hB > 0.3:
            print("CONFIRMED + it is a THREADING RACE: batched multi-thread search is "
                  "broken on this dragon index, single-query and threads=1 are fine. "
                  "The index is CORRECT. Fix: run index.search with threads=1 (or in "
                  "per-query/small batches) for the eval, or patch/upgrade faiss. No "
                  "rebuild / normalisation / geometry issue.", flush=True)
        else:
            print("CONFIRMED: the batched search CALL returns wrong results here "
                  "regardless of threads; only the single-query loop is correct. The "
                  "index is CORRECT -- loop single queries in the eval. No rebuild.",
                  flush=True)
    elif abs(hA - hC) < 0.05 and abs(hB - hC) < 0.05:
        print("NOT this bug: batched and single-query agree (A==B==C). The failure is "
              "the index/geometry, not the batched call -> run "
              "diagnose_dragon_hnsw_sample.py.", flush=True)
    else:
        print("Mixed -> inspect the agreement + hit-rate numbers above (raise --nq).",
              flush=True)


if __name__ == "__main__":
    main()
