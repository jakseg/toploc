#!/usr/bin/env python3
"""LOW-RAM version of the batched-vs-single-query test (colleague's hypothesis:
FAISS batched search() returns wrong results on the dragon index, single-query is
correct). Instead of loading the 129 GB real HNSW, this reproduces the SEARCH CODE
PATH on a small HNSW built from a SAMPLE of real dragon doc vectors -- the suspected
batched multi-thread bug lives in the search kernel, not in the index size, so if it
is real it shows here too. ~4-5 GB RAM, a few minutes.

Doc vectors come via reconstruct_n from the MMAP'd exact index (reads only the
sampled blocks, ~1 GB -- NOT the 118 GB full scan). Sample spans evenly-spaced
blocks so the MARCO/CAR mixture (and its geometry) is preserved.

Three search paths on the SAME small HNSW + SAME queries:
  A. batched, threads = cpu_count   (what a default run uses)
  B. batched, threads = 1
  C. single-query loop, threads = 1 (the "correct" reference)
plus the sample's own exact top-10 (IndexFlatIP) as the correctness anchor.

Read-off:
  - A disagrees with C (and with the exact top-10) while B/C agree with exact
      -> REPRODUCED: batched multi-thread HNSW search is wrong on dragon geometry.
      Strongly implies the full-index run was poisoned the same way -> the index is
      fine; fix the eval's search call (threads=1 / per-query). No rebuild.
  - A == B == C == exact (all agree) -> the batched bug does NOT reproduce at small
      scale. Either it is specific to the full saved index (then confirm with the
      real-index test / a --threads 1 re-run) OR it is not this bug at all ->
      diagnose_dragon_hnsw_sample.py for the geometry angle.

NOTE: a NEGATIVE result here is not conclusive (a full-index-only bug would be
missed); a POSITIVE result is strong. Run from inside toploc2/:
    python diagnose_dragon_batched_small.py                       # 300k, 500 q
    python diagnose_dragon_batched_small.py --sample 1000000 --nq 1000
"""
import argparse
import os

import numpy as np

import toploc2_hnsw_pure_python as qlr

MODEL = "dragon"
M = 32
EF_CONSTRUCTION = 200
cache_dir = qlr.CACHE_DIRS[MODEL]
exact_index_path = os.path.join(cache_dir, "exact_index.index")


def sample_doc_vectors(exact, n_total, n_sample, n_blocks):
    block = max(1, n_sample // n_blocks)
    offsets = np.linspace(0, n_total - block, n_blocks).astype(np.int64)
    parts = [exact.reconstruct_n(int(off), int(block)) for off in offsets]
    return np.ascontiguousarray(np.vstack(parts), dtype="float32")


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300_000)
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--nq", type=int, default=500)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    ncpu = os.cpu_count() or 1

    print("=== sample dragon doc vectors (mmap exact, sampled blocks only) ===",
          flush=True)
    exact = faiss.read_index(exact_index_path, faiss.IO_FLAG_MMAP)
    vecs = sample_doc_vectors(exact, exact.ntotal, args.sample, args.blocks)
    del exact  # done with the big file
    print(f"sampled {len(vecs):,} docs (dim={vecs.shape[1]}) | cpu={ncpu}", flush=True)

    # small HNSW, exactly like the real dragon build: raw IP, un-normalised
    idx = faiss.IndexHNSWFlat(vecs.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    faiss.omp_set_num_threads(ncpu)  # build can use all cores
    idx.add(vecs)
    idx.hnsw.efSearch = args.ef

    # sample-exact top-k = the correctness anchor
    flat = faiss.IndexFlatIP(vecs.shape[1])
    flat.add(vecs)

    # dev queries (any subset; qrels not needed -- we compare search paths to each
    # other and to the sample-exact)
    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=False)
    qv = np.ascontiguousarray(q_emb[: args.nq], dtype="float32")
    print(f"searching {len(qv)} dev queries at ef={args.ef}\n", flush=True)

    faiss.omp_set_num_threads(ncpu)
    _, I_exact = flat.search(qv, args.k)

    # A. batched, threads = cpu_count
    faiss.omp_set_num_threads(ncpu)
    _, I_bN = idx.search(qv, args.k)
    # B. batched, threads = 1
    faiss.omp_set_num_threads(1)
    _, I_b1 = idx.search(qv, args.k)
    # C. single-query loop, threads = 1
    I_loop = np.full((len(qv), args.k), -1, dtype="int64")
    for r in range(len(qv)):
        _, i1 = idx.search(qv[r:r + 1], args.k)
        I_loop[r] = i1[0]

    def agree(a, b):
        return float(np.mean([len(set(a[r]) & set(b[r])) / args.k
                              for r in range(len(a))]))

    exact_top = [set(int(x) for x in row if x != -1) for row in I_exact]

    def recall(I):
        rows = [set(int(x) for x in row if x != -1) for row in I]
        return float(np.mean([len(rows[r] & exact_top[r]) / args.k
                              for r in range(len(rows))]))

    print("=== top-%d recall vs the sample's exact NN (100%% = search is correct) ==="
          % args.k, flush=True)
    print(f"  A batched(threads={ncpu}) : {100*recall(I_bN):5.1f}%", flush=True)
    print(f"  B batched(threads=1)      : {100*recall(I_b1):5.1f}%", flush=True)
    print(f"  C single-query loop       : {100*recall(I_loop):5.1f}%", flush=True)

    print("\n=== agreement A vs C (do the two search paths return the same nodes?) ===",
          flush=True)
    a2c = agree([set(map(int, r)) for r in I_bN], [set(map(int, r)) for r in I_loop])
    b2c = agree([set(map(int, r)) for r in I_b1], [set(map(int, r)) for r in I_loop])
    print(f"  A batched(threads={ncpu}) vs C loop : {100*a2c:5.1f}%", flush=True)
    print(f"  B batched(threads=1)      vs C loop : {100*b2c:5.1f}%", flush=True)

    print("\n=== verdict ===", flush=True)
    rA, rB, rC = recall(I_bN), recall(I_b1), recall(I_loop)
    if rC > 0.6 and rA < 0.5 * rC:
        tag = ("THREADING RACE (threads=1 fine, multi-thread broken)" if rB > 0.6
               else "the batched CALL itself (both batched paths broken)")
        print(f"REPRODUCED: batched multi-thread search is wrong on dragon geometry "
              f"-> {tag}. The full run was almost certainly poisoned the same way; "
              f"the INDEX is fine -> fix the eval search (threads=1 / per-query), no "
              f"rebuild. Confirm on the real index with a --threads 1 re-run when RAM "
              f"allows.", flush=True)
    elif min(rA, rB, rC) > 0.6:
        print("All paths agree and are correct at small scale -> the batched bug does "
              "NOT reproduce here. Either it is specific to the full saved index "
              "(confirm with the real-index test / --threads 1 re-run) or it is not "
              "this bug -> diagnose_dragon_hnsw_sample.py (geometry).", flush=True)
    else:
        print("Low recall on ALL paths incl. single-query -> not a batched issue; the "
              "small HNSW itself can't find its NN -> geometry "
              "(diagnose_dragon_hnsw_sample.py).", flush=True)


if __name__ == "__main__":
    main()
