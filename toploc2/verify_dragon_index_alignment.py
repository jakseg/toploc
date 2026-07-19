#!/usr/bin/env python3
"""DECIDE whether the dragon HNSW index+ids are internally aligned, or whether an
abort/resume during the (non-atomic) checkpoint left the vectors and the pid-list
out of sync. This is the go/no-go test before spending 4-5h on a rebuild.

Background: create_index.py writes index -> ids -> checkpoint NON-atomically. A
kill between those writes + a later resume can leave the HNSW index and hnsw_ids
on different states, so node i holds a different vector than hnsw_ids[i] claims.
Symptom already seen: good IP scores (~35) but 0/10 overlap vs exact (wrong pids).

Method (uses the TRUSTED exact IndexFlatIP as reference; exact hits the qrels):
for a random sample of node indices i,
  - reconstruct the HNSW vector at node i and the exact vector at position i;
  - `same_pos`  = are they the SAME vector? -> is HNSW's vector-order == exact's?
  - look the HNSW vector up IN exact: `self_found` = is it present at all
    (self-match score ~= ||v||^2)? -> distinguishes "reordered" from
    "different vectors entirely";
  - `node_ok`   = does exact assign the HNSW vector the SAME pid that hnsw_ids[i]
    claims? -> the decisive per-node correctness check.
Also records the offset (j - i) between where a vector sits in HNSW vs exact, so a
constant/block shift (typical resume artifact) is distinguishable from a full
scramble.

Decision matrix (printed at the end):
  self_found~100 & same_pos~100 & node_ok~100 -> INDEX IS FINE. Do NOT rebuild;
      the accuracy bug is elsewhere (query side / eval), investigate there.
  self_found~100 & same_pos low & node_ok low -> SAME vectors, WRONG order/ids
      -> the resume desync. REBUILD (clean) is the safe fix; a pid-remap is
      possible but risky on a possibly-disturbed graph.
  self_found low -> HNSW holds DIFFERENT vectors than exact (wrong/re-encoded
      embeddings) -> REBUILD from the correct embeddings.

RAM: full-loads HNSW (~129 GB) + MMAPs exact. Same footprint as
diagnose_dragon_idmap.py. Run from inside toploc2/:
    python verify_dragon_index_alignment.py            # 2000 sample nodes
    python verify_dragon_index_alignment.py --n 5000
"""
import argparse
import os

import numpy as np

import qlr

MODEL = "dragon"

cache_dir = qlr.CACHE_DIRS[MODEL]
hnsw_index_path = os.path.join(cache_dir, "hnsw_index.index")
hnsw_ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
exact_index_path = os.path.join(cache_dir, "exact_index.index")
exact_ids_path = os.path.join(cache_dir, "exact_ids.npy")


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="sample node count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (os.path.exists(exact_index_path) and os.path.exists(exact_ids_path)):
        raise SystemExit(
            f"exact index/ids missing under {cache_dir} -> this test needs the "
            f"trusted exact IndexFlatIP as reference. Build it first "
            f"(create_index.py dragon exact) or point cache_dir at it.")

    print("=== load ids ===", flush=True)
    h_ids = np.load(hnsw_ids_path, allow_pickle=True).astype(str)
    e_ids = np.load(exact_ids_path, allow_pickle=True).astype(str)
    print(f"len(hnsw_ids)={len(h_ids):,}  len(exact_ids)={len(e_ids):,}", flush=True)
    ids_identical = len(h_ids) == len(e_ids) and np.array_equal(h_ids, e_ids)
    print(f"hnsw_ids == exact_ids element-wise: {ids_identical}", flush=True)

    print(f"\n=== load indexes (HNSW full ~{os.path.getsize(hnsw_index_path)/1e9:.0f} GB, "
          f"exact mmap) ===", flush=True)
    hnsw = faiss.read_index(hnsw_index_path)
    exact = faiss.read_index(exact_index_path, faiss.IO_FLAG_MMAP)
    print(f"HNSW ntotal={hnsw.ntotal:,} dim={hnsw.d}", flush=True)
    print(f"exact ntotal={exact.ntotal:,} dim={exact.d}", flush=True)
    if hnsw.ntotal != len(h_ids):
        print(f"  !! HNSW ntotal ({hnsw.ntotal:,}) != len(hnsw_ids) ({len(h_ids):,}) "
              f"-> length desync already proves the artifacts don't match. REBUILD.",
              flush=True)

    n = min(args.n, hnsw.ntotal)
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(hnsw.ntotal, size=n, replace=False)
    sample.sort()

    same_pos = 0          # HNSW vec at i == exact vec at i
    self_found = 0        # HNSW vec at i exists in exact (self-match)
    node_ok = 0           # exact's pid for HNSW vec at i == hnsw_ids[i]
    offsets = []          # j - i, where j = exact position of HNSW's vec at i

    print(f"\n=== probing {n} random nodes ===", flush=True)
    for c, i in enumerate(sample):
        i = int(i)
        hv = hnsw.reconstruct(i).astype("float32")
        ev = exact.reconstruct(i).astype("float32")
        if np.allclose(hv, ev, atol=1e-4):
            same_pos += 1
        D, I = exact.search(hv[None], 1)
        j = int(I[0, 0])
        selfnorm = float(hv @ hv)
        is_self = abs(float(D[0, 0]) - selfnorm) < 1e-3 * max(selfnorm, 1.0)
        if is_self:
            self_found += 1
            offsets.append(j - i)
            if e_ids[j] == h_ids[i]:
                node_ok += 1
        if c < 15:
            print(f"  node {i:>10}  same_pos={np.allclose(hv, ev, atol=1e-4)!s:>5}  "
                  f"self_found={is_self!s:>5}  exact_pos_j={j:>10}  "
                  f"pid@node='{h_ids[i]}'  correct_pid='{e_ids[j] if is_self else '?'}'",
                  flush=True)

    print("\n=== summary ===", flush=True)
    print(f"same_pos   (HNSW order == exact order):        {same_pos}/{n} "
          f"= {100*same_pos/n:.1f}%", flush=True)
    print(f"self_found (HNSW vec present in exact):         {self_found}/{n} "
          f"= {100*self_found/n:.1f}%", flush=True)
    denom = self_found or 1
    print(f"node_ok    (hnsw_ids[i] is the vector's pid):  {node_ok}/{self_found} "
          f"= {100*node_ok/denom:.1f}%  (of self_found)", flush=True)
    if offsets:
        off = np.array(offsets)
        uniq, cnt = np.unique(off, return_counts=True)
        top = uniq[np.argsort(-cnt)[:5]]
        print(f"offset (j-i): min={off.min()} max={off.max()} "
              f"nonzero={100*np.mean(off != 0):.1f}%  most-common={top.tolist()}",
              flush=True)

    print("\n=== verdict ===", flush=True)
    if self_found < 0.9 * n:
        print("HNSW vectors are largely NOT in exact -> the HNSW index holds "
              "DIFFERENT/re-encoded vectors than the exact reference. REBUILD the "
              "dragon HNSW index from the correct embeddings.", flush=True)
    elif node_ok >= 0.98 * self_found and same_pos >= 0.98 * n:
        print("INDEX IS ALIGNED (vectors present, right order, right pids). The "
              "accuracy bug is NOT in the index artifact -> look at the query/eval "
              "side. Do NOT rebuild yet.", flush=True)
    else:
        print("Vectors ARE present in exact but node->pid is WRONG (and/or order "
              "differs) -> classic abort/resume DESYNC: index and hnsw_ids are on "
              "different build states. SAFE FIX = clean REBUILD (delete index+ids+"
              "checkpoint first, one uninterrupted run). A pid-remap from this test "
              "is possible but not recommended on a graph built during the chaos.",
              flush=True)


if __name__ == "__main__":
    main()
