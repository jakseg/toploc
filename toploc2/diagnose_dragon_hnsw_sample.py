#!/usr/bin/env python3
"""DECISIVE cheap experiment for the dragon HNSW failure -- runs in MINUTES, no
4-5h full rebuild, no 129 GB load.

State of the investigation (see CLAUDE.md):
  - exact (brute-force IP) is HEALTHY on the dev queries (hit-rate ~0.68).
  - HNSW recall@10 vs exact is ~0 even at efSearch=4000, and the batched baseline
    latency is ~20x FASTER than snowflake (dragon 0.13 ms/q vs snowflake 2.6 ms/q
    at ef=200) -> the graph is barely being traversed.
  - A CLEAN, uninterrupted REBUILD reproduced the bug -> it is NOT one-time
    abort/resume corruption. It is SYSTEMATIC to the dragon build.
  - Doc norms are ~constant (~65, std 0.41).

The constant-norm finding has a sharp consequence that this script tests: HNSW
neighbour selection is SCALE-INVARIANT (all comparisons are relative), so for
constant-norm vectors, building on normalised vs un-normalised vectors must yield
the SAME graph -> the CLAUDE.md "normalise / MIPS->L2" fixes would change NOTHING.
This script checks that prediction directly, and whether swapping the METRIC (L2
instead of IP) or the geometry helps -- on a representative SAMPLE small enough to
build in minutes, so we stop guessing before committing to another long rebuild.

Method:
  1. Reconstruct a representative sample of doc vectors from the (mmap'd) exact
     index -- evenly spaced contiguous blocks across the whole 38.6M so the
     MARCO/CAR mixture (and thus the navigation problem) is preserved. A pure
     low-node (all-MARCO) sample would hide the bug.
  2. Ground truth = brute-force IP top-10 over THAT sample (isolates HNSW
     navigability from qrels relevance -- we ask only "does the graph find its
     own exact NN?").
  3. Build small HNSW indexes three ways and sweep efSearch, reporting recall@10
     vs the sample-exact:
        (i)   raw IP, un-normalised           -> should reproduce ~0 recall
        (ii)  IP, L2-normalised docs+queries  -> if == (i): normalisation is
              provably irrelevant (scale-invariance confirmed); the two CLAUDE.md
              fixes are dead ends
        (iii) L2 metric, L2-normalised         -> tests whether the FAISS IP-HNSW
              path specifically is the problem vs the well-trodden L2 path

Read-off:
  - (i) bad, (ii) bad, (iii) GOOD  -> rebuild dragon HNSW as normalised + METRIC_L2
        (MIPS->L2 preserves the exact IP ranking; 0 accuracy loss). Real fix found.
  - (i) bad, (ii) GOOD             -> norms are NOT as constant as measured; plain
        normalised-IP rebuild fixes it.
  - all three bad                  -> intrinsic dragon-plus geometry (flat IP
        landscape); graph ANN is not viable for dragon at this scale -> use exact
        or IVF for the dragon arm and report HNSW as a negative result. No amount
        of rebuilding helps.

RAM: mmaps the exact index; only the SAMPLE (N x 768 float32) is resident
(default 300k -> ~0.9 GB). Run from inside toploc2/:
    python diagnose_dragon_hnsw_sample.py                 # 300k sample
    python diagnose_dragon_hnsw_sample.py --sample 1000000 --blocks 40
"""
import argparse
import os
import time

import numpy as np

import toploc2_hnsw_pure_python as qlr

MODEL = "dragon"
M = 32
EF_CONSTRUCTION = 200  # same as the real dragon build (see create_index.py)

cache_dir = qlr.CACHE_DIRS[MODEL]
exact_index_path = os.path.join(cache_dir, "exact_index.index")


def sample_doc_vectors(exact, n_total, n_sample, n_blocks):
    """Representative sample via evenly spaced contiguous blocks (sequential
    reconstruct_n reads -> fast; spans the MARCO+CAR regions)."""
    block = max(1, n_sample // n_blocks)
    offsets = np.linspace(0, n_total - block, n_blocks).astype(np.int64)
    parts = []
    for off in offsets:
        parts.append(exact.reconstruct_n(int(off), int(block)))
    vecs = np.ascontiguousarray(np.vstack(parts), dtype="float32")
    return vecs


def build_hnsw(vecs, metric, normalize):
    import faiss

    x = vecs.copy()
    if normalize:
        faiss.normalize_L2(x)
    idx = faiss.IndexHNSWFlat(x.shape[1], M, metric)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    t0 = time.time()
    idx.add(x)
    return idx, time.time() - t0


def recall_vs_exact(idx, qv, normalize, gt_top, ef_list, k=10):
    import faiss

    q = qv.copy()
    if normalize:
        faiss.normalize_L2(q)
    out = {}
    for ef in ef_list:
        idx.hnsw.efSearch = ef
        t0 = time.time()
        _, I = idx.search(q, k)
        ms = (time.time() - t0) * 1000 / len(q)
        rec = np.mean([len(set(int(x) for x in I[r] if x != -1) & gt_top[r]) / k
                       for r in range(len(q))])
        out[ef] = (float(rec), ms)
    return out


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300_000)
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--ef-list", default="10,64,200,1000")
    args = ap.parse_args()
    ef_list = [int(x) for x in args.ef_list.split(",")]

    print("=== load exact (mmap) + sample doc vectors ===", flush=True)
    exact = faiss.read_index(exact_index_path, faiss.IO_FLAG_MMAP)
    vecs = sample_doc_vectors(exact, exact.ntotal, args.sample, args.blocks)
    norms = np.linalg.norm(vecs, axis=1)
    print(f"sampled {len(vecs):,} doc vecs (dim={vecs.shape[1]}) across {args.blocks} "
          f"blocks | norm min={norms.min():.2f} p50={np.median(norms):.2f} "
          f"max={norms.max():.2f} std={norms.std():.3f}", flush=True)

    # dev queries (with qrels, so it's the same target region as the real run)
    qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=False)
    have = set(qrels)
    eval_keys = [q for q in q_ids if q in have][: args.nq]
    row_of = {q: i for i, q in enumerate(q_ids)}
    qv = np.ascontiguousarray(q_emb[[row_of[k] for k in eval_keys]], dtype="float32")
    print(f"using {len(eval_keys)} dev queries", flush=True)

    # ground truth = brute-force IP top-10 over the SAMPLE (raw IP == the model's
    # native metric; normalising the query doesn't change the argmax here)
    flat = faiss.IndexFlatIP(vecs.shape[1])
    flat.add(vecs)
    _, gtI = flat.search(qv, 10)
    gt_top = [set(int(x) for x in row if x != -1) for row in gtI]

    configs = [
        ("(i)   raw IP, un-normalised   ", faiss.METRIC_INNER_PRODUCT, False),
        ("(ii)  IP, L2-normalised       ", faiss.METRIC_INNER_PRODUCT, True),
        ("(iii) L2 metric, L2-normalised", faiss.METRIC_L2, True),
    ]
    print("\n=== HNSW recall@10 vs the sample's own exact top-10 ===", flush=True)
    print(f"(M={M}, efConstruction={EF_CONSTRUCTION}; recall near 100% = the graph "
          f"finds its own NN)\n", flush=True)
    results = {}
    for label, metric, norm in configs:
        idx, build_s = build_hnsw(vecs, metric, norm)
        res = recall_vs_exact(idx, qv, norm, gt_top, ef_list)
        results[label] = res
        cells = "  ".join(f"ef{ef}={100*res[ef][0]:5.1f}% ({res[ef][1]:.2f}ms)"
                          for ef in ef_list)
        print(f"  {label}  build={build_s:5.1f}s  {cells}", flush=True)
        del idx

    print("\n=== verdict ===", flush=True)
    top_ef = ef_list[-1]
    r_i = results["(i)   raw IP, un-normalised   "][top_ef][0]
    r_ii = results["(ii)  IP, L2-normalised       "][top_ef][0]
    r_iii = results["(iii) L2 metric, L2-normalised"][top_ef][0]
    if r_iii > 0.8 and r_i < 0.3:
        print("L2-metric build recovers recall -> the FAISS IP-HNSW path is the "
              "problem. FIX = rebuild dragon HNSW as MIPS->L2 (normalise + "
              "METRIC_L2), which preserves the exact IP ranking (0 accuracy loss).",
              flush=True)
    elif r_ii > 0.8 and r_i < 0.3:
        print("Normalised-IP build recovers recall while raw does not -> norms are "
              "NOT effectively constant; a normalised rebuild fixes it.", flush=True)
    elif r_i < 0.3 and r_ii < 0.3 and r_iii < 0.3:
        print("ALL three fail even on a small in-RAM build -> this is intrinsic to "
              "the dragon-plus embedding geometry (flat IP landscape), NOT the "
              "build/normalisation/metric. Graph ANN is not viable for dragon here; "
              "use exact/IVF for the dragon arm and report HNSW as a negative "
              "result. Another full rebuild will NOT help.", flush=True)
    else:
        print("Mixed -> raise --sample / --nq and read the per-config recalls above. "
              "abs(recall(i) - recall(ii)) ~ 0 confirms normalisation is irrelevant "
              "(scale-invariance).", flush=True)


if __name__ == "__main__":
    main()
