#!/usr/bin/env python3
"""We now KNOW the dragon HNSW index has the right vectors, right order, right ids
(verify_dragon_index_alignment.py: same_pos=True everywhere). So the accuracy loss
is NOT a label bug -- it is HNSW SEARCH QUALITY: the graph returns wrong neighbours
while exact (exhaustive) is fine. Prime suspect: dragon is NOT L2-normalised, and
an HNSW graph on raw inner product with varying norms degenerates (dominant
high-norm 'hub' vectors, no triangle inequality). Symptom already seen: searching
any doc vector in exact returns the SAME node 15666126 -> a hub.

This script settles the cause with two cheap measurements:

  1. NORM DISTRIBUTION. If vector norms vary a lot and a few (incl. node 15666126)
     are far above the rest, raw-IP HNSW is the wrong tool -> the graph is
     degenerate by construction, not by the chaotic build. Also reports the hub's
     norm and how often exact's own dev-query top-10 contains the hub.

  2. RECALL@10 vs efSearch. HNSW top-10 overlap with the exact top-10 on real dev
     queries, sweeping efSearch. If recall stays ~0 even at efSearch=4000 -> the
     graph is fundamentally broken/degenerate (normalisation problem, a same-params
     rebuild will NOT help). If recall climbs steeply with efSearch -> it was just
     an operating-point / efSearch issue, not the vectors.

RAM: full-loads HNSW (~129 GB) + MMAPs exact. Run from inside toploc2/:
    python diagnose_dragon_hnsw_graph.py            # 20 dev queries
    python diagnose_dragon_hnsw_graph.py --nq 50
"""
import argparse
import os

import numpy as np

import toploc2_hnsw_pure_python as qlr

MODEL = "dragon"
DATASET = "msmarco-on-cast"
HUB = 15666126  # the node exact keeps returning for arbitrary doc vectors

cache_dir = qlr.CACHE_DIRS[MODEL]
hnsw_index_path = os.path.join(cache_dir, "hnsw_index.index")
hnsw_ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
exact_index_path = os.path.join(cache_dir, "exact_index.index")


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=20, help="dev queries to test")
    ap.add_argument("--norm-sample", type=int, default=20000)
    ap.add_argument("--ef-list", default="10,64,200,1000,4000")
    args = ap.parse_args()
    ef_list = [int(x) for x in args.ef_list.split(",")]

    print("=== load indexes ===", flush=True)
    hnsw = faiss.read_index(hnsw_index_path)
    exact = faiss.read_index(exact_index_path, faiss.IO_FLAG_MMAP)
    h_ids = np.load(hnsw_ids_path, allow_pickle=True).astype(str)
    print(f"HNSW ntotal={hnsw.ntotal:,} dim={hnsw.d}  exact ntotal={exact.ntotal:,}",
          flush=True)

    # ---- 1. norm distribution ------------------------------------------------
    print("\n=== 1. vector norm distribution ===", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(exact.ntotal, size=min(args.norm_sample, exact.ntotal),
                     replace=False)
    vecs = np.vstack([exact.reconstruct(int(i)) for i in idx])
    norms = np.linalg.norm(vecs, axis=1)
    pct = np.percentile(norms, [0, 50, 90, 99, 99.9, 100])
    print(f"norms over {len(idx):,} sampled vecs: "
          f"min={pct[0]:.2f} p50={pct[1]:.2f} p90={pct[2]:.2f} "
          f"p99={pct[3]:.2f} p99.9={pct[4]:.2f} max={pct[5]:.2f}", flush=True)
    print(f"mean={norms.mean():.2f} std={norms.std():.2f} "
          f"max/median={pct[5]/max(pct[1],1e-6):.1f}x", flush=True)
    hub_norm = float(np.linalg.norm(exact.reconstruct(HUB)))
    print(f"HUB node {HUB} norm = {hub_norm:.2f}  "
          f"(= {hub_norm/max(pct[1],1e-6):.1f}x the median)  pid={h_ids[HUB]}",
          flush=True)
    if pct[5] > 1.3 * pct[1]:
        print("  -> norms vary substantially: raw inner-product HNSW is prone to "
              "hub domination / bad greedy routing. Strong support for the "
              "normalisation hypothesis.", flush=True)
    else:
        print("  -> norms are fairly uniform; hub domination is a weaker "
              "explanation. Focus on the recall-vs-ef result below.", flush=True)

    # ---- 2. recall@10 vs efSearch on real dev queries ------------------------
    print("\n=== 2. HNSW recall@10 vs exact, sweeping efSearch ===", flush=True)
    qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
    indexed = set(h_ids.tolist())
    fq = {q: {p: s for p, s in d.items() if p in indexed} for q, d in qrels.items()}
    fq = {q: d for q, d in fq.items() if d}
    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=False)
    row_of = {q: i for i, q in enumerate(q_ids)}
    eval_keys = [q for q in q_ids if q in fq][: args.nq]
    qv = np.ascontiguousarray(q_emb[[row_of[k] for k in eval_keys]], dtype="float32")
    print(f"testing {len(eval_keys)} dev queries", flush=True)

    eD, eI = exact.search(qv, 10)
    exact_top = [set(int(x) for x in row if x != -1) for row in eI]
    # does the hub pollute exact's own dev-query results?
    hub_in_exact = np.mean([HUB in s for s in exact_top])
    print(f"hub node {HUB} appears in exact dev-query top-10: "
          f"{100*hub_in_exact:.1f}% of queries", flush=True)

    for ef in ef_list:
        hnsw.hnsw.efSearch = ef
        hD, hI = hnsw.search(qv, 10)
        recalls = []
        for r in range(len(eval_keys)):
            got = set(int(x) for x in hI[r] if x != -1)
            recalls.append(len(got & exact_top[r]) / 10.0)
        print(f"  efSearch={ef:>5}  mean recall@10 vs exact = "
              f"{100*np.mean(recalls):5.1f}%", flush=True)

    print("\n=== verdict ===", flush=True)
    print("If recall stays near 0 even at the largest efSearch AND norms vary a lot "
          "-> the raw-IP dragon HNSW graph is degenerate. FIX = rebuild the dragon "
          "HNSW on NORMALISED vectors (cosine) or via a MIPS->L2 transform; a "
          "same-params rebuild will reproduce the problem. If recall climbs with "
          "efSearch -> just raise efSearch, no rebuild needed.", flush=True)


if __name__ == "__main__":
    main()
