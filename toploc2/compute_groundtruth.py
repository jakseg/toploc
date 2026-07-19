#!/usr/bin/env python3
"""
Exact (exhaustive) top-k for the msmarco dev queries — the ground truth behind
the paper's Accuracy@10 metric (fraction of the true top-k retrieved).

Run this ONCE per (model, dataset). It loads only the document embeddings (or the
exact IndexFlatIP), NOT the big HNSW index, so it does not compete for RAM with a
QLR/baseline run. The result is a small JSON map {dev_query_id: [pid, ...]} that
toploc2_hnsw_pure_python.py picks up automatically to report Accuracy@k.

Two methods:
  --method stream   (default) stream the document-embedding parquet shards in
                    tiles and keep a running exact top-k. Only needs the doc
                    embeddings (which the HNSW index was built from), so it always
                    works. Reproduces create_index.py's per-model normalize rule.
  --method exact    use the prebuilt exact_index.index (IndexFlatIP) if present.

The pid space matches the run side: for msmarco-on-cast the document ids are
MARCO_<n>/CAR_<hash> (the CAST2019 collection), so Accuracy@k intersects directly
with the run's pids and the MARCO_-prefixed qrels.

Examples:
    python compute_groundtruth.py snowflake --dataset msmarco-on-cast
    python compute_groundtruth.py dragon   --dataset msmarco-on-cast --method exact
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import faiss

import toploc2_hnsw_pure_python as qlr  # path constants + loaders (import-safe)

# Document-embedding collections, mirroring create_index.py's DATASETS. msmarco-on-cast
# searches the CAST2019 collection (its HNSW index is I_D), so it reuses cast2019.
DOC_EMB_BASE = {
    "cast2019": os.environ.get(
        "CAST_EMB_BASE", "/home/toploc2/Datasets/conversational/CAST2019"),
    "msmarco": os.environ.get(
        "MSMARCO_EMB_BASE", "/home/toploc2/Datasets/conversational/msmarco"),
}
DOC_EMB_SUBDIR = {
    "cast2019": {"snowflake": "snowflake_embeddings", "dragon": "dragon_embeddings"},
    "msmarco": {"snowflake": "snowflake", "dragon": "dragon"},
}


def collection_for(dataset):
    return "cast2019" if dataset in ("cast2019", "msmarco-on-cast") else "msmarco"


def read_parquet_shard(pf, dim):
    """Return (ids, (n,dim) float32) for one shard — same schema as create_index.py."""
    table = pq.read_table(pf, columns=["id", "embedding"])
    ids = [str(x) for x in table.column("id").to_pylist()]
    try:
        flat = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        embs = flat.reshape(len(ids), dim).astype("float32")
    except Exception:
        embs = np.array(table.column("embedding").to_pylist(), dtype=np.float32)
    return ids, embs


def select_dev_queries(model_name, dataset, max_turns):
    """Dev-query ids + embeddings, restricted to ids that appear in the qrels —
    a superset of the driver's eval set, so every evaluated query has ground truth.
    Mirrors the driver's normalize rule (both models cosine / L2-normalised now)."""
    normalize = True
    qrels_path = qlr.MSMARCO_QRELS  # msmarco dev qrels (qids are model-agnostic)
    qrels = qlr.load_qrels(qrels_path)
    qids = set(qrels.keys())

    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[model_name]
    print(f"Loading dev-query embeddings from {dev_dir} (normalize={normalize})...", flush=True)
    ids, emb = qlr.load_parquet_embeddings(dev_dir, normalize=normalize)
    keep = [i for i, qid in enumerate(ids) if qid in qids]
    if max_turns:
        keep = keep[:max_turns]
    dev_ids = [ids[i] for i in keep]
    dev_emb = np.ascontiguousarray(emb[keep], dtype="float32")
    print(f"Dev queries with qrels: {len(dev_ids):,} (of {len(ids):,} embedded), dim={dev_emb.shape[1]}",
          flush=True)
    return dev_ids, dev_emb, normalize


def topk_stream(dev_emb, model_name, dataset, k, tile):
    """Exact top-k by streaming the document parquet in doc-tiles of `tile` rows."""
    collection = collection_for(dataset)
    emb_dir = os.path.join(DOC_EMB_BASE[collection], DOC_EMB_SUBDIR[collection][model_name])
    shards = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not shards:
        print(f"ERROR: no document parquet in {emb_dir}", flush=True)
        sys.exit(1)
    dim = len(pq.read_table(shards[0], columns=["embedding"]).column("embedding")[0].as_py())
    normalize = True  # dragon HNSW/index is cosine now (see create_index.py / driver)
    n = dev_emb.shape[0]
    print(f"Streaming {len(shards)} shards from {emb_dir} (dim={dim}, normalize={normalize}, "
          f"tile={tile})...", flush=True)

    # Carry GLOBAL INTEGER doc indices (not pid strings) through the merge so the
    # hot loop stays fully vectorised (object arrays would be ~100x slower); map
    # the final n*k indices back to pids once at the end.
    best_scores = np.full((n, k), -np.inf, dtype="float32")
    best_gidx = np.full((n, k), -1, dtype="int64")
    all_ids = []
    t0 = time.time()
    offset = 0
    for si, pf in enumerate(shards):
        ids, embs = read_parquet_shard(pf, dim)
        if normalize:
            faiss.normalize_L2(embs)
        all_ids.extend(ids)
        m = embs.shape[0]
        for start in range(0, m, tile):
            block = embs[start:start + tile]
            bt = block.shape[0]
            scores = dev_emb @ block.T                      # (n, bt)
            gidx = (offset + start) + np.arange(bt, dtype="int64")
            cat_s = np.concatenate([best_scores, scores], axis=1)
            cat_i = np.concatenate([best_gidx, np.broadcast_to(gidx, (n, bt))], axis=1)
            part = np.argpartition(-cat_s, kth=k - 1, axis=1)[:, :k]
            best_scores = np.take_along_axis(cat_s, part, axis=1)
            best_gidx = np.take_along_axis(cat_i, part, axis=1)
        offset += m
        if (si + 1) % 10 == 0 or (si + 1) == len(shards):
            print(f"  shard {si + 1}/{len(shards)}  docs={offset:,}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    # sort each row's k by descending score, then map global idx -> pid
    order = np.argsort(-best_scores, axis=1)
    best_gidx = np.take_along_axis(best_gidx, order, axis=1)
    all_ids = np.asarray(all_ids, dtype=object)
    return all_ids[best_gidx]


def topk_exact_index(dev_emb, model_name, dataset, k):
    """Exact top-k via the prebuilt exact IndexFlatIP (+ its ids)."""
    cache_dir = qlr.CACHE_DIRS[model_name]
    index_path = os.path.join(cache_dir, "exact_index.index")
    ids_path = os.path.join(cache_dir, "exact_ids.npy")
    if not (os.path.exists(index_path) and os.path.exists(ids_path)):
        print(f"ERROR: exact index/ids not found in {cache_dir} — use --method stream", flush=True)
        sys.exit(1)
    print(f"Loading exact index {index_path} (mmap)...", flush=True)
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    id_array = np.load(ids_path, allow_pickle=True)
    _, idxs = index.search(dev_emb, k)
    return np.array([[str(id_array[j]) if j >= 0 else "" for j in row] for row in idxs], dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["snowflake", "dragon"], nargs="?", default="snowflake")
    ap.add_argument("--dataset", default="msmarco-on-cast",
                    choices=["cast2019", "msmarco", "msmarco-on-cast"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--method", choices=["stream", "exact"], default="stream")
    ap.add_argument("--tile", type=int, default=20000, help="doc rows per matmul tile (stream method)")
    ap.add_argument("--max-turns", type=int, default=0, help="debug cap on dev queries")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", os.cpu_count() or 1)))
    args = ap.parse_args()
    faiss.omp_set_num_threads(args.threads)

    dev_ids, dev_emb, _ = select_dev_queries(args.model, args.dataset, args.max_turns)

    if args.method == "exact":
        gt_pids = topk_exact_index(dev_emb, args.model, args.dataset, args.k)
    else:
        gt_pids = topk_stream(dev_emb, args.model, args.dataset, args.k, args.tile)

    gt = {qid: [p for p in gt_pids[i].tolist() if p] for i, qid in enumerate(dev_ids)}

    cache_dir = qlr.CACHE_DIRS[args.model]
    out_path = qlr.groundtruth_path(cache_dir, args.model, args.dataset, args.k)
    os.makedirs(cache_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt, f)
    print(f"\nWrote exact top-{args.k} for {len(gt):,} dev queries -> {out_path}", flush=True)
    sample = next(iter(gt.items()))
    print(f"  sample {sample[0]} -> {sample[1][:3]} ...", flush=True)


if __name__ == "__main__":
    main()
