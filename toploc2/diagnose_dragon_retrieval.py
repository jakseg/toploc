#!/usr/bin/env python3
"""Why is dragon NDCG/MRR/Accuracy@10 ~0 for --dataset msmarco-on-cast, while
routing looks sane? Three independent checks, cheapest first:

  1. qrels / dev / groundtruth coverage (no big index load) - is N (eval
     queries with qrels) actually healthy, or did coverage silently collapse?
  2. hnsw_ids.npy vs index.ntotal + mtimes - catches a STALE ids file left
     over from before the dragon dot-product rebuild (routing never touches
     pids, so this kind of mismatch would not show up there at all, only in
     NDCG/MRR/Accuracy - a "wrong label on a right vector" bug).
  3. real retrieval spot check - run actual HNSW search for a handful of dev
     queries that DO have qrels, and check whether ANY relevant pid shows up
     in the top-100. If routing works but the run dict never contains a
     relevant pid, the problem is in query/doc embedding alignment or the
     id_map, not in QLR itself.

Run from inside toploc2/ (same directory as toploc2_hnsw_pure_python.py):
    MMAP=1 python diagnose_dragon_retrieval.py
"""
import os

import numpy as np
import faiss

import toploc2_hnsw_pure_python as qlr

MODEL = "dragon"
DATASET = "msmarco-on-cast"
K = 10

cache_dir = qlr.CACHE_DIRS[MODEL]
index_path = os.path.join(cache_dir, "hnsw_index.index")
ids_path = os.path.join(cache_dir, "hnsw_ids.npy")

print("=== 1. qrels / dev / groundtruth coverage ===", flush=True)
qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
id_array = np.load(ids_path, allow_pickle=True)
indexed_pids = set(str(x) for x in id_array)

filtered_qrels = {}
for qid, pid_scores in qrels.items():
    valid = {p: s for p, s in pid_scores.items() if p in indexed_pids}
    if valid:
        filtered_qrels[qid] = valid
print(f"qrels queries total: {len(qrels):,} | with >=1 indexed relevant pid: "
      f"{len(filtered_qrels):,}", flush=True)

normalize = MODEL != "dragon"
dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=normalize)
eval_keys = [qid for qid in q_ids if qid in filtered_qrels]
print(f"dragon dev queries: {len(q_ids):,} | with qrels (N): {len(eval_keys):,}",
      flush=True)

sf_dir = qlr.MSMARCO_DEV_QUERY_DIRS["snowflake"]
sf_ids, _ = qlr.load_parquet_embeddings(sf_dir, normalize=True)
if set(q_ids) == set(sf_ids):
    print(f"parity vs snowflake dev ids: IDENTICAL ({len(set(q_ids)):,} ids)", flush=True)
else:
    print(f"parity vs snowflake dev ids: DIFFERS "
          f"(dragon-only={len(set(q_ids) - set(sf_ids))}, "
          f"snowflake-only={len(set(sf_ids) - set(q_ids))})", flush=True)

gt_path = qlr.groundtruth_path(cache_dir, MODEL, DATASET, K)
gt = qlr.load_groundtruth(gt_path)
if gt is None:
    print(f"groundtruth file MISSING: {gt_path}", flush=True)
else:
    n_gt = sum(1 for k in eval_keys if gt.get(k))
    print(f"groundtruth entries covering eval set: {n_gt:,}/{len(eval_keys):,}  "
          f"({gt_path})", flush=True)

print("\n=== 2. hnsw_ids.npy vs index.ntotal (stale-rebuild check) ===", flush=True)
index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
ok = index.ntotal == len(id_array)
print(f"index.ntotal = {index.ntotal:,} | len(hnsw_ids.npy) = {len(id_array):,} "
      f"-> {'OK' if ok else 'MISMATCH!!'}", flush=True)
print(f"index file mtime: {os.path.getmtime(index_path)} "
      f"({os.path.getctime(index_path)})", flush=True)
print(f"ids file mtime:   {os.path.getmtime(ids_path)} "
      f"({os.path.getctime(ids_path)})", flush=True)
print("(if the ids file is much OLDER than the index, it may be a stale copy "
      "from before the dragon dot-product rebuild -> every retrieved internal "
      "node id would map to the WRONG pid string.)", flush=True)

print("\n=== 3. real retrieval spot check (does HNSW ever hit a qrels pid?) ===",
      flush=True)
id_map = {i: str(pid) for i, pid in enumerate(id_array)}
row_of = {qid: i for i, qid in enumerate(q_ids)}
sample = eval_keys[:15]
qv = np.ascontiguousarray(q_emb[[row_of[k] for k in sample]], dtype="float32")
index.hnsw.efSearch = 200
scores, idxs = index.search(qv, 100)

hits = 0
for row, qid in enumerate(sample):
    rel_pids = set(filtered_qrels[qid].keys())
    ret_pids = [id_map[i] for i in idxs[row] if i != -1]
    overlap = rel_pids & set(ret_pids)
    if overlap:
        hits += 1
    print(f"  qid={qid:>10}  relevant(sample)={list(rel_pids)[:3]}  "
          f"top3_retrieved={ret_pids[:3]}  top3_scores={scores[row][:3].round(3).tolist()}  "
          f"hit_in_top100={bool(overlap)}", flush=True)

print(f"\n{hits}/{len(sample)} sample queries had >=1 relevant pid in top-100.",
      flush=True)
print("Expect most to hit (like the snowflake run). 0/15 means the dragon dev "
      "query embeddings and the dragon document index genuinely don't align "
      "(wrong encoder call, wrong normalization baked into the file, or the "
      "index/ids pairing itself is stale) - not a QLR-specific issue.",
      flush=True)
