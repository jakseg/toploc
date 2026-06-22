#!/usr/bin/env python3
"""Smoke test for the msmarco-on-cast QLR setup — checks the DATA lines up the
way the eval needs it, WITHOUT loading the 169 GB HNSW index.

It loads only:
  - hnsw_ids.npy (the CAST index's pid list, MARCO_/CAR_ space),
  - qrels.dev.small (MARCO_-prefixed, as the eval does),
  - the dev-query ids + dim (parquet metadata),
  - the first train-log shard's dim.

and answers the make-or-break questions:
  1. do the MARCO_-mapped qrels pids actually exist in the index?  (~100% expected)
  2. how many dev queries will be scored (have qrels)?
  3. do dev / log / index embedding dims agree?

    python smoke_test_msmarco_on_cast.py snowflake
    python smoke_test_msmarco_on_cast.py dragon     # after encode_msmarco_dev_dragon.py
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

from toploc2_hnsw_pure_python import (
    CACHE_DIRS, MSMARCO_QRELS, MSMARCO_DEV_QUERY_DIRS, MSMARCO_FULL_LOG_DIRS,
    load_qrels,
)

EXPECTED_DIM = {"snowflake": 1024, "dragon": 768}


def parquet_ids_and_dim(emb_dir):
    """Return (ids, dim) reading only the id column (+ dim from the first shard)."""
    files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {emb_dir}")
    dim = len(pq.read_table(files[0], columns=["embedding"]).column("embedding")[0].as_py())
    ids = []
    for f in files:
        ids.extend(str(x) for x in pq.read_table(f, columns=["id"]).column("id").to_pylist())
    return ids, dim


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
    print(f"=== msmarco-on-cast smoke test [{model}] ===\n", flush=True)

    # 1. CAST index id space (no big index load — just the ids npy).
    ids_path = os.path.join(CACHE_DIRS[model], "hnsw_ids.npy")
    print(f"[index] loading id list {ids_path} (this takes a moment)...", flush=True)
    indexed = set(str(x) for x in np.load(ids_path, allow_pickle=True))
    n_marco = sum(1 for x in indexed if x.startswith("MARCO_"))
    print(f"  {len(indexed):,} pids | MARCO_={n_marco:,} | CAR_={len(indexed) - n_marco:,}")
    print(f"  sample: {list(indexed)[:3]}\n")

    # 2. qrels coverage with the MARCO_ prefix (exactly what the eval does).
    qrels = load_qrels(MSMARCO_QRELS, pid_prefix="MARCO_")
    all_pids = {p for d in qrels.values() for p in d}
    present_pids = sum(1 for p in all_pids if p in indexed)
    covered_q = sum(1 for d in qrels.values() if any(p in indexed for p in d))
    print(f"[qrels] {len(qrels):,} queries, {len(all_pids):,} unique pids (MARCO_-mapped)")
    print(f"  pids present in index:        {present_pids:,}/{len(all_pids):,} ({present_pids / len(all_pids):.1%})")
    print(f"  queries with >=1 indexed pid: {covered_q:,}/{len(qrels):,}\n")

    # 3. dev queries (test set): ids + dim, and overlap with qrels = eval set size.
    dev_dir = MSMARCO_DEV_QUERY_DIRS[model]
    if not glob.glob(os.path.join(dev_dir, "*.parquet")):
        print(f"[dev] MISSING: no parquet in {dev_dir}")
        if model == "dragon":
            print("      -> run encode_msmarco_dev_dragon.py first.\n")
        dev_ids, dev_dim = [], None
    else:
        dev_ids, dev_dim = parquet_ids_and_dim(dev_dir)
        eval_n = sum(1 for q in dev_ids if q in qrels)
        print(f"[dev] {len(dev_ids):,} queries, dim {dev_dim} | {eval_n:,} have qrels -> EVAL SET\n")

    # 4. train log (Q_L): shard count + first-shard dim (don't load the full log).
    log_dir = MSMARCO_FULL_LOG_DIRS[model]
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.parquet")))
    if not log_files:
        print(f"[log] MISSING: no parquet in {log_dir}")
        log_dim = None
    else:
        t0 = pq.read_table(log_files[0], columns=["embedding"])
        log_dim = len(t0.column("embedding")[0].as_py())
        print(f"[log] {len(log_files)} shards in {log_dir} | first-shard dim {log_dim}, rows {t0.num_rows:,}\n")

    # 5. verdict.
    exp = EXPECTED_DIM.get(model)
    dims_ok = dev_dim == exp and log_dim == exp
    pids_ok = present_pids / max(len(all_pids), 1) > 0.5
    ready = bool(dev_ids) and dims_ok and pids_ok
    print("=" * 60)
    print(f"dims (expect {exp}): dev={dev_dim} log={log_dim} -> {'OK' if dims_ok else 'MISMATCH'}")
    print(f"qrels coverage > 50%: {'OK' if pids_ok else 'TOO LOW (mapping wrong?)'}")
    print(f"VERDICT: {'READY ✅' if ready else 'NOT READY — see above ❌'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
