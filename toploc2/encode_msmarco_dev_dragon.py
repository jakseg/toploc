#!/usr/bin/env python3
"""Encode the msmarco dev queries with the DRAGON query encoder and write them
as (id, embedding) parquet — the dragon counterpart of the precomputed snowflake
dev_query embeddings, so `--dataset msmarco-on-cast` can also run on dragon.

Only snowflake dev embeddings (1024-d) exist on the server; dragon's 768-d set
must be produced from the dev-query TEXT. To keep the eval set identical across
encoders (apples-to-apples), we encode exactly the query ids that the snowflake
dev set uses, look their text up in dev_queries.jsonl, and write the result into
MSMARCO_DEV_QUERY_DIRS["dragon"].

Run on the cluster (needs the dragon encoder weights + the text file):
    python encode_msmarco_dev_dragon.py
"""
import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss

from toploc2_hnsw_pure_python import load_query_encoder, MSMARCO_DEV_QUERY_DIRS

DEV_JSONL = os.environ.get(
    "DEV_JSONL",
    "/home/toploc1/Datasets/conversational/CAST2019/msmarco/msmarco_queries/dev_queries.jsonl",
)
SNOWFLAKE_DEV_DIR = MSMARCO_DEV_QUERY_DIRS["snowflake"]
OUT_DIR = MSMARCO_DEV_QUERY_DIRS["dragon"]


def read_parquet_ids(emb_dir):
    """Return the query ids (in file order) of every parquet shard in emb_dir."""
    files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {emb_dir}")
    ids = []
    for f in files:
        ids.extend(str(x) for x in pq.read_table(f, columns=["id"]).column("id").to_pylist())
    return ids


def load_jsonl_text(path):
    """dev_queries.jsonl -> {qid: text}. Field names are auto-detected."""
    id2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id") or obj.get("qid") or obj.get("_id") or obj.get("query_id")
            txt = obj.get("text") or obj.get("query") or obj.get("contents") or obj.get("title")
            if qid is None or txt is None:
                raise KeyError(f"Unexpected jsonl schema {list(obj)}; adjust field names.")
            id2text[str(qid)] = str(txt)
    return id2text


def main():
    sf_ids = read_parquet_ids(SNOWFLAKE_DEV_DIR)
    id2text = load_jsonl_text(DEV_JSONL)
    print(f"snowflake dev ids: {len(sf_ids):,} | jsonl texts: {len(id2text):,}", flush=True)

    keep = [qid for qid in sf_ids if qid in id2text]
    missing = len(sf_ids) - len(keep)
    if missing:
        print(f"  WARN: {missing} snowflake dev ids have no text in the jsonl — skipped.", flush=True)
    if not keep:
        raise RuntimeError("No dev ids matched between the snowflake set and the jsonl.")
    texts = [id2text[qid] for qid in keep]
    print(f"Encoding {len(keep):,} dev queries with dragon | e.g. {keep[0]} -> {texts[0][:60]!r}", flush=True)

    encode_batch = load_query_encoder("dragon")
    emb = encode_batch(texts).astype("float32")   # (N, 768), encoder L2-normalises
    faiss.normalize_L2(emb)                        # idempotent safety
    assert emb.shape == (len(keep), 768), emb.shape

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "dev_dragon.part00000.parquet")
    table = pa.table({
        "id": pa.array([str(x) for x in keep], type=pa.string()),
        "embedding": pa.array([row.tolist() for row in emb], type=pa.list_(pa.float32())),
    })
    pq.write_table(table, out_path)
    print(f"Wrote {emb.shape} to {out_path}", flush=True)


if __name__ == "__main__":
    main()
