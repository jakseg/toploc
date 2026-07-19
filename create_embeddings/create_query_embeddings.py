#!/usr/bin/env python3
"""Encode QUERIES with the paper's query encoder -> parquet (id, embedding).

This is the piece the two document scripts (create_embeddings_{snowflake,dragon}.py)
do NOT cover, and the only extra embeddings QLR (paper 2) needs: QLR reuses the
document index (I_D) from paper 1 and adds a query-log index I_Q, so all it needs
on top is QUERY vectors. It serves three query sets, all with the same encoder:

  * QLR historical log  Q_L   (msmarco train queries, ~808k)
  * QLR test queries          (msmarco dev.small)
  * paper-1 eval topics       (CAST2019/2020 rewritten utterances)

Query encoding differs from document encoding by exactly one thing per model:
  * snowflake -> add the "query" prompt (prompt_name="query"); documents get none.
  * dragon    -> use dragon-plus-QUERY-encoder (docs use the CONTEXT encoder);
                 CLS token, L2-normalise (same as the document script).
This matches `load_query_encoder` in combine_hnsw.py / the QLR driver verbatim.

Input: a text file of id + query, either
  tsv   : "<id><SEP><query>" per line (SEP configurable; CAST topics use ",")
  jsonl : {"id"/"qid"/"_id"/"query_id": ..., "text"/"query"/...: ...} per line
Output: parquet shards "<prefix>.partNNNNN.parquet" with columns id, embedding —
the format create_index.py / the QLR driver / combine_hnsw.py all consume.

Examples
--------
# QLR test queries (snowflake dev.small), jsonl source
python create_query_embeddings.py snowflake \
  --input  /home/toploc2/Datasets/conversational/CAST2019/msmarco/msmarco_queries/dev_queries.jsonl \
  --input-format jsonl \
  --out-dir /home/toploc2/Datasets/conversational/CAST2019/msmarco/msmarco_embeddings/dev_query

# QLR historical log Q_L (dragon train queries), tsv "qid<TAB>text"
python create_query_embeddings.py dragon \
  --input /home/toploc2/Datasets/.../queries.train.tsv --input-format tsv --sep $'\t' \
  --out-dir /home/toploc2/Datasets/conversational/msmarco/dragon

# paper-1 CAST2020 topics ("turn_id,query"), one parquet
python create_query_embeddings.py snowflake \
  --input .../CAST2020/topics/topics.tsv --input-format tsv --sep ',' \
  --out-dir .../CAST2020/topics --prefix topics_snowflake_embeddings --shard-size 0
"""
import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DIMS = {"snowflake": 1024, "dragon": 768}


def build_query_encoder(model_name: str):
    """f(list[str]) -> (N, dim) float32. Identical to load_query_encoder."""
    if model_name == "snowflake":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode(texts):
            return model.encode(
                list(texts),
                prompt_name="query",          # queries get the query prompt
                normalize_embeddings=True,     # cosine
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            ).astype("float32")

        return encode

    if model_name == "dragon":
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
        model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
        model.eval()

        def encode(texts, chunk=32):
            outs = []
            for i in range(0, len(texts), chunk):
                batch = list(texts[i : i + chunk])
                tok = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
                with torch.no_grad():
                    emb = model(**tok).last_hidden_state[:, 0, :]   # CLS token
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # cosine
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode

    raise ValueError(f"Unknown model: {model_name}")


def iter_queries(path: str, fmt: str, sep: str):
    """Yield (id, text) from a tsv/jsonl query file."""
    with open(path, "r", encoding="utf-8") as f:
        if fmt == "tsv":
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                qid, _, text = line.partition(sep)
                if not text:                      # malformed / no separator -> skip
                    continue
                if qid in ("id", "qid", "turn_id", "topic_id", "tid"):  # tolerate a header
                    continue
                yield qid, text
        elif fmt == "jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                qid = o.get("id") or o.get("qid") or o.get("_id") or o.get("query_id")
                text = o.get("text") or o.get("query") or o.get("contents") or o.get("title")
                if qid is None or text is None:
                    raise KeyError(f"Unexpected jsonl schema {list(o)}; adjust field names.")
                yield str(qid), str(text)
        else:
            raise ValueError(f"unknown input-format {fmt}")


def write_shard(out_dir, prefix, shard_idx, ids, emb):
    path = os.path.join(out_dir, f"{prefix}.part{shard_idx:05d}.parquet")
    table = pa.table({
        "id": pa.array([str(x) for x in ids], type=pa.string()),
        "embedding": pa.array([row.tolist() for row in emb], type=pa.list_(pa.float32())),
    })
    pq.write_table(table, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["snowflake", "dragon"])
    ap.add_argument("--input", required=True, help="query text file (tsv/jsonl)")
    ap.add_argument("--input-format", choices=["tsv", "jsonl"], required=True)
    ap.add_argument("--sep", default="\t", help="tsv separator (CAST topics use ',')")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default=None,
                    help="output basename (default: <model>_query)")
    ap.add_argument("--shard-size", type=int, default=100_000,
                    help="rows per parquet shard (0 = single file)")
    ap.add_argument("--limit", type=int, default=0, help="cap #queries (0 = all)")
    args = ap.parse_args()

    prefix = args.prefix or f"{args.model}_query"
    os.makedirs(args.out_dir, exist_ok=True)
    encode = build_query_encoder(args.model)
    print(f"[{args.model}] encoding queries from {args.input} -> {args.out_dir}", flush=True)

    ids_buf, txt_buf = [], []
    shard_idx, n_total = 0, 0
    written = []

    def flush():
        nonlocal shard_idx, n_total, ids_buf, txt_buf
        if not ids_buf:
            return
        emb = encode(txt_buf)
        assert emb.shape[1] == DIMS[args.model], (emb.shape, DIMS[args.model])
        p = write_shard(args.out_dir, prefix, shard_idx, ids_buf, emb)
        written.append(p)
        n_total += len(ids_buf)
        print(f"  shard {shard_idx}: +{len(ids_buf)} -> {n_total} total ({os.path.basename(p)})",
              flush=True)
        shard_idx += 1
        ids_buf, txt_buf = [], []

    shard_size = args.shard_size if args.shard_size > 0 else float("inf")
    for qid, text in iter_queries(args.input, args.input_format, args.sep):
        ids_buf.append(qid)
        txt_buf.append(text)
        if args.limit and (n_total + len(ids_buf)) >= args.limit:
            flush()
            break
        if len(ids_buf) >= shard_size:
            flush()
    flush()

    print(f"Done: {n_total} query embeddings in {len(written)} shard(s) under {args.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
