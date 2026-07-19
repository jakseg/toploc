#!/usr/bin/env python3
"""Verify that OUR embedding-creation logic reproduces the supervisor's stored
embeddings (a paper-conformance proof).

We received the finished embeddings from the supervisor and never ran the
encoders ourselves. This script proves the encoding logic in
`create_embeddings_{snowflake,dragon}.py` (documents) and `load_query_encoder`
(queries) is correct: it re-encodes a small SAMPLE of ids with exactly that
logic and compares to the stored vectors, per id.

Per sampled id it reports:
  - cosine(ours, theirs)  -> ~1.0  ==> the encoder matches (THE conformance test)
  - ||ours|| , ||theirs|| -> norm comparison. NB dragon documents are stored
                             RAW (norm ~65, dragon-plus is dot-product); our
                             script L2-normalises (norm 1.0). So the raw values
                             differ but cosine is ~1.0 — that is expected and
                             fine (create_index.py L2-normalises anyway).

Text sources (the raw text that was encoded):
  passages : the collection TSV, "id<TAB>text"   --text-format tsv
  queries  : a JSONL of {id,text}                 --text-format jsonl

Runs on the cluster (text + parquet both live there), or locally if you copy a
handful of parquet rows + the matching text lines down.

Examples
--------
# snowflake passages (document encoder, no query prompt)
python verify_embeddings.py snowflake --mode passage \
  --emb-dir   /home/toploc2/Datasets/conversational/CAST2019/snowflake_embeddings \
  --text-file /home/toploc2/Datasets/conversational/CAST2019/CAST2019collection.tsv \
  --text-format tsv --n 20

# dragon passages (context encoder, CLS, stored RAW -> compare via cosine)
python verify_embeddings.py dragon --mode passage \
  --emb-dir   /home/toploc2/Datasets/conversational/CAST2019/dragon_embeddings \
  --text-file /home/toploc2/Datasets/conversational/CAST2019/CAST2019collection.tsv \
  --text-format tsv --n 20

# snowflake dev queries (query encoder, prompt_name="query")
python verify_embeddings.py snowflake --mode query \
  --emb-dir   /home/toploc2/Datasets/conversational/CAST2019/msmarco/msmarco_embeddings/dev_query \
  --text-file /home/toploc2/Datasets/conversational/CAST2019/msmarco/msmarco_queries/dev_queries.jsonl \
  --text-format jsonl --n 20
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


# ----------------------------------------------------------------------------
# Encoders — MUST stay identical to create_embeddings_*.py / load_query_encoder.
# ----------------------------------------------------------------------------
def build_encoder(model_name: str, is_query: bool):
    """Return f(list[str]) -> (N, dim) float32, matching the project encoders."""
    if model_name == "snowflake":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode(texts):
            kwargs = dict(
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            )
            if is_query:
                # queries get the "query" prompt; documents get NO prompt
                kwargs["prompt_name"] = "query"
            return model.encode(list(texts), **kwargs).astype("float32")

        return encode

    if model_name == "dragon":
        import torch
        from transformers import AutoModel, AutoTokenizer

        repo = (
            "facebook/dragon-plus-query-encoder"
            if is_query
            else "facebook/dragon-plus-context-encoder"
        )
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo)
        model.eval()

        def encode(texts, chunk=32):
            outs = []
            for i in range(0, len(texts), chunk):
                batch = list(texts[i : i + chunk])
                tok = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                with torch.no_grad():
                    emb = model(**tok).last_hidden_state[:, 0, :]  # CLS token
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode

    raise ValueError(f"Unknown model: {model_name}")


# ----------------------------------------------------------------------------
def read_sample_from_parquet(emb_dir: str, n: int):
    """First n (id, embedding) rows across the sorted parquet shards."""
    files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards in {emb_dir}")
    ids, embs = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["id", "embedding"])
            ids.extend(str(x) for x in t.column("id").to_pylist())
            embs.extend(t.column("embedding").to_pylist())
            if len(ids) >= n:
                break
        if len(ids) >= n:
            break
    embs = np.asarray(embs[:n], dtype=np.float32)
    return ids[:n], embs


def load_texts(ids, path: str, fmt: str):
    """Return {id: text} for the wanted ids; scans the file, stops when complete."""
    want = set(ids)
    found = {}
    with open(path, "r", encoding="utf-8") as f:
        if fmt == "tsv":
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                pid = parts[0]
                if pid in want and pid not in found:
                    found[pid] = parts[1]
                    if len(found) == len(want):
                        break
        elif fmt == "jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                qid = o.get("id") or o.get("qid") or o.get("_id") or o.get("query_id")
                txt = o.get("text") or o.get("query") or o.get("contents") or o.get("title")
                if qid is None:
                    continue
                qid = str(qid)
                if qid in want and qid not in found:
                    found[qid] = str(txt)
                    if len(found) == len(want):
                        break
        else:
            raise ValueError(f"unknown text-format {fmt}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["snowflake", "dragon"])
    ap.add_argument("--mode", choices=["passage", "query"], required=True,
                    help="passage -> document encoder; query -> query encoder")
    ap.add_argument("--emb-dir", required=True, help="supervisor parquet dir (id, embedding)")
    ap.add_argument("--text-file", required=True, help="raw text source that was encoded")
    ap.add_argument("--text-format", choices=["tsv", "jsonl"], required=True)
    ap.add_argument("--n", type=int, default=20, help="sample size")
    ap.add_argument("--tol", type=float, default=0.999,
                    help="min cosine to count an id as matching")
    args = ap.parse_args()

    is_query = args.mode == "query"
    print(f"[{args.model}/{args.mode}] sampling {args.n} ids from {args.emb_dir}", flush=True)
    sample_ids, sample_emb = read_sample_from_parquet(args.emb_dir, args.n)
    print(f"  got {len(sample_ids)} stored embeddings, dim={sample_emb.shape[1]}", flush=True)
    stored = {i: sample_emb[p] for p, i in enumerate(sample_ids)}

    id2text = load_texts(sample_ids, args.text_file, args.text_format)
    ids = [i for i in sample_ids if i in id2text]
    if not ids:
        raise RuntimeError("No sampled id had matching text — check --text-file / ids.")
    if len(ids) < len(sample_ids):
        print(f"  WARN: {len(sample_ids) - len(ids)} sampled ids had no text — skipped.", flush=True)
    texts = [id2text[i] for i in ids]
    theirs = np.asarray([stored[i] for i in ids], dtype=np.float32)

    print(f"  encoding {len(ids)} texts with our {args.model} "
          f"{'query' if is_query else 'document'} encoder...", flush=True)
    encode = build_encoder(args.model, is_query)
    ours = encode(texts)

    def unit(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    cos = np.sum(unit(ours) * unit(theirs), axis=1)
    on = np.linalg.norm(ours, axis=1)
    tn = np.linalg.norm(theirs, axis=1)

    print("\n  id                         cosine   ||ours||  ||theirs||")
    print("  " + "-" * 58)
    for i, c, a, b in zip(ids, cos, on, tn):
        flag = "" if c >= args.tol else "  <-- MISMATCH"
        print(f"  {i:<24} {c:8.5f}  {a:8.3f}  {b:9.3f}{flag}")

    n_ok = int(np.sum(cos >= args.tol))
    print("\n  " + "=" * 58)
    print(f"  matching (cos >= {args.tol}): {n_ok}/{len(ids)}")
    print(f"  cosine  min={cos.min():.5f}  mean={cos.mean():.5f}  max={cos.max():.5f}")
    print(f"  norms   ours mean={on.mean():.3f}   theirs mean={tn.mean():.3f}")
    if n_ok == len(ids):
        print("  VERDICT: our encoding reproduces the supervisor embeddings (cosine-identical).")
    else:
        print("  VERDICT: MISMATCH — encoding differs (model / pooling / prompt / normalize).")


if __name__ == "__main__":
    main()
