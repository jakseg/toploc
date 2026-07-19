#!/usr/bin/env python3
"""Encode the DOCUMENT collection with the paper's document encoder -> parquet
(id, embedding) — the paper-1 passage embeddings (reused as I_D by QLR too).

Reproduces, conceptually, the embeddings our supervisor handed us: it streams the
full CAsT2019 collection (38,636,446 passages: MS MARCO v1 + TREC CAR) and writes
sharded parquet with columns `id`, `embedding` — exactly the format
create_index.py consumes (get_parquet_info / read_parquet). Streaming + sharding
keep memory flat, so the same command scales from a --limit smoke test to the
full collection.

Encoding (paper toploc.pdf §Models, footnotes 2-3):
  snowflake  Snowflake/snowflake-arctic-embed-l-v2.0, NO query prompt for
             documents, L2-normalised (cosine), 1024-d.
  dragon     facebook/dragon-plus-CONTEXT-encoder (queries use the query encoder),
             CLS token, 768-d. dragon-plus is trained for RAW dot product "without
             a mandatory normalization step", so documents are stored UN-normalised
             by default (norm ~65) — create_index.py L2-normalises at build time for
             the HNSW/IVF graph, so the index & metrics are identical either way.
             Pass --normalize to store cosine-normalised dragon instead.

Examples
--------
# smoke test (2k passages, one shard)
python create_document_embeddings.py snowflake \
  --input dataset/head_2000_rows.tsv --out-dir data/snowflake --limit 2000

# full collection on the cluster
python create_document_embeddings.py dragon \
  --input  /home/toploc2/Datasets/conversational/CAST2019/CAST2019collection.tsv \
  --out-dir /home/toploc2/Datasets/conversational/CAST2019/dragon_embeddings
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DIMS = {"snowflake": 1024, "dragon": 768}
DEFAULT_BATCH = {"snowflake": 256, "dragon": 64}
# dragon-plus is dot-product (store raw); snowflake is cosine (store normalised).
DEFAULT_NORMALIZE = {"snowflake": True, "dragon": False}


def build_doc_encoder(model_name: str, normalize: bool, batch_size: int):
    """f(list[str]) -> (N, dim) float32, using the DOCUMENT encoder."""
    if model_name == "snowflake":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode(texts):
            return model.encode(
                list(texts),
                # documents get NO query prompt (that is the query-side difference)
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
            ).astype("float32")

        return encode

    if model_name == "dragon":
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-context-encoder")
        model = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")
        model.eval()

        def encode(texts):
            outs = []
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                tok = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
                with torch.no_grad():
                    emb = model(**tok).last_hidden_state[:, 0, :]   # CLS token
                    if normalize:
                        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode

    raise ValueError(f"Unknown model: {model_name}")


def iter_passages(path: str, sep: str):
    """Yield (id, text) from the collection tsv; tolerates an 'id<SEP>text' header."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            pid, _, text = line.partition(sep)
            if not text:
                continue
            if pid == "id":            # tolerate a header row
                continue
            yield pid, text


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
    ap.add_argument("--input", required=True, help="collection tsv 'id<SEP>text'")
    ap.add_argument("--sep", default="\t", help="tsv separator (default TAB)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default=None, help="output basename (default <model>_doc)")
    ap.add_argument("--shard-size", type=int, default=100_000,
                    help="rows per parquet shard (0 = single file)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="encode batch (default 256 snowflake / 64 dragon)")
    ap.add_argument("--limit", type=int, default=0, help="cap #passages (0 = all)")
    norm = ap.add_mutually_exclusive_group()
    norm.add_argument("--normalize", dest="normalize", action="store_true")
    norm.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=None)
    args = ap.parse_args()

    normalize = DEFAULT_NORMALIZE[args.model] if args.normalize is None else args.normalize
    batch_size = args.batch_size or DEFAULT_BATCH[args.model]
    prefix = args.prefix or f"{args.model}_doc"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[{args.model}] encoding docs from {args.input} -> {args.out_dir} "
          f"(normalize={normalize}, batch={batch_size})", flush=True)
    encode = build_doc_encoder(args.model, normalize, batch_size)

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
    for pid, text in iter_passages(args.input, args.sep):
        ids_buf.append(pid)
        txt_buf.append(text)
        if args.limit and (n_total + len(ids_buf)) >= args.limit:
            flush()
            break
        if len(ids_buf) >= shard_size:
            flush()
    flush()

    print(f"Done: {n_total} document embeddings in {len(written)} shard(s) under {args.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
