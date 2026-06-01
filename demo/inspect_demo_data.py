"""One-off inspection helper for the demo data.

Run on the server (where the data lives) and paste the output:

    python3 inspect_demo_data.py > out.txt

It prints, in a copy-pasteable way:
  - what's in the CAST2019 directory
  - the first rows of the text collection + ID mapping (text files)
  - sample IDs + embedding dim of every *.parquet embedding folder it finds
  - whether the embedding IDs match the collection IDs (MARCO_x)
"""

import os
import glob

import pyarrow.parquet as pq

BASE = "/home/toploc2/Datasets/conversational/CAST2019"

# Candidate folders that might hold document embeddings.
EMB_DIR_CANDIDATES = [
    "snowflake_embeddings",
    "dragon_embeddings",
    "star_embeddings",
]
TEXT_FILES = ["CAST2019collection.tsv", "CAST2019_ID_Mapping.tsv"]


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


rule(f"Directory listing: {BASE}")
try:
    for name in sorted(os.listdir(BASE)):
        full = os.path.join(BASE, name)
        kind = "dir " if os.path.isdir(full) else "file"
        print(f"  [{kind}] {name}")
except OSError as e:
    print(f"  ERROR: {e}")


for fname in TEXT_FILES:
    path = os.path.join(BASE, fname)
    rule(f"First 3 lines: {fname}")
    if not os.path.exists(path):
        print("  (not found)")
        continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.rstrip("\n").split("\t")
            preview = [p[:80] for p in parts]
            print(f"  line {i}: {len(parts)} tab-cols -> {preview}")


collection_ids = set()
coll_path = os.path.join(BASE, "CAST2019collection.tsv")
if os.path.exists(coll_path):
    with open(coll_path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)  # skip header
        for i, line in enumerate(f):
            if i >= 5000:
                break
            collection_ids.add(line.split("\t", 1)[0])


for sub in EMB_DIR_CANDIDATES:
    emb_dir = os.path.join(BASE, sub)
    rule(f"Embedding folder: {sub}")
    if not os.path.isdir(emb_dir):
        print("  (folder does not exist)")
        continue
    files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")))
    print(f"  {len(files)} parquet files")
    if not files:
        continue
    table = pq.read_table(files[0])
    print(f"  columns: {table.column_names}")
    if "id" in table.column_names:
        sample_ids = [str(x) for x in table.column("id").to_pylist()[:5]]
        print(f"  sample ids: {sample_ids}")
        if collection_ids:
            hits = sum(1 for x in sample_ids if x in collection_ids)
            print(f"  ids match collection (MARCO_x)? {hits}/{len(sample_ids)} of the sample")
    if "embedding" in table.column_names:
        dim = len(table.column("embedding")[0].as_py())
        print(f"  embedding dim: {dim}")
