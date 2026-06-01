"""Build a small, self-contained demo subset for the Streamlit dashboard.

Run ONCE on the server (where the full data lives):

    python3 build_demo_subset.py

It writes a tiny ``demo/`` folder (~a few MB) that the Streamlit app loads
locally — the full 38M collection is never needed on the laptop.

The subset = all judged passages (from qrels) + random distractors, filled up
to TARGET_N. Because every judged passage is included, the qrels stay fully
valid and NDCG/MRR can be computed live in the demo.

Outputs (in OUT_DIR):
  passages.parquet     id, text, embedding   (the subset)
  exact_index.index    IndexFlatIP over the subset
  ivf_index.index      IndexIVFFlat over the subset
  ids.npy              passage ids in index order (row -> pid)
  topic_embeddings.parquet  id, embedding     (precomputed query vectors)
  topics.json          turn_key -> query text
  qrels.json           qid -> {pid: score}
  meta.json            dim, nlist, counts, model
"""

import os
import json
import glob
import random
from collections import defaultdict

import numpy as np
import faiss
import pyarrow as pa
import pyarrow.parquet as pq

# ================= CONFIGURATION =================
BASE = "/home/toploc2/Datasets/conversational/CAST2019"
MODEL = "snowflake"  # which document embeddings to use for the demo

EMB_DIR = os.path.join(BASE, f"{MODEL}_embeddings")
COLLECTION_TSV = os.path.join(BASE, "CAST2019collection.tsv")
ID_MAPPING_TSV = os.path.join(BASE, "CAST2019_ID_Mapping.tsv")
TOPICS_DIR = os.path.join(BASE, "topics")
TOPICS_TSV = os.path.join(TOPICS_DIR, "topics.tsv")
QRELS_PATH = os.path.join(TOPICS_DIR, "qrels.qrel")
TOPIC_EMB_PARQUET = os.path.join(TOPICS_DIR, f"topics_{MODEL}_embeddings.parquet")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TARGET_N = 2000
SEED = 42

random.seed(SEED)


# ================= LOADERS =================
def load_collection_texts():
    """canonical passage id -> text (from CAST2019collection.tsv, header id\\ttext)."""
    texts = {}
    with open(COLLECTION_TSV, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()  # skip "id\ttext"
        if "\t" not in header:
            # No header — treat first line as data.
            pid, _, text = header.partition("\t")
            texts[pid.strip()] = text.strip()
        for line in f:
            pid, _, text = line.partition("\t")
            if pid:
                texts[pid.strip()] = text.rstrip("\n")
    print(f"Collection texts: {len(texts):,} passages")
    return texts


def load_qrels():
    """qid -> {pid: score}; comma-separated 'qid,_,pid,score' like evaluate_baseline."""
    qrels = defaultdict(dict)
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            qid, _, pid, score = (p.strip() for p in parts)
            try:
                score = int(score)
            except ValueError:
                continue
            qrels[qid][pid] = score
    judged = {pid for d in qrels.values() for pid in d}
    print(f"QRELS: {len(qrels)} turns, {len(judged):,} judged passages")
    return qrels, judged


def load_topics():
    """turn_key -> query text (comma-split, first field is the key)."""
    topics = {}
    with open(TOPICS_TSV, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                topics[parts[0].strip()] = parts[1].strip()
    print(f"Topics: {len(topics)} turns")
    return topics


def load_id_mapping():
    """Optional translator. Returns dict mapping each column value to the other.

    The mapping file format is unknown, so we read the first two tab/whitespace
    columns and index both directions; whichever side matches the collection
    ids is the canonical one.
    """
    if not os.path.exists(ID_MAPPING_TSV):
        return {}
    mapping = {}
    with open(ID_MAPPING_TSV, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").replace("\t", " ").split()
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
                mapping[a] = b
                mapping[b] = a
    print(f"ID mapping: {len(mapping):,} entries (bidirectional)")
    return mapping


# ================= ID RESOLUTION =================
def make_resolver(collection_texts):
    """Return resolve(emb_id) -> canonical collection id (or None).

    Auto-detects whether embedding ids already are collection ids; if not,
    falls back to CAST2019_ID_Mapping.tsv.
    """
    sample_files = sorted(glob.glob(os.path.join(EMB_DIR, "*.parquet")))
    if not sample_files:
        raise FileNotFoundError(f"No parquet files in {EMB_DIR}")
    sample_ids = [str(x) for x in
                  pq.read_table(sample_files[0], columns=["id"]).column("id").to_pylist()[:50]]
    direct_hits = sum(1 for x in sample_ids if x in collection_texts)
    print(f"ID check: {direct_hits}/{len(sample_ids)} embedding ids match collection directly")

    if direct_hits >= len(sample_ids) * 0.8:
        return (lambda emb_id: emb_id if emb_id in collection_texts else None), sample_files

    print("Embedding ids do not match collection directly — using ID mapping.")
    mapping = load_id_mapping()
    if not mapping:
        raise RuntimeError(
            "Embedding ids are not collection ids and no usable ID mapping was "
            f"found at {ID_MAPPING_TSV}. Please share its format.")

    def resolve(emb_id):
        if emb_id in collection_texts:
            return emb_id
        mapped = mapping.get(emb_id)
        return mapped if mapped in collection_texts else None

    mapped_hits = sum(1 for x in sample_ids if resolve(x) is not None)
    print(f"ID check via mapping: {mapped_hits}/{len(sample_ids)} resolved")
    if mapped_hits == 0:
        raise RuntimeError("ID mapping did not resolve any sample ids — aborting.")
    return resolve, sample_files


# ================= SUBSET SELECTION =================
def select_subset(collection_texts, judged, resolve, parquet_files):
    """Stream embeddings once; keep all judged passages + reservoir of distractors."""
    must = {}            # canonical_id -> embedding
    reservoir = []       # list of (canonical_id, embedding)
    seen_distractors = 0
    dim = None

    for pf in parquet_files:
        table = pq.read_table(pf)
        ids = [str(x) for x in table.column("id").to_pylist()]
        embs = table.column("embedding").to_pylist()
        for emb_id, emb in zip(ids, embs):
            canon = resolve(emb_id)
            if canon is None:
                continue
            if dim is None:
                dim = len(emb)
            if canon in judged:
                if canon not in must:
                    must[canon] = emb
            else:
                # Reservoir sampling of distractors (size TARGET_N is enough).
                seen_distractors += 1
                if len(reservoir) < TARGET_N:
                    reservoir.append((canon, emb))
                else:
                    j = random.randint(0, seen_distractors - 1)
                    if j < TARGET_N:
                        reservoir[j] = (canon, emb)
        print(f"  scanned {os.path.basename(pf)}: kept {len(must)} judged so far")

    budget = max(0, TARGET_N - len(must))
    random.shuffle(reservoir)
    distractors = reservoir[:budget]

    subset = list(must.items()) + distractors
    print(f"Subset: {len(must)} judged + {len(distractors)} distractors "
          f"= {len(subset)} passages (dim={dim})")
    return subset, dim


# ================= INDEX BUILDING =================
def build_indexes(embs, dim):
    faiss.normalize_L2(embs)  # IP == cosine, matches the paper / create_index.py

    exact = faiss.IndexFlatIP(dim)
    exact.add(embs)

    nlist = max(16, min(256, len(embs) // 16))
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(embs)
    ivf.add(embs)
    ivf.make_direct_map()
    print(f"Indexes built: exact (ntotal={exact.ntotal}), ivf (nlist={nlist})")
    return exact, ivf, nlist


# ================= MAIN =================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    collection_texts = load_collection_texts()
    qrels, judged = load_qrels()
    topics = load_topics()
    resolve, parquet_files = make_resolver(collection_texts)

    subset, dim = select_subset(collection_texts, judged, resolve, parquet_files)
    subset_ids = [pid for pid, _ in subset]
    embs = np.asarray([e for _, e in subset], dtype="float32")

    # Build & save indexes (embs is normalized in place).
    exact, ivf, nlist = build_indexes(embs, dim)
    faiss.write_index(exact, os.path.join(OUT_DIR, "exact_index.index"))
    faiss.write_index(ivf, os.path.join(OUT_DIR, "ivf_index.index"))
    np.save(os.path.join(OUT_DIR, "ids.npy"), np.array(subset_ids, dtype=object))

    # Save passages (id, text, normalized embedding) in index order.
    pa_table = pa.table({
        "id": pa.array(subset_ids, type=pa.string()),
        "text": pa.array([collection_texts[pid] for pid in subset_ids], type=pa.string()),
        "embedding": pa.array([row.tolist() for row in embs],
                              type=pa.list_(pa.float32())),
    })
    pq.write_table(pa_table, os.path.join(OUT_DIR, "passages.parquet"))

    # Precomputed topic (query) embeddings, normalized, for the known turns.
    subset_set = set(subset_ids)
    if os.path.exists(TOPIC_EMB_PARQUET):
        t = pq.read_table(TOPIC_EMB_PARQUET)
        cols = t.column_names
        id_col = next((c for c in ("id", "qid", "turn_id", "topic_id") if c in cols), None)
        emb_col = next((c for c in ("embedding", "embeddings", "vector") if c in cols), None)
        if id_col and emb_col:
            tids = [str(x) for x in t.column(id_col).to_pylist()]
            tembs = np.asarray(t.column(emb_col).to_pylist(), dtype="float32")
            faiss.normalize_L2(tembs)
            pq.write_table(pa.table({
                "id": pa.array(tids, type=pa.string()),
                "embedding": pa.array([r.tolist() for r in tembs],
                                      type=pa.list_(pa.float32())),
            }), os.path.join(OUT_DIR, "topic_embeddings.parquet"))
            print(f"Topic embeddings saved: {len(tids)} turns")
        else:
            print(f"WARN: topic embedding columns not found ({cols}); skipping.")
    else:
        print(f"WARN: {TOPIC_EMB_PARQUET} not found; demo will encode queries live.")

    # Filter qrels to the subset (relevant passages are all present, but a few
    # judged-nonrelevant ones may be distractor-excluded — keep what's in).
    qrels_out = {qid: {pid: s for pid, s in d.items() if pid in subset_set}
                 for qid, d in qrels.items()}
    qrels_out = {qid: d for qid, d in qrels_out.items() if d}

    with open(os.path.join(OUT_DIR, "qrels.json"), "w", encoding="utf-8") as f:
        json.dump(qrels_out, f)
    with open(os.path.join(OUT_DIR, "topics.json"), "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "dim": dim, "nlist": nlist,
                   "n_passages": len(subset_ids), "n_judged": len(judged),
                   "target_n": TARGET_N}, f)

    print(f"\nDone. Demo subset written to {OUT_DIR}/")
    print("Copy that folder to the laptop and run:  streamlit run demo_app.py")


if __name__ == "__main__":
    main()
