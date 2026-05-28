#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import faiss
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR
from toploc_search import toploc_ivf_search  # C++ version

# ================= CONFIGURATION =================
CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc2/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc2/Datasets/conversational/CAST2019/topics"
)

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
cache_dir = CACHE_DIRS[model_name]

H_CACHED_CENTROIDS = int(os.environ.get("H_CACHED", 1024))
NP = int(os.environ.get("NP", 32))
USE_MMAP = os.environ.get("MMAP", "0") == "1"


# ================= ❌ REMOVED =================
# def dcg(scores, k): ...          ← deleted, ir_measures handles this
# def ndcg(...): ...               ← deleted, ir_measures handles this
# def mrr(...): ...                ← deleted, ir_measures handles this


# ================= QUERY ENCODER =================
def load_query_encoder(model_name):
    if model_name == "snowflake":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode(query):
            return model.encode(
                [query],
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

        return encode

    elif model_name == "dragon":
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
        model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
        model.eval()

        def encode(query):
            tokens = tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                emb = torch.nn.functional.normalize(
                    model(**tokens).last_hidden_state[:, 0, :], p=2, dim=1
                )
            return emb.cpu().numpy().astype("float32")

        return encode

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================= FLEXIBLE PARSERS =================
def split_flexible(line, expected):
    for sep in ("\t", ","):
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) == expected:
            return parts
    parts = line.split()
    if len(parts) == expected:
        return parts
    if expected == 4 and len(parts) > 4:
        return [parts[0], parts[1], " ".join(parts[2:-1]), parts[-1]]
    return None


# ================= LOAD INDEX =================
print(f"Evaluating TopLoc-IVF for: {model_name} ({index_type})")
index_path = os.path.join(cache_dir, f"{index_type}_index.index")

if USE_MMAP:
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)
ivf_index.nprobe = NP

try:
    ivf_index.make_direct_map()
except Exception:
    pass

print(
    f"Index loaded: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}"
)

H = min(H_CACHED_CENTROIDS, ivf_index.nlist)
if H != H_CACHED_CENTROIDS:
    print(f"Reducing cached centroids to nlist: {H}")

# ================= LOAD ID MAPPING =================
ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())

# ================= LOAD TOPICS =================
topics = {}
with open(os.path.join(DATASET_DIR, "topics.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = split_flexible(line, 2)
        if parts:
            topics[parts[0]] = parts[1]

if not topics:
    raise RuntimeError("Parsed 0 topics.")

# ================= LOAD QRELS =================
qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = split_flexible(line, 4)
        if not parts:
            continue
        qid, _, pid, score = parts
        try:
            score = int(score)
        except ValueError:
            continue
        if score > 0:
            qrels[qid][pid] = score

filtered_qrels = {
    k: {p: s for p, s in v.items() if p in indexed_pids}
    for k, v in qrels.items()
    if any(p in indexed_pids for p in v)
}

if not filtered_qrels:
    raise RuntimeError("No qrels survived filtering against indexed pids.")

# ================= GROUP TURNS BY CONVERSATION =================
conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)

print(f"Grouped {len(topics)} turns into {len(conversations)} conversations")

# ================= EVALUATION LOOP =================
print(f"\nLoading {model_name} query encoder...")
encode_query = load_query_encoder(model_name)
print("Running TopLoc-IVF evaluation...")

conv_cache = {}
k = 10
warmup = 5
times = []
evaluated_turns = 0

# ✅ CHANGED: run dict to collect results, computed at end like baseline
run = defaultdict(dict)

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    if q0_key not in filtered_qrels:
        continue

    # ---- TURN 0: full search + cache the top-H hot centroids ----
    q0_emb = encode_query(topics[q0_key])
    start = time.perf_counter()
    _, c0_indices = ivf_index.quantizer.search(q0_emb, H)
    conv_cache[conv_id] = c0_indices[0].astype("int64")
    scores, indices = base_index.search(q0_emb, k)
    end = time.perf_counter()

    if evaluated_turns >= warmup:
        times.append((end - start) * 1000)

    # ✅ CHANGED: build run dict instead of computing ndcg/mrr per turn
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        pid = id_map.get(str(idx))
        if pid is not None:
            run[q0_key][pid] = float(score)

    evaluated_turns += 1

    # ---- TURNS 1+: search only within the cached centroids ----
    for turn_key in turns[1:]:
        if turn_key not in filtered_qrels:
            continue

        q_emb = encode_query(topics[turn_key])
        start = time.perf_counter()
        scores, indices = toploc_ivf_search(
            ivf_index, q_emb, conv_cache[conv_id], NP, k
        )
        end = time.perf_counter()

        if evaluated_turns >= warmup:
            times.append((end - start) * 1000)

        # ✅ CHANGED: build run dict instead of computing ndcg/mrr per turn
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            pid = id_map.get(str(idx))
            if pid is not None:
                run[turn_key][pid] = float(score)

        evaluated_turns += 1

# ✅ CHANGED: compute all metrics at once at the end, same as baseline
measures = [nDCG @ 3, nDCG @ k, RR @ k]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

# ================= RESULTS =================
print("\n" + "=" * 60)
print(f"TOPLOC-IVF EVALUATION RESULTS ({index_type.upper()}, {model_name})")
print("=" * 60)
print(f"Turns evaluated:           {evaluated_turns}")
# ✅ CHANGED: print metrics from ir_measures results
print(f"NDCG@3:                    {results[nDCG @ 3]:.4f}")
print(f"NDCG@10:                   {results[nDCG @ k]:.4f}")
print(f"MRR@10:                    {results[RR @ k]:.4f}")
print(f"Avg Time:                  {np.mean(times) if times else float('nan'):.2f} ms")
print(f"Centroids cached per conv: {H}")
print(f"nprobe:                    {NP}")
print("=" * 60)
