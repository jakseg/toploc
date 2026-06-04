#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import faiss
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR
from toploc_search import toploc_ivf_search, toploc_ivf_search_ptr  # C++ version

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
# Match baseline nprobe default
NP = int(os.environ.get("NP", 128))
USE_MMAP = os.environ.get("MMAP", "0") == "1"


# ================= QUERY ENCODER (Batched) =================
def load_query_encoder(model_name):
    if model_name == "snowflake":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode_batch(queries):
            return model.encode(
                queries,
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

        return encode_batch

    elif model_name == "dragon":
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
        model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
        model.eval()

        def encode_batch(queries, chunk=32):
            outs = []
            for i in range(0, len(queries), chunk):
                batch = queries[i : i + chunk]
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = model(**tokens)
                    emb = torch.nn.functional.normalize(
                        outputs.last_hidden_state[:, 0, :], p=2, dim=1
                    )
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch

    else:
        raise ValueError(f"Unknown model: {model_name}")


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
# Match baseline: split on first comma only, preserving commas in query text
topics = {}
topics_path = os.path.join(DATASET_DIR, "topics.tsv")
with open(topics_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            topics[parts[0].strip()] = parts[1].strip()

if not topics:
    raise RuntimeError("Parsed 0 topics.")

# ================= LOAD QRELS =================
# Match baseline: qrels are comma-separated
qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        qid, _, pid, score = parts
        qid = qid.strip()
        pid = pid.strip()
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
encode_batch = load_query_encoder(model_name)
print("Running TopLoc-IVF evaluation...")

conv_cache = {}
k = 10
run = defaultdict(dict)

first_times = []
followup_times = []
first_n = 0
followup_n = 0

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    followup_keys = [t for t in turns[1:] if t in filtered_qrels]

    # ---- TURN 0: Single Search + Cache Centroids (ALWAYS RUN) ----
    q0_emb = encode_batch([topics[q0_key]])

    start_q0 = time.perf_counter()
    _, c0_indices = ivf_index.quantizer.search(q0_emb, H)
    conv_cache[conv_id] = c0_indices[0].astype("int64")
    scores_0, indices_0 = base_index.search(q0_emb, k)

    # Only record the time and results if q0 is judged
    if q0_key in filtered_qrels:
        first_times.append((time.perf_counter() - start_q0) * 1000)
        first_n += 1

        for idx, score in zip(indices_0[0], scores_0[0]):
            if idx >= 0 and id_map.get(str(idx)):
                run[q0_key][id_map[str(idx)]] = float(score)

    # ---- TURNS 1+: Batched Cached Search ----
    if not followup_keys:
        continue

    fu_texts = [topics[tk] for tk in followup_keys]
    fu_embs = encode_batch(fu_texts)

    start_fu = time.perf_counter()
    # scores_fu, indices_fu = toploc_ivf_search(
    #     ivf_index, fu_embs, conv_cache[conv_id], NP, k
    # )
    scores_fu, indices_fu = toploc_ivf_search_ptr(
        int(ivf_index.this), fu_embs, conv_cache[conv_id], NP, k
    )
    followup_times.append((time.perf_counter() - start_fu) * 1000)
    followup_n += len(followup_keys)

    for row_idx, turn_key in enumerate(followup_keys):
        for idx, score in zip(indices_fu[row_idx], scores_fu[row_idx]):
            if idx >= 0 and id_map.get(str(idx)):
                run[turn_key][id_map[str(idx)]] = float(score)

# ================= COMPUTE METRICS =================
# Sanity check to compare with the baseline run dict
print(f"\nDEBUG: I have {len(run)} turns in my run dict.")

measures = [nDCG @ 3, nDCG @ k, RR @ k]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

# ================= RESULTS =================
avg_first_ms = sum(first_times) / first_n if first_n else float("nan")
avg_followup_ms = sum(followup_times) / followup_n if followup_n else float("nan")

print("\n" + "=" * 60)
print(f"TOPLOC-IVF EVALUATION RESULTS ({index_type.upper()}, {model_name})")
print("=" * 60)
print(
    f"Turns evaluated:           {first_n + followup_n} ({first_n} first, {followup_n} follow-ups)"
)
print(f"NDCG@3:                    {results[nDCG @ 3]:.4f}")
print(f"NDCG@10:                   {results[nDCG @ k]:.4f}")
print(f"MRR@10:                    {results[RR @ k]:.4f}")
print("-" * 60)
print("Latency (Per-Query Averages):")
print(f"  First-Turn (Full Search):  {avg_first_ms:.2f} ms")
print(f"  Follow-Up (TopLoc Search): {avg_followup_ms:.2f} ms")
print("-" * 60)
print(f"Centroids cached per conv: {H}")
print(f"nprobe:                    {NP}")
print("=" * 60)
