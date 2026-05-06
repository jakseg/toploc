import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import time
import math
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
data_dir = f"data/{model_name}"

# ================= METRIC FUNCTIONS =================
def dcg(scores, k):
    """Discounted Cumulative Gain @ k"""
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))

def ndcg(retrieved_ids, qrel_dict, k=10):
    """Normalized DCG @ k"""
    rel_scores = [qrel_dict.get(pid, 0) for pid in retrieved_ids[:k]]
    ideal_scores = sorted(qrel_dict.values(), reverse=True)[:k]
    if not ideal_scores or max(ideal_scores) == 0:
        return 0.0
    return dcg(rel_scores, k) / dcg(ideal_scores, k)

def mrr(retrieved_ids, qrel_dict, k=10):
    """Mean Reciprocal Rank @ k"""
    for rank, pid in enumerate(retrieved_ids[:k], 1):
        if qrel_dict.get(pid, 0) > 0:
            return 1.0 / rank
    return 0.0

# ================= QUERY ENCODER =================
def load_query_encoder(model_name):
    if model_name == "snowflake":
        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
        def encode(query):
            return model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        return encode
    elif model_name == "dragon":
        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
        model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
        model.eval()
        def encode(query):
            tokens = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokens)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().astype("float32")
        return encode
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ================= LOAD DATA =================
print(f"Evaluating baseline for: {model_name}")
print("Loading components...")

# 1. Load index (exact, ivf, or hnsw)
index = faiss.read_index(f"{data_dir}/{index_type}_index.index")
print(f"{index_type.upper()} index loaded: {index.ntotal} vectors, trained={index.is_trained}")

# Set search parameters depending on index type
if index_type == "ivf":
    index.nprobe = 8  # Number of clusters to scan (tune later)
    print(f"Set nprobe={index.nprobe}")
elif index_type == "hnsw":
    index.hnsw.efSearch = 8  # Number of candidates to explore (tune later)
    print(f"Set efSearch={index.hnsw.efSearch}")

# 2. ID Mapping
with open(f"{data_dir}/passage_id_map.json", "r") as f:
    id_map = json.load(f)
indexed_pids = set(id_map.values())
print(f"ID map loaded: {len(id_map)} passages")

# 3. Topics (Queries)
topics = {}
with open("dataset/topics.tsv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            topics[parts[0].strip()] = parts[1].strip()
print(f"Topics loaded: {len(topics)} turns")

# 4. QRELS (Relevance Judgments)
# Format: qid,0,pid,relevance (e.g. 31_1,0,CAR_xxx,2)
qrels = defaultdict(dict)
with open("dataset/qrels.qrel", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 4: continue
        qid, _, pid, score = parts
        score = int(score)
        if score > 0:
            qrels[qid][pid] = score

# Filter qrels to only include passages we actually indexed
filtered_qrels = {}
total_turns = len(qrels)
for turn_key, pid_scores in qrels.items():
    valid_pids = {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
    if valid_pids:
        filtered_qrels[turn_key] = valid_pids

print(f"Subset Warning: {len(filtered_qrels)}/{total_turns} turns have relevant passages in your 2k subset.")
print("   (Metrics will be artificially low until you index the full collection)")

# ================= EVALUATION LOOP =================
print(f"\nLoading {model_name} query encoder...")
encode_query = load_query_encoder(model_name)
print("Running baseline evaluation...")

k = 10
warmup = 5
times = []
ndcgs = []
mrrs = []

evaluated_turns = 0
for i, (turn_key, query) in enumerate(topics.items()):
    if turn_key not in filtered_qrels:
        continue

    # Embed query
    q_emb = encode_query(query)

    # Search & measure latency
    start = time.perf_counter()
    scores, indices = index.search(q_emb, k)
    end = time.perf_counter()

    if i >= warmup:
        times.append((end - start) * 1000)  # ms

    # Map FAISS indices to passage IDs
    retrieved_ids = [id_map.get(str(idx)) for idx in indices[0] if id_map.get(str(idx)) is not None]

    # Evaluate
    qrel_dict = filtered_qrels[turn_key]
    ndcgs.append(ndcg(retrieved_ids, qrel_dict, k))
    mrrs.append(mrr(retrieved_ids, qrel_dict, k))
    evaluated_turns += 1

# ================= RESULTS =================
print("\n" + "=" * 60)
print(f"BASELINE EVALUATION RESULTS ({index_type.upper()}, {model_name})")
print("=" * 60)
print(f"Turns evaluated: {evaluated_turns}")
print(f"NDCG@10: {np.mean(ndcgs):.4f}")
print(f"MRR@10:  {np.mean(mrrs):.4f}")
print(f"Avg Time: {np.mean(times):.2f} ms")
print("=" * 60)

if model_name == "snowflake":
    print("\nPaper Baseline Targets (Snowflake, Full Collection):")
    print("   NDCG@10 = 0.497 | MRR@10 = 0.815 | Time = 24.9 ms")
elif model_name == "dragon":
    print("\nPaper Baseline Targets (Dragon, Full Collection):")
    print("   NDCG@10 = 0.486 | MRR@10 = 0.813 | Time = 33.0 ms")
print("Your scores will be lower due to the 2k passage subset.")
