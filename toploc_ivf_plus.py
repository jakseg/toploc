import os
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

# ================= CONFIGURATION =================
CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc2/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc2/Datasets/conversational/CAST2019/topics"
)
CACHE_DIRS = {"snowflake": os.path.join(CACHE_BASE, "snowflake")}

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
cache_dir = CACHE_DIRS[model_name]

# TopLoc-IVF+ Hyperparameters (Paper Section 3)
H_CACHED_CENTROIDS = 1024  # h ∈ {512, 1024, 4096, 8192}
NP = 8  # nprobe
ALPHA = 0.1  # α ∈ {0.0, 0.05, 0.1, 0.2}

# Standardize threading for reproducible latency measurements
faiss.omp_set_num_threads(1)


# ================= METRIC FUNCTIONS =================
def dcg(scores, k):
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg(retrieved_ids, qrel_dict, k=10):
    rel_scores = [qrel_dict.get(pid, 0) for pid in retrieved_ids[:k]]
    ideal_scores = sorted(qrel_dict.values(), reverse=True)[:k]
    if not ideal_scores or max(ideal_scores) == 0:
        return 0.0
    return dcg(rel_scores, k) / dcg(ideal_scores, k)


def mrr(retrieved_ids, qrel_dict, k=10):
    for rank, pid in enumerate(retrieved_ids[:k], 1):
        if qrel_dict.get(pid, 0) > 0:
            return 1.0 / rank
    return 0.0


# ================= QUERY ENCODER =================
def load_query_encoder(model_name):
    if model_name == "snowflake":
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


# ================= TOPLOC IVF HELPERS =================
def get_centroid_vectors(quantizer, centroid_indices):
    """Python-safe centroid reconstruction (avoids C++ reconstruct_batch)"""
    d = quantizer.d
    vecs = np.empty((len(centroid_indices), d), dtype="float32")
    for local_i, global_i in enumerate(centroid_indices):
        vecs[local_i] = quantizer.reconstruct(int(global_i))
    return vecs


def toploc_ivf_search(index, q_emb, cached_centroid_indices, nprobe, k):
    """Restricted IVF search using cached centroids C0"""
    centroid_vecs = get_centroid_vectors(index.quantizer, cached_centroid_indices)

    if index.metric_type == faiss.METRIC_INNER_PRODUCT:
        sims = (centroid_vecs @ q_emb.T).squeeze()
        top_local = np.argsort(-sims)[:nprobe]
    else:
        dists = ((centroid_vecs - q_emb) ** 2).sum(axis=1)
        top_local = np.argsort(dists)[:nprobe]

    sel_centroids = cached_centroid_indices[top_local].astype("int64").reshape(1, -1)

    try:
        scores, indices = index.search_preassigned(q_emb, k, sel_centroids)
    # except AttributeError:
    #     scores, indices = index.search(q_emb, k)

    # or show an error ....
    except AttributeError:
        raise RuntimeError(
            "search_preassigned is not available in your FAISS build. "
            "Upgrade to faiss-cpu>=1.7.3 to run TopLoc-IVF correctly."
        )

    return scores, indices


# ================= LOAD DATA =================
print(f"Evaluating TopLoc-IVF+ for: {model_name} ({index_type})")

base_index = faiss.read_index(os.path.join(cache_dir, f"{index_type}_index.index"))
ivf_index = faiss.extract_index_ivf(base_index)
ivf_index.nprobe = NP

ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())

topics = {}
with open(os.path.join(DATASET_DIR, "topics.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            topics[parts[0].strip()] = parts[1].strip()

qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        qid, _, pid, score = parts
        if int(score) > 0:
            qrels[qid][pid] = int(score)

filtered_qrels = {
    k: {p: s for p, s in v.items() if p in indexed_pids}
    for k, v in qrels.items()
    if any(p in indexed_pids for p in v)
}

conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)

print(f"Grouped {len(topics)} turns into {len(conversations)} conversations")


# ================= EVALUATION LOOP =================
print(f"\nLoading {model_name} query encoder...")
encode_query = load_query_encoder(model_name)
print(f"Running TopLoc-IVF+ evaluation (h={H_CACHED_CENTROIDS}, α={ALPHA}, np={NP})...")

conv_cache = {}  # {conv_id: {"c0": np.array, "q0_emb": np.array}}
k, warmup = 10, 5
times, ndcgs, mrrs, evaluated_turns = [], [], [], 0
refresh_count = 0

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    if q0_key not in filtered_qrels:
        continue

    q0_emb = encode_query(topics[q0_key])

    # TURN 0: Build initial cache + search
    start = time.perf_counter()
    _, c0_indices = ivf_index.quantizer.search(q0_emb, H_CACHED_CENTROIDS)
    conv_cache[conv_id] = {"c0": c0_indices[0].astype("int64"), "q0_emb": q0_emb}
    scores, indices = base_index.search(q0_emb, k)
    end = time.perf_counter()

    if evaluated_turns >= warmup:
        times.append((end - start) * 1000)
    retrieved_ids = [id_map.get(str(idx)) for idx in indices[0] if id_map.get(str(idx))]
    ndcgs.append(ndcg(retrieved_ids, filtered_qrels[q0_key], k))
    mrrs.append(mrr(retrieved_ids, filtered_qrels[q0_key], k))
    evaluated_turns += 1

    # TURNS 1+: I0 proxy check + conditional refresh + restricted search
    for turn_key in turns[1:]:
        if turn_key not in filtered_qrels:
            continue
        qj_emb = encode_query(topics[turn_key])

        start = time.perf_counter()

        cache_data = conv_cache[conv_id]
        c0_vecs = get_centroid_vectors(ivf_index.quantizer, cache_data["c0"])

        # Compute I0 proxy
        sims_j = (c0_vecs @ qj_emb.T).squeeze()
        sims_0 = (c0_vecs @ cache_data["q0_emb"].T).squeeze()

        top_rel_j = np.argsort(-sims_j)[:NP]
        top_rel_0 = np.argsort(-sims_0)[:NP]

        global_j = cache_data["c0"][top_rel_j]
        global_0 = cache_data["c0"][top_rel_0]

        i0_size = len(np.intersect1d(global_j, global_0))

        if i0_size < ALPHA * NP:
            _, new_c0 = ivf_index.quantizer.search(qj_emb, H_CACHED_CENTROIDS)
            conv_cache[conv_id] = {"c0": new_c0[0].astype("int64"), "q0_emb": qj_emb}
            cache_data = conv_cache[conv_id]
            refresh_count += 1

        # Restricted search
        scores, indices = toploc_ivf_search(ivf_index, qj_emb, cache_data["c0"], NP, k)
        end = time.perf_counter()

        if evaluated_turns >= warmup:
            times.append((end - start) * 1000)
        retrieved_ids = [
            id_map.get(str(idx)) for idx in indices[0] if id_map.get(str(idx))
        ]
        ndcgs.append(ndcg(retrieved_ids, filtered_qrels[turn_key], k))
        mrrs.append(mrr(retrieved_ids, filtered_qrels[turn_key], k))
        evaluated_turns += 1


# ================= RESULTS =================
print("\n" + "=" * 60)
print(f"TOPLOC-IVF+ EVALUATION RESULTS ({index_type.upper()}, {model_name})")
print("=" * 60)
print(f"Turns evaluated:           {evaluated_turns}")
print(f"NDCG@10:                   {np.mean(ndcgs):.4f}")
print(f"MRR@10:                    {np.mean(mrrs):.4f}")
print(f"Avg Time:                  {np.mean(times):.2f} ms")
print(f"Centroids cached per conv: {H_CACHED_CENTROIDS}")
print(f"nprobe:                    {NP}")
print(f"Alpha threshold (α):       {ALPHA}")
print(f"Cache refreshes triggered: {refresh_count}")
print("=" * 60)
