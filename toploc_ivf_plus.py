#!/usr/bin/env python3
"""
TopLoc-IVF+ evaluation (topical-locality centroid caching WITH refresh).

Same as TopLoc-IVF, but the cached "hot" centroid set C0 is refreshed whenever
the conversation drifts away from its starting topic. Drift is detected with the
cheap I0 proxy from the paper:

    I0 = | top_np(qj, C0)  ∩  top_np(q0, C0) |

When |I0| falls below alpha * np, a refresh is triggered: C0 is recomputed from
the current utterance and that utterance becomes the new reference q0.

Run:
    python3 toploc_ivf_plus_eval.py snowflake ivf
    python3 toploc_ivf_plus_eval.py dragon    ivf

Env vars:
    CACHE_BASE   base dir holding <model>/<index>_index.index and _ids.npy
    DATASET_DIR  dir holding topics.tsv and qrels.qrel
    MMAP=1       memory-map the index (set this if you get "Killed" = OOM)
    H_CACHED     number of cached centroids h        (default 1024)
    NP           nprobe                              (default 8)
    ALPHA        refresh threshold fraction of np    (default 0.1)
"""

import os
import sys
import time
import math
import numpy as np
import faiss
from collections import defaultdict

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

# TopLoc-IVF+ hyperparameters (paper Section 3)
H_CACHED_CENTROIDS = int(os.environ.get("H_CACHED", 1024))  # h ∈ {512,1024,4096,8192}
NP = int(os.environ.get("NP", 8))  # nprobe
ALPHA = float(os.environ.get("ALPHA", 0.1))  # α ∈ {0.0,0.05,0.1,0.2}

USE_MMAP = os.environ.get("MMAP", "0") == "1"

# faiss.omp_set_num_threads(1)  # uncomment for single-threaded reproducible timing


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


# ================= TOPLOC IVF HELPERS =================
def get_centroid_vectors(quantizer, centroid_indices):
    """Fetch centroid vectors from the IVF quantizer, batched."""
    idx = np.asarray(centroid_indices, dtype="int64")
    try:
        return quantizer.reconstruct_batch(idx).astype("float32")
    except (AttributeError, RuntimeError):
        d = quantizer.d
        vecs = np.empty((len(idx), d), dtype="float32")
        for local_i, global_i in enumerate(idx):
            vecs[local_i] = quantizer.reconstruct(int(global_i))
        return vecs


def rank_within_cache(centroid_vecs, q_emb, nprobe, use_ip):
    """Return the local indices of the top-nprobe cached centroids for q_emb,
    plus the coarse scores for ALL cached centroids (reused by the I0 proxy)."""
    if use_ip:
        coarse = (centroid_vecs @ q_emb.T).reshape(-1)  # higher = closer
        order = np.argsort(-coarse)
    else:
        coarse = ((centroid_vecs - q_emb) ** 2).sum(axis=1)  # lower = closer
        order = np.argsort(coarse)
    return order[:nprobe], coarse


def toploc_ivf_search(
    index, q_emb, cached_centroid_indices, centroid_vecs, top_local, coarse, k, use_ip
):
    """Restricted IVF search over the cached centroids.

    centroid_vecs / top_local / coarse are passed in so we don't reconstruct or
    re-rank the cache twice (the IVF+ refresh check already did that work).
    """
    sel_centroids = (
        np.asarray(cached_centroid_indices)[top_local].astype("int64").reshape(1, -1)
    )
    sel_coarse = coarse[top_local].astype("float32").reshape(1, -1)

    try:
        scores, indices = index.search_preassigned(q_emb, k, sel_centroids, sel_coarse)
    except TypeError:
        scores, indices = index.search_preassigned(q_emb, k, sel_centroids)
    except AttributeError:
        raise RuntimeError(
            "search_preassigned is not available in your FAISS build. "
            "Upgrade to faiss-cpu>=1.7.3 to run TopLoc-IVF correctly."
        )
    return scores, indices


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


# ================= LOAD INDEX (memory-safe) =================
print(f"Evaluating TopLoc-IVF+ for: {model_name} ({index_type})")
index_path = os.path.join(cache_dir, f"{index_type}_index.index")

if USE_MMAP:
    print("Loading index with mmap (IO_FLAG_MMAP) to keep RAM usage low...")
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)
ivf_index.nprobe = NP

try:
    ivf_index.make_direct_map()
except Exception:
    pass

USE_IP = ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT
print(
    f"Index loaded: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if USE_IP else 'L2'}"
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
    raise RuntimeError("Parsed 0 topics. Check the delimiter/format of topics.tsv.")

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
print(f"Running TopLoc-IVF+ evaluation (h={H}, alpha={ALPHA}, np={NP})...")

# conv_cache[conv_id] = {"c0": int64 array, "q0_emb": float32 (1,d)}
conv_cache = {}
k, warmup = 10, 5
times, ndcgs, mrrs = [], [], []
evaluated_turns = 0
refresh_count = 0

for conv_id, turns in conversations.items():
    q0_key = turns[0]
    if q0_key not in filtered_qrels:
        continue

    # ---- TURN 0: full search + build initial cache ----
    q0_emb = encode_query(topics[q0_key])
    start = time.perf_counter()
    _, c0_indices = ivf_index.quantizer.search(q0_emb, H)
    conv_cache[conv_id] = {"c0": c0_indices[0].astype("int64"), "q0_emb": q0_emb}
    scores, indices = base_index.search(q0_emb, k)
    end = time.perf_counter()

    if evaluated_turns >= warmup:
        times.append((end - start) * 1000)
    retrieved_ids = [id_map.get(str(idx)) for idx in indices[0] if id_map.get(str(idx))]
    ndcgs.append(ndcg(retrieved_ids, filtered_qrels[q0_key], k))
    mrrs.append(mrr(retrieved_ids, filtered_qrels[q0_key], k))
    evaluated_turns += 1

    # ---- TURNS 1+: I0 proxy -> conditional refresh -> restricted search ----
    for turn_key in turns[1:]:
        if turn_key not in filtered_qrels:
            continue
        qj_emb = encode_query(topics[turn_key])

        start = time.perf_counter()
        cache = conv_cache[conv_id]

        # Reconstruct + rank the cached centroids ONCE for this query.
        c0_vecs = get_centroid_vectors(ivf_index.quantizer, cache["c0"])
        top_j_local, coarse_j = rank_within_cache(c0_vecs, qj_emb, NP, USE_IP)

        # I0 proxy: overlap between top-np for qj and top-np for q0, within C0.
        top_0_local, _ = rank_within_cache(c0_vecs, cache["q0_emb"], NP, USE_IP)
        i0_size = len(np.intersect1d(top_j_local, top_0_local))

        # Refresh when the conversation has drifted (overlap too small).
        if i0_size < ALPHA * NP:
            _, new_c0 = ivf_index.quantizer.search(qj_emb, H)
            cache = {"c0": new_c0[0].astype("int64"), "q0_emb": qj_emb}
            conv_cache[conv_id] = cache
            refresh_count += 1
            # Re-rank against the freshly cached centroids.
            c0_vecs = get_centroid_vectors(ivf_index.quantizer, cache["c0"])
            top_j_local, coarse_j = rank_within_cache(c0_vecs, qj_emb, NP, USE_IP)

        scores, indices = toploc_ivf_search(
            ivf_index,
            qj_emb,
            cache["c0"],
            c0_vecs,
            top_j_local,
            coarse_j,
            k,
            USE_IP,
        )
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
print(f"NDCG@10:                   {np.mean(ndcgs) if ndcgs else float('nan'):.4f}")
print(f"MRR@10:                    {np.mean(mrrs) if mrrs else float('nan'):.4f}")
print(f"Avg Time:                  {np.mean(times) if times else float('nan'):.2f} ms")
print(f"Centroids cached per conv: {H}")
print(f"nprobe:                    {NP}")
print(f"Alpha threshold (alpha):   {ALPHA}")
print(f"Cache refreshes triggered: {refresh_count}")
print("=" * 60)
