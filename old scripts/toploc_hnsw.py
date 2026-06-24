"""
TopLoc-HNSW prototype for TREC CAsT 2019.

Idea:
- Build/load the normal FAISS HNSW index from create_index.py.
- For the first query q0 of each conversation:
    1. run normal HNSW with a larger efSearch
    2. save the best returned document node as the conversation entry point
- For later turns q1, q2, ...:
    1. skip the normal top-layer HNSW routing
    2. start a Python beam search directly from the cached entry point on level 0

Important note:
FAISS Python does not expose a clean public API to start IndexHNSWFlat.search()
from a custom entry point. Therefore this file implements the level-0 beam search
in Python using the stored FAISS HNSW graph neighbors. This is useful as a clear
prototype and correctness baseline. For final speed measurements, the same logic
should ideally be moved to C++ or to a library that exposes custom HNSW entry points.

Usage:
    python toploc_hnsw.py snowflake
    python toploc_hnsw.py snowflake --ef-search 64 --up 2 --entry-points 1
"""

import argparse
import heapq
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss is required. Install it with something like: pip install faiss-cpu"
    ) from exc

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Missing model dependencies. Install sentence-transformers, transformers, and torch."
    ) from exc


# ================= DEFAULT PATHS =================
CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc2/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc2/Datasets/conversational/CAST2019/topics"
)

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}


# ================= METRICS =================
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


# ================= QUERY ENCODERS =================
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

    if model_name == "dragon":
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
                outputs = model(**tokens)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().astype("float32")

        return encode

    raise ValueError(f"Unknown model: {model_name}")


# ================= DATA LOADING =================
def load_topics(path):
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                topics[parts[0].strip()] = parts[1].strip()
    return topics


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            qid, _, pid, score = parts
            try:
                score = int(score)
            except ValueError:
                continue
            if score > 0:
                qrels[qid][pid] = score
    return qrels


def turn_sort_key(turn_key):
    """Sort keys like '31_1', '31_2', ... in conversation order."""
    numbers = re.findall(r"\d+", turn_key)
    return tuple(int(x) for x in numbers) if numbers else (10**9,)


def group_conversations(topics):
    conversations = defaultdict(list)
    for turn_key in topics:
        conv_id = turn_key.split("_")[0]
        conversations[conv_id].append(turn_key)

    for conv_id in conversations:
        conversations[conv_id].sort(key=turn_sort_key)

    return conversations


# ================= HNSW GRAPH ACCESS =================
def hnsw_level0_neighbors(index, node_id):
    """Return valid level-0 neighbors for one FAISS IndexHNSWFlat node."""
    hnsw = index.hnsw

    # FAISS stores neighbors in flattened arrays. For level 0, the neighbor block
    # starts at offsets[node_id] and has hnsw.nb_neighbors(0) slots.
    offsets = faiss.vector_to_array(hnsw.offsets)
    neighbors = faiss.vector_to_array(hnsw.neighbors)

    start = int(offsets[node_id])
    degree = int(hnsw.nb_neighbors(0))
    end = start + degree

    neigh = neighbors[start:end]
    return [int(x) for x in neigh if int(x) >= 0]


def reconstruct_vector(index, node_id):
    vec = np.empty(index.d, dtype="float32")
    index.reconstruct(int(node_id), vec)
    return vec


def similarity(index, query_vec, node_id):
    """Higher is better. Current indexes use inner product over normalized vectors."""
    vec = reconstruct_vector(index, node_id)
    return float(np.dot(query_vec, vec))


# ================= TOPLOC-HNSW LEVEL-0 SEARCH =================
def toploc_hnsw_level0_search(index, q_emb, entry_points, k=10, ef_search=64):
    """
    Python implementation of a simple HNSW level-0 beam search.

    Parameters
    ----------
    index:
        FAISS IndexHNSWFlat.
    q_emb:
        Query embedding with shape (1, dim).
    entry_points:
        Internal FAISS node ids used as starting points.
    k:
        Number of final results.
    ef_search:
        Candidate/result beam size. Larger means better quality but more work.

    Returns
    -------
    scores, indices:
        Same shape style as FAISS search: (1, k), (1, k)
    visited_count:
        Number of graph nodes whose score was computed.
    """
    query_vec = q_emb.reshape(-1).astype("float32")

    # candidate heap: (-score, node_id), so best score is popped first
    candidates = []

    # result heap: (score, node_id), so worst current result is at heap[0]
    results = []

    visited = set()

    def add_candidate(node_id):
        if node_id < 0 or node_id in visited:
            return
        visited.add(node_id)
        score = similarity(index, query_vec, node_id)
        heapq.heappush(candidates, (-score, node_id))

        if len(results) < ef_search:
            heapq.heappush(results, (score, node_id))
        elif score > results[0][0]:
            heapq.heapreplace(results, (score, node_id))

    for ep in entry_points:
        add_candidate(int(ep))

    while candidates:
        neg_score, current = heapq.heappop(candidates)
        current_score = -neg_score

        # Standard HNSW stopping condition:
        # if the best unexplored candidate is worse than the worst element in the
        # result beam, further expansion is unlikely to improve the result set.
        if len(results) >= ef_search and current_score < results[0][0]:
            break

        for nb in hnsw_level0_neighbors(index, current):
            add_candidate(nb)

    top = sorted(results, key=lambda x: x[0], reverse=True)[:k]

    scores = np.full((1, k), -np.inf, dtype="float32")
    indices = np.full((1, k), -1, dtype="int64")

    for rank, (score, node_id) in enumerate(top):
        scores[0, rank] = score
        indices[0, rank] = node_id

    return scores, indices, len(visited)


# ================= MAIN EVALUATION =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["snowflake", "dragon"], nargs="?", default="snowflake")
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--up", type=int, default=2, help="Upscaling factor for the first query q0")
    parser.add_argument("--entry-points", type=int, default=1, help="How many q0 results to cache as entry points")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)

    model_name = args.model
    cache_dir = CACHE_DIRS[model_name]
    index_path = os.path.join(cache_dir, "hnsw_index.index")
    ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
    topics_path = os.path.join(DATASET_DIR, "topics.tsv")
    qrels_path = os.path.join(DATASET_DIR, "qrels.qrel")

    print(f"Evaluating TopLoc-HNSW prototype for: {model_name}")
    print(f"Index path: {index_path}")

    if not os.path.exists(index_path):
        print(f"ERROR: HNSW index not found: {index_path}")
        print(f"Run first: python create_index.py {model_name} hnsw")
        sys.exit(1)

    index = faiss.read_index(index_path)
    if not hasattr(index, "hnsw"):
        raise TypeError("Loaded index is not an HNSW index.")

    index.hnsw.efSearch = args.ef_search
    print(f"Loaded HNSW index: ntotal={index.ntotal:,}, dim={index.d}")
    print(f"Base efSearch={args.ef_search}, q0 upscaling factor={args.up}")

    id_array = np.load(ids_path, allow_pickle=True)
    id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
    indexed_pids = set(id_map.values())
    print(f"ID map loaded: {len(id_map):,} passages")

    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    conversations = group_conversations(topics)

    filtered_qrels = {
        qid: {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
        for qid, pid_scores in qrels.items()
    }
    filtered_qrels = {qid: vals for qid, vals in filtered_qrels.items() if vals}

    print(f"Topics loaded: {len(topics):,} turns")
    print(f"Conversations: {len(conversations):,}")
    print(f"Turns with relevant passages in index: {len(filtered_qrels):,}/{len(qrels):,}")

    print(f"\nLoading {model_name} query encoder...")
    encode_query = load_query_encoder(model_name)

    times = []
    ndcgs = []
    mrrs = []
    visited_counts = []
    evaluated_turns = 0
    first_turns = 0
    followup_turns = 0

    print("Running TopLoc-HNSW evaluation...")

    for conv_id, turns in conversations.items():
        valid_turns = [t for t in turns if t in filtered_qrels]
        if not valid_turns:
            continue

        q0_key = valid_turns[0]
        q0_emb = encode_query(topics[q0_key])

        # First turn: normal HNSW with larger efSearch to get a strong entry point.
        old_ef = index.hnsw.efSearch
        index.hnsw.efSearch = args.ef_search * args.up

        start = time.perf_counter()
        q0_scores, q0_indices = index.search(q0_emb, max(args.k, args.entry_points))
        end = time.perf_counter()

        index.hnsw.efSearch = old_ef

        entry_points = [int(x) for x in q0_indices[0][: args.entry_points] if int(x) >= 0]
        if not entry_points:
            continue

        if evaluated_turns >= args.warmup:
            times.append((end - start) * 1000)

        retrieved_ids = [id_map.get(str(idx)) for idx in q0_indices[0][: args.k] if id_map.get(str(idx))]
        ndcgs.append(ndcg(retrieved_ids, filtered_qrels[q0_key], args.k))
        mrrs.append(mrr(retrieved_ids, filtered_qrels[q0_key], args.k))

        evaluated_turns += 1
        first_turns += 1

        # Follow-up turns: start from cached q0 entry point(s), search level 0.
        for turn_key in valid_turns[1:]:
            q_emb = encode_query(topics[turn_key])

            start = time.perf_counter()
            scores, indices, visited_count = toploc_hnsw_level0_search(
                index=index,
                q_emb=q_emb,
                entry_points=entry_points,
                k=args.k,
                ef_search=args.ef_search,
            )
            end = time.perf_counter()

            if evaluated_turns >= args.warmup:
                times.append((end - start) * 1000)

            retrieved_ids = [id_map.get(str(idx)) for idx in indices[0] if id_map.get(str(idx))]
            ndcgs.append(ndcg(retrieved_ids, filtered_qrels[turn_key], args.k))
            mrrs.append(mrr(retrieved_ids, filtered_qrels[turn_key], args.k))
            visited_counts.append(visited_count)

            evaluated_turns += 1
            followup_turns += 1

    print("\n" + "=" * 70)
    print(f"TOPLOC-HNSW PROTOTYPE RESULTS ({model_name})")
    print("=" * 70)
    print(f"Turns evaluated:          {evaluated_turns}")
    print(f"First turns:              {first_turns}")
    print(f"Follow-up turns:          {followup_turns}")
    print(f"NDCG@10:                  {np.mean(ndcgs):.4f}" if ndcgs else "NDCG@10:                  n/a")
    print(f"MRR@10:                   {np.mean(mrrs):.4f}" if mrrs else "MRR@10:                   n/a")
    print(f"Avg Time:                 {np.mean(times):.2f} ms" if times else "Avg Time:                 n/a")
    print(f"Base efSearch:            {args.ef_search}")
    print(f"q0 efSearch:              {args.ef_search * args.up}")
    print(f"Cached entry points/conv: {args.entry_points}")
    if visited_counts:
        print(f"Avg visited nodes:        {np.mean(visited_counts):.1f}")
    print("=" * 70)
    print("NOTE: Follow-up search uses a Python level-0 beam search. It implements the")
    print("TopLoc-HNSW idea, but final latency should be measured with a C++/native version.")


if __name__ == "__main__":
    main()
