#!/usr/bin/env python3
"""
TopLoc-HNSW prototype for TREC CAsT 2019.

This is the HNSW analogue of Ahmad's TopLoc-IVF code:
- q0 of each conversation: run normal FAISS HNSW with higher efSearch.
- cache the best result(s) of q0 as privileged entry point(s).
- follow-up turns: start a custom level-0 beam search from the cached entry point(s).

Important:
FAISS Python does not expose a public "search from this custom HNSW entry point"
API. Therefore the follow-up search is implemented in Python. This version is
optimized compared to the first prototype by:
- lazy-loading heavy ML libraries only inside load_query_encoder,
- optionally memory-mapping the FAISS index with MMAP=1,
- reading the FAISS HNSW neighbor arrays only once,
- scoring newly visited graph nodes in batches with reconstruct_batch,
- printing progress with flush=True.

Run:
    python -u toploc_hnsw.py snowflake
    python -u toploc_hnsw.py snowflake --ef-search 64 --up 2 --entry-points 1
    python -u toploc_hnsw.py snowflake --max-turns 20

Env vars:
    CACHE_BASE   base dir holding <model>/hnsw_index.index and hnsw_ids.npy
    DATASET_DIR  dir holding topics.tsv and qrels.qrel
    MMAP=1       memory-map the index if RAM is a problem
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
    raise ImportError("faiss is required. Install it with: pip install faiss-cpu") from exc


# ================= DEFAULT PATHS =================
CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc2/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc2/Datasets/conversational/CAST2019/topics"
)

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

USE_MMAP = os.environ.get("MMAP", "0") == "1"


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

    if model_name == "dragon":
        import torch
        from transformers import AutoModel, AutoTokenizer

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


# ================= FLEXIBLE PARSERS =================
def split_flexible(line, expected):
    """Accept TSV, CSV, or whitespace-separated topic/qrel files."""
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


def load_topics(path):
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = split_flexible(line, 2)
            if parts:
                topics[parts[0]] = parts[1]
    return topics


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
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
def load_hnsw_level0_graph(index):
    """Copy FAISS HNSW graph arrays once instead of on every neighbor lookup."""
    hnsw = index.hnsw
    offsets = faiss.vector_to_array(hnsw.offsets).astype("int64", copy=False)
    neighbors = faiss.vector_to_array(hnsw.neighbors).astype("int64", copy=False)
    degree0 = int(hnsw.nb_neighbors(0))
    return offsets, neighbors, degree0


def level0_neighbors(graph, node_id):
    offsets, neighbors, degree0 = graph
    start = int(offsets[int(node_id)])
    block = neighbors[start : start + degree0]
    return block[block >= 0]


def reconstruct_batch_safe(index, node_ids):
    """Return vectors for node_ids using batched FAISS reconstruction when possible."""
    ids = np.asarray(node_ids, dtype="int64")
    try:
        return index.reconstruct_batch(ids).astype("float32")
    except Exception:
        vecs = np.empty((len(ids), index.d), dtype="float32")
        for i, node_id in enumerate(ids):
            index.reconstruct(int(node_id), vecs[i])
        return vecs


# ================= TOPLOC-HNSW LEVEL-0 SEARCH =================
def toploc_hnsw_level0_search(index, graph, q_emb, entry_points, k=10, ef_search=64):
    """Python implementation of HNSW level-0 beam search from custom entry points.

    This is still slower than native FAISS/C++, but much faster than reconstructing
    one vector and reading the graph arrays for every single neighbor.
    """
    query_vec = q_emb.reshape(-1).astype("float32")

    # candidate heap: (-score, node_id), so best score is popped first.
    candidates = []
    # result heap: (score, node_id), so the worst current result is at heap[0].
    results = []
    visited = set()

    def add_candidates(node_ids):
        new_ids = []
        for node_id in node_ids:
            node_id = int(node_id)
            if node_id >= 0 and node_id not in visited:
                visited.add(node_id)
                new_ids.append(node_id)

        if not new_ids:
            return

        vecs = reconstruct_batch_safe(index, new_ids)
        scores = vecs @ query_vec

        for node_id, score in zip(new_ids, scores):
            score = float(score)
            heapq.heappush(candidates, (-score, node_id))
            if len(results) < ef_search:
                heapq.heappush(results, (score, node_id))
            elif score > results[0][0]:
                heapq.heapreplace(results, (score, node_id))

    add_candidates(entry_points)

    while candidates:
        neg_score, current = heapq.heappop(candidates)
        current_score = -neg_score

        if len(results) >= ef_search and current_score < results[0][0]:
            break

        add_candidates(level0_neighbors(graph, current))

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
    parser.add_argument("--ef-search", type=int, default=int(os.environ.get("EF_SEARCH", 64)))
    parser.add_argument("--up", type=int, default=int(os.environ.get("UP", 2)), help="Upscaling factor for q0")
    parser.add_argument("--entry-points", type=int, default=int(os.environ.get("ENTRY_POINTS", 1)))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=0, help="Debug limit. 0 means all turns.")
    parser.add_argument("--threads", type=int, default=1, help="FAISS OpenMP threads for native q0 search")
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)

    model_name = args.model
    cache_dir = CACHE_DIRS[model_name]
    index_path = os.path.join(cache_dir, "hnsw_index.index")
    ids_path = os.path.join(cache_dir, "hnsw_ids.npy")
    topics_path = os.path.join(DATASET_DIR, "topics.tsv")
    qrels_path = os.path.join(DATASET_DIR, "qrels.qrel")

    print(f"Evaluating TopLoc-HNSW prototype for: {model_name}", flush=True)
    print(f"Index path: {index_path}", flush=True)

    if not os.path.exists(index_path):
        print(f"ERROR: HNSW index not found: {index_path}", flush=True)
        print(f"Run first: python create_index.py {model_name} hnsw", flush=True)
        sys.exit(1)

    if USE_MMAP:
        print("Loading HNSW index with mmap (IO_FLAG_MMAP)...", flush=True)
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(index_path)

    if not hasattr(index, "hnsw"):
        raise TypeError("Loaded index is not an HNSW index.")

    index.hnsw.efSearch = args.ef_search
    print(f"Loaded HNSW index: ntotal={index.ntotal:,}, dim={index.d}", flush=True)
    print(f"Base efSearch={args.ef_search}, q0 upscaling factor={args.up}", flush=True)

    print("Loading HNSW level-0 graph arrays...", flush=True)
    graph = load_hnsw_level0_graph(index)
    print(f"Level-0 degree slots: {graph[2]}", flush=True)

    id_array = np.load(ids_path, allow_pickle=True)
    id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
    indexed_pids = set(id_map.values())
    print(f"ID map loaded: {len(id_map):,} passages", flush=True)

    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    conversations = group_conversations(topics)

    filtered_qrels = {
        qid: {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
        for qid, pid_scores in qrels.items()
    }
    filtered_qrels = {qid: vals for qid, vals in filtered_qrels.items() if vals}

    if not topics:
        raise RuntimeError("Parsed 0 topics. Check topics.tsv delimiter/format.")
    if not filtered_qrels:
        raise RuntimeError("No qrels survived filtering against indexed passage ids.")

    print(f"Topics loaded: {len(topics):,} turns", flush=True)
    print(f"Conversations: {len(conversations):,}", flush=True)
    print(f"Turns with relevant passages in index: {len(filtered_qrels):,}/{len(qrels):,}", flush=True)

    print(f"\nLoading {model_name} query encoder...", flush=True)
    encode_query = load_query_encoder(model_name)

    times, ndcgs, mrrs = [], [], []
    first_times, followup_times = [], []
    visited_counts = []
    evaluated_turns = 0
    first_turns = 0
    followup_turns = 0

    print("Running TopLoc-HNSW evaluation...", flush=True)
    wall_start = time.perf_counter()

    stop = False
    for conv_i, (conv_id, turns) in enumerate(conversations.items(), start=1):
        valid_turns = [t for t in turns if t in filtered_qrels]
        if not valid_turns:
            continue

        q0_key = valid_turns[0]
        q0_emb = encode_query(topics[q0_key])

        old_ef = index.hnsw.efSearch
        index.hnsw.efSearch = args.ef_search * args.up

        start = time.perf_counter()
        q0_scores, q0_indices = index.search(q0_emb, max(args.k, args.entry_points))
        end = time.perf_counter()

        index.hnsw.efSearch = old_ef

        entry_points = [int(x) for x in q0_indices[0][: args.entry_points] if int(x) >= 0]
        if not entry_points:
            continue

        dt_ms = (end - start) * 1000
        if evaluated_turns >= args.warmup:
            times.append(dt_ms)
            first_times.append(dt_ms)

        retrieved_ids = [id_map.get(str(idx)) for idx in q0_indices[0][: args.k] if id_map.get(str(idx))]
        ndcgs.append(ndcg(retrieved_ids, filtered_qrels[q0_key], args.k))
        mrrs.append(mrr(retrieved_ids, filtered_qrels[q0_key], args.k))

        evaluated_turns += 1
        first_turns += 1

        if args.max_turns and evaluated_turns >= args.max_turns:
            stop = True

        for turn_key in valid_turns[1:]:
            if stop:
                break

            q_emb = encode_query(topics[turn_key])

            start = time.perf_counter()
            scores, indices, visited_count = toploc_hnsw_level0_search(
                index=index,
                graph=graph,
                q_emb=q_emb,
                entry_points=entry_points,
                k=args.k,
                ef_search=args.ef_search,
            )
            end = time.perf_counter()

            dt_ms = (end - start) * 1000
            if evaluated_turns >= args.warmup:
                times.append(dt_ms)
                followup_times.append(dt_ms)

            retrieved_ids = [id_map.get(str(idx)) for idx in indices[0] if id_map.get(str(idx))]
            ndcgs.append(ndcg(retrieved_ids, filtered_qrels[turn_key], args.k))
            mrrs.append(mrr(retrieved_ids, filtered_qrels[turn_key], args.k))
            visited_counts.append(visited_count)

            evaluated_turns += 1
            followup_turns += 1

            if args.max_turns and evaluated_turns >= args.max_turns:
                stop = True
                break

        print(
            f"Progress: conv {conv_i}/{len(conversations)} | evaluated={evaluated_turns} "
            f"| followups={followup_turns}",
            flush=True,
        )

        if stop:
            break

    wall_time = time.perf_counter() - wall_start

    print("\n" + "=" * 70)
    print(f"TOPLOC-HNSW PROTOTYPE RESULTS ({model_name})")
    print("=" * 70)
    print(f"Turns evaluated:          {evaluated_turns}")
    print(f"First turns:              {first_turns}")
    print(f"Follow-up turns:          {followup_turns}")
    print(f"NDCG@10:                  {np.mean(ndcgs):.4f}" if ndcgs else "NDCG@10:                  n/a")
    print(f"MRR@10:                   {np.mean(mrrs):.4f}" if mrrs else "MRR@10:                   n/a")
    print(f"Avg Time:                 {np.mean(times):.2f} ms" if times else "Avg Time:                 n/a")
    if first_times:
        print(f"Avg q0 native time:       {np.mean(first_times):.2f} ms")
    if followup_times:
        print(f"Avg follow-up time:       {np.mean(followup_times):.2f} ms")
    print(f"Total wall time:          {wall_time / 60:.2f} min")
    print(f"Base efSearch:            {args.ef_search}")
    print(f"q0 efSearch:              {args.ef_search * args.up}")
    print(f"Cached entry points/conv: {args.entry_points}")
    if visited_counts:
        print(f"Avg visited nodes:        {np.mean(visited_counts):.1f}")
    print("=" * 70)
    print("NOTE: Follow-up search still uses Python level-0 beam search.")
    print("It tests the TopLoc-HNSW logic, but real latency needs C++/native HNSW.")


if __name__ == "__main__":
    main()
