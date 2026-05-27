#!/usr/bin/env python3
"""
TopLoc combined evaluation — loads everything ONCE, runs both IVF and IVF+.

What used to cost:
  run toploc_ivf.py      → load index + load model + load topics + load qrels
  run toploc_ivf_plus.py → load index + load model + load topics + load qrels (AGAIN)

Now costs:
  run toploc_combined.py → load index + load model + load topics + load qrels (ONCE)
                         → run IVF eval
                         → run IVF+ eval

Run:
    python3 toploc_combined.py snowflake ivf
    python3 toploc_combined.py dragon    ivf

Env vars:
    CACHE_BASE    base dir
    DATASET_DIR   dir holding topics.tsv and qrels.qrel
    MMAP=1        memory-map the index if RAM is tight
    H_CACHED      cached centroids    (default 1024)
    NP            nprobe              (default 8)
    ALPHA         IVF+ drift thresh   (default 0.1)
"""

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
NP = int(os.environ.get("NP", 8))
ALPHA = float(os.environ.get("ALPHA", 0.1))
USE_MMAP = os.environ.get("MMAP", "0") == "1"


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


# ================= IVF+ HELPERS =================
# These are only needed for the IVF+ drift check.
# The C++ function handles everything for plain IVF.


def get_centroid_vectors(quantizer, centroid_indices):
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
    if use_ip:
        coarse = (centroid_vecs @ q_emb.T).reshape(-1)
        order = np.argsort(-coarse)
    else:
        coarse = ((centroid_vecs - q_emb) ** 2).sum(axis=1)
        order = np.argsort(coarse)
    return order[:nprobe], coarse


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


# =================================================================
# LOAD EVERYTHING ONCE
# =================================================================
print(f"Model: {model_name} | Index: {index_type}")
print("=" * 60)
print("Loading shared resources (index, encoder, topics, qrels)...")

# ── Index ─────────────────────────────────────────────────────────
index_path = os.path.join(cache_dir, f"{index_type}_index.index")
if USE_MMAP:
    print("  Loading index with mmap...")
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)
ivf_index.nprobe = NP
USE_IP = ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT

try:
    ivf_index.make_direct_map()
except Exception:
    pass

print(
    f"  Index: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if USE_IP else 'L2'}"
)

H = min(H_CACHED_CENTROIDS, ivf_index.nlist)
if H != H_CACHED_CENTROIDS:
    print(f"  Reducing cached centroids to nlist: {H}")

# ── ID mapping ────────────────────────────────────────────────────
ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())
print(f"  ID map: {len(id_map)} passages")

# ── Topics ────────────────────────────────────────────────────────
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
print(f"  Topics: {len(topics)}")

# ── Qrels ─────────────────────────────────────────────────────────
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
print(f"  Qrels: {len(filtered_qrels)} turns with relevant passages")

# ── Conversations ─────────────────────────────────────────────────
conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)
print(f"  Conversations: {len(conversations)} ({len(topics)} turns)")

# ── Query encoder ─────────────────────────────────────────────────
print(f"  Loading {model_name} query encoder...")
encode_query = load_query_encoder(model_name)
print("Shared resources loaded. Starting evaluations...\n")


# =================================================================
# EVAL 1 — TopLoc-IVF
# =================================================================
def run_ivf_eval():
    print("=" * 60)
    print(f"Running TopLoc-IVF (H={H}, NP={NP})...")
    print("=" * 60)

    conv_cache = {}
    k, warmup = 10, 5
    times = []
    evaluated_turns = 0
    run = defaultdict(dict)

    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        if q0_key not in filtered_qrels:
            continue

        # Turn 0 — full search + cache centroids
        q0_emb = encode_query(topics[q0_key])
        start = time.perf_counter()
        _, c0_indices = ivf_index.quantizer.search(q0_emb, H)
        conv_cache[conv_id] = c0_indices[0].astype("int64")
        scores, indices = base_index.search(q0_emb, k)
        end = time.perf_counter()

        if evaluated_turns >= warmup:
            times.append((end - start) * 1000)

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            pid = id_map.get(str(idx))
            if pid is not None:
                run[q0_key][pid] = float(score)
        evaluated_turns += 1

        # Turns 1+ — C++ restricted search
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

            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                pid = id_map.get(str(idx))
                if pid is not None:
                    run[turn_key][pid] = float(score)
            evaluated_turns += 1

    measures = [nDCG @ 3, nDCG @ k, RR @ k]
    results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

    print(f"Turns evaluated:           {evaluated_turns}")
    print(f"NDCG@3:                    {results[nDCG @ 3]:.4f}")
    print(f"NDCG@10:                   {results[nDCG @ k]:.4f}")
    print(f"MRR@10:                    {results[RR @ k]:.4f}")
    print(
        f"Avg Time:                  {np.mean(times) if times else float('nan'):.2f} ms"
    )
    print(f"Centroids cached per conv: {H}")
    print(f"nprobe:                    {NP}")


# =================================================================
# EVAL 2 — TopLoc-IVF+
# =================================================================
def run_ivf_plus_eval():
    print("\n" + "=" * 60)
    print(f"Running TopLoc-IVF+ (H={H}, NP={NP}, ALPHA={ALPHA})...")
    print("=" * 60)

    conv_cache = {}
    k, warmup = 10, 5
    times = []
    evaluated_turns = 0
    refresh_count = 0
    run = defaultdict(dict)

    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        if q0_key not in filtered_qrels:
            continue

        # Turn 0 — full search + build cache (vecs + top_0 saved)
        q0_emb = encode_query(topics[q0_key])
        start = time.perf_counter()
        _, c0_indices = ivf_index.quantizer.search(q0_emb, H)

        c0_vecs = get_centroid_vectors(ivf_index.quantizer, c0_indices[0])
        top_0_local, _ = rank_within_cache(c0_vecs, q0_emb, NP, USE_IP)

        conv_cache[conv_id] = {
            "c0": c0_indices[0].astype("int64"),
            "q0_emb": q0_emb,
            "c0_vecs": c0_vecs,
            "top_0_local": top_0_local,
        }

        scores, indices = base_index.search(q0_emb, k)
        end = time.perf_counter()

        if evaluated_turns >= warmup:
            times.append((end - start) * 1000)

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            pid = id_map.get(str(idx))
            if pid is not None:
                run[q0_key][pid] = float(score)
        evaluated_turns += 1

        # Turns 1+ — drift check + C++ restricted search
        for turn_key in turns[1:]:
            if turn_key not in filtered_qrels:
                continue

            qj_emb = encode_query(topics[turn_key])
            start = time.perf_counter()
            cache = conv_cache[conv_id]

            c0_vecs = cache["c0_vecs"]
            top_0_local = cache["top_0_local"]

            top_j_local, _ = rank_within_cache(c0_vecs, qj_emb, NP, USE_IP)
            i0_size = len(np.intersect1d(top_j_local, top_0_local))

            if i0_size < ALPHA * NP:
                _, new_c0 = ivf_index.quantizer.search(qj_emb, H)
                new_c0_vecs = get_centroid_vectors(ivf_index.quantizer, new_c0[0])
                new_top_0, _ = rank_within_cache(new_c0_vecs, qj_emb, NP, USE_IP)
                cache = {
                    "c0": new_c0[0].astype("int64"),
                    "q0_emb": qj_emb,
                    "c0_vecs": new_c0_vecs,
                    "top_0_local": new_top_0,
                }
                conv_cache[conv_id] = cache
                refresh_count += 1

            scores, indices = toploc_ivf_search(ivf_index, qj_emb, cache["c0"], NP, k)
            end = time.perf_counter()

            if evaluated_turns >= warmup:
                times.append((end - start) * 1000)

            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                pid = id_map.get(str(idx))
                if pid is not None:
                    run[turn_key][pid] = float(score)
            evaluated_turns += 1

    measures = [nDCG @ 3, nDCG @ k, RR @ k]
    results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))

    print(f"Turns evaluated:           {evaluated_turns}")
    print(f"NDCG@3:                    {results[nDCG @ 3]:.4f}")
    print(f"NDCG@10:                   {results[nDCG @ k]:.4f}")
    print(f"MRR@10:                    {results[RR @ k]:.4f}")
    print(
        f"Avg Time:                  {np.mean(times) if times else float('nan'):.2f} ms"
    )
    print(f"Centroids cached per conv: {H}")
    print(f"nprobe:                    {NP}")
    print(f"Alpha threshold (alpha):   {ALPHA}")
    print(f"Cache refreshes triggered: {refresh_count}")


# =================================================================
# RUN BOTH
# =================================================================
run_ivf_eval()
run_ivf_plus_eval()
