#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import faiss
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR

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

USE_MMAP = os.environ.get("MMAP", "0") == "1"

# ================= PARAMETER GRID =================
H_VALUES = [512, 1024, 4096, 8192]
NP_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


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


# ================= TOPLOC IVF SEARCH =================
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


def toploc_ivf_search(index, q_emb, cached_centroid_indices, nprobe, k):
    centroid_vecs = get_centroid_vectors(index.quantizer, cached_centroid_indices)
    use_ip = index.metric_type == faiss.METRIC_INNER_PRODUCT
    if use_ip:
        coarse = (centroid_vecs @ q_emb.T).squeeze(axis=1)
        top_local = np.argpartition(-coarse, min(nprobe, len(coarse) - 1))[:nprobe]
        top_local = top_local[np.argsort(-coarse[top_local])]
    else:
        coarse = ((centroid_vecs - q_emb) ** 2).sum(axis=1)
        top_local = np.argpartition(coarse, min(nprobe, len(coarse) - 1))[:nprobe]
        top_local = top_local[np.argsort(coarse[top_local])]
    sel_centroids = (
        np.asarray(cached_centroid_indices)[top_local].astype("int64").reshape(1, -1)
    )
    sel_coarse = coarse[top_local].astype("float32").reshape(1, -1)
    try:
        scores, indices = index.search_preassigned(q_emb, k, sel_centroids, sel_coarse)
    except TypeError:
        scores, indices = index.search_preassigned(q_emb, k, sel_centroids)
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


# =================================================================
# LOAD EVERYTHING ONCE
# =================================================================
print(f"Model: {model_name} | Index: {index_type}")
print("=" * 60)
print("Loading shared resources (index, encoder, topics, qrels)...")

index_path = os.path.join(cache_dir, f"{index_type}_index.index")
if USE_MMAP:
    print("  Loading index with mmap...")
    base_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
else:
    base_index = faiss.read_index(index_path)

ivf_index = faiss.extract_index_ivf(base_index)
try:
    ivf_index.make_direct_map()
except Exception:
    pass

print(
    f"  Index: ntotal={ivf_index.ntotal}, nlist={ivf_index.nlist}, "
    f"metric={'IP' if ivf_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}"
)

ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())
print(f"  ID map: {len(id_map)} passages")

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
print(f"  Qrels: {len(filtered_qrels)} turns")

conversations = defaultdict(list)
for turn_key in topics:
    conv_id = turn_key.split("_")[0]
    conversations[conv_id].append(turn_key)
print(f"  Conversations: {len(conversations)} ({len(topics)} turns)")

print(f"  Loading {model_name} query encoder...")
encode_query = load_query_encoder(model_name)
print("Shared resources loaded.\n")


# =================================================================
# SINGLE EVAL FUNCTION
# =================================================================
def run_eval(H, NP):
    ivf_index.nprobe = NP
    h = min(H, ivf_index.nlist)

    conv_cache = {}
    k, warmup = 10, 5
    times = []
    evaluated_turns = 0
    run = defaultdict(dict)

    for conv_id, turns in conversations.items():
        q0_key = turns[0]
        if q0_key not in filtered_qrels:
            continue

        q0_emb = encode_query(topics[q0_key])
        start = time.perf_counter()
        _, c0_indices = ivf_index.quantizer.search(q0_emb, h)
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

    return {
        "H": h,
        "NP": NP,
        "turns": evaluated_turns,
        "ndcg3": results[nDCG @ 3],
        "ndcg10": results[nDCG @ k],
        "mrr10": results[RR @ k],
        "avg_time_ms": np.mean(times) if times else float("nan"),
    }


# =================================================================
# PARAMETER SWEEP
# =================================================================
total_runs = len(H_VALUES) * len(NP_VALUES)
print(
    f"Starting sweep: {len(H_VALUES)} H values x "
    f"{len(NP_VALUES)} NP values = {total_runs} runs\n"
)

all_results = []
run_num = 0

for H in H_VALUES:
    for NP in NP_VALUES:
        run_num += 1
        print(f"[{run_num}/{total_runs}] H={H}, NP={NP} ...", end=" ", flush=True)
        r = run_eval(H, NP)
        all_results.append(r)
        print(
            f"NDCG@10={r['ndcg10']:.4f}  MRR@10={r['mrr10']:.4f}  "
            f"Time={r['avg_time_ms']:.2f}ms"
        )

# =================================================================
# RESULTS TABLE
# =================================================================
print("\n" + "=" * 75)
print(f"TOPLOC-IVF SWEEP RESULTS ({index_type.upper()}, {model_name})")
print("=" * 75)
print(
    f"{'H':>6}  {'NP':>4}  {'NDCG@3':>8}  {'NDCG@10':>8}  "
    f"{'MRR@10':>8}  {'Time(ms)':>10}"
)
print("-" * 75)
for r in all_results:
    print(
        f"{r['H']:>6}  {r['NP']:>4}  {r['ndcg3']:>8.4f}  "
        f"{r['ndcg10']:>8.4f}  {r['mrr10']:>8.4f}  "
        f"{r['avg_time_ms']:>10.2f}"
    )

best = max(all_results, key=lambda x: x["ndcg10"])
print("=" * 75)
print(
    f"BEST NDCG@10: {best['ndcg10']:.4f}  "
    f"->  H={best['H']}, NP={best['NP']}, "
    f"Time={best['avg_time_ms']:.2f}ms"
)
print("=" * 75)
