import os
import sys
import time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import ir_measures
from ir_measures import nDCG, RR

# ================= CONFIGURATION =================
CACHE_BASE = "/home/toploc2/Datasets/toploc2"
DATASET_DIR = "/home/toploc2/Datasets/conversational/CAST2019/topics"

CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

BATCH_WARMUP_RUNS = 2
BATCH_TIMED_RUNS = 5

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
index_type = sys.argv[2] if len(sys.argv) > 2 else "ivf"
cache_dir = CACHE_DIRS[model_name]

USE_MMAP = os.environ.get("MMAP", "0") == "1"
faiss.omp_set_num_threads(os.cpu_count() or 1)


# ================= TOPLOC IVF IMPLEMENTATION =================
class TopLocIVF:
    def __init__(self, index, h=4096, nprobe=128):
        self.top_index = index  # Retain for fallback

        if not isinstance(index, faiss.IndexIVF):
            self.ivf_index = faiss.downcast_index(index.index)
        else:
            self.ivf_index = faiss.downcast_index(index)

        self.quantizer = faiss.downcast_index(self.ivf_index.quantizer)
        self.d = self.ivf_index.d
        self.h = h
        self.nprobe = nprobe
        self.ivf_index.nprobe = nprobe

        # Safely extract all centroid vectors
        try:
            self.all_centroids = faiss.vector_to_array(self.quantizer.xb).reshape(
                -1, self.d
            )
        except Exception:
            self.all_centroids = self.quantizer.reconstruct_n(0, self.ivf_index.nlist)

        self.cache = {}

    def search_first_turn(self, q_emb, conv_id, k):
        # Ensure memory is contiguous to prevent FAISS copying overhead
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        # 1. Search full quantizer for top h centroids
        coarse_dis, coarse_assign = self.quantizer.search(q_emb, self.h)

        C0_ids = coarse_assign[0]
        C0_vecs = np.ascontiguousarray(self.all_centroids[C0_ids], dtype=np.float32)

        # 2. Build a native FAISS C++ index for the cache.
        # This completely eliminates NumPy overhead and handles exact sorting natively.
        c0_index = faiss.IndexFlatIP(self.d)
        c0_index.add(C0_vecs)

        self.cache[conv_id] = {
            "C0_ids": C0_ids,
            "c0_index": c0_index,
        }

        # 3. Use top nprobe for the actual preassigned search
        assign = np.ascontiguousarray(coarse_assign[:, : self.nprobe], dtype=np.int64)
        c_dis = np.ascontiguousarray(coarse_dis[:, : self.nprobe], dtype=np.float32)

        distances, labels = self.ivf_index.search_preassigned(q_emb, k, assign, c_dis)
        return distances, labels

    def search_followup(self, q_batch, conv_id, k):
        cached = self.cache.get(conv_id)
        if cached is None:
            return self.top_index.search(q_batch, k)

        q_batch = np.ascontiguousarray(q_batch, dtype=np.float32)

        # 1. Bypass the massive quantizer matrix. Search ONLY the h cached centroids using C++
        c_dis, sub_assign = cached["c0_index"].search(q_batch, self.nprobe)

        # 2. Map local 0-to-h index back to global centroid IDs
        assign = np.ascontiguousarray(cached["C0_ids"][sub_assign], dtype=np.int64)
        c_dis = np.ascontiguousarray(c_dis, dtype=np.float32)

        # 3. Preassigned FAISS scan
        distances, labels = self.ivf_index.search_preassigned(q_batch, k, assign, c_dis)

        return distances, labels


# ================= QUERY ENCODER =================
def load_query_encoder(model_name):
    if model_name == "snowflake":
        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode_batch(queries):
            return model.encode(
                queries,
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
            ).astype("float32")

        return encode_batch
    elif model_name == "dragon":
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


# ================= LOAD DATA =================
print(f"Evaluating TopLoc IVF for: {model_name} ({index_type})")

index_path = os.path.join(cache_dir, f"{index_type}_index.index")
index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP if USE_MMAP else 0)

ids_path = os.path.join(cache_dir, f"{index_type}_ids.npy")
id_array = np.load(ids_path, allow_pickle=True)
id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
indexed_pids = set(id_map.values())

topics = {}
with open(os.path.join(DATASET_DIR, "topics.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",", 1)
        if len(parts) == 2:
            topics[parts[0].strip()] = parts[1].strip()

qrels = defaultdict(dict)
with open(os.path.join(DATASET_DIR, "qrels.qrel"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 4 and int(parts[3]) > 0:
            qrels[parts[0].strip()][parts[2].strip()] = int(parts[3])

filtered_qrels = {
    k: {pid: s for pid, s in v.items() if pid in indexed_pids}
    for k, v in qrels.items()
    if any(pid in indexed_pids for pid in v)
}
eval_keys = [k for k in topics if k in filtered_qrels]
eval_queries = [topics[k] for k in eval_keys]
N = len(eval_queries)

print(f"Loading {model_name} query encoder...")
encode_batch = load_query_encoder(model_name)
query_matrix = encode_batch(eval_queries)

conv_rows = defaultdict(list)
for row, key in enumerate(eval_keys):
    conv_rows[key.split("_")[0]].append(row)

first_rows = [rows[0] for rows in conv_rows.values()]
followup_rows_per_conv = [rows[1:] for rows in conv_rows.values()]
first_n = len(first_rows)
followup_n = sum(len(r) for r in followup_rows_per_conv)

k = 10

# ================= WRAP INDEX IN TOPLOC =================
# Set h=4096 per the paper's grid search to ensure we maintain exact NDCG bounds
toploc = TopLocIVF(index, h=4096, nprobe=128)

# ================= BUILD RUN (untimed) =================
print("\nBuilding run dict for metrics...")
run = defaultdict(dict)
toploc.cache.clear()

for q0_row, fu_rows in zip(first_rows, followup_rows_per_conv):
    conv_id = eval_keys[q0_row].split("_")[0]
    turn_key = eval_keys[q0_row]

    D, I = toploc.search_first_turn(query_matrix[q0_row : q0_row + 1], conv_id, k)
    for idx, score in zip(I[0], D[0]):
        if idx >= 0 and id_map.get(str(idx)):
            run[turn_key][id_map[str(idx)]] = float(score)

    if fu_rows:
        D_fu, I_fu = toploc.search_followup(query_matrix[fu_rows], conv_id, k)
        for local_i, row in enumerate(fu_rows):
            turn_key = eval_keys[row]
            for idx, score in zip(I_fu[local_i], D_fu[local_i]):
                if idx >= 0 and id_map.get(str(idx)):
                    run[turn_key][id_map[str(idx)]] = float(score)

measures = [nDCG @ 3, nDCG @ k, RR @ k]
results = ir_measures.calc_aggregate(measures, dict(filtered_qrels), dict(run))


# ================= TIMING =================
def timed_sweep():
    toploc.cache.clear()
    first_total_ms, followup_total_ms = 0.0, 0.0

    for q0_row, fu_rows in zip(first_rows, followup_rows_per_conv):
        conv_id = eval_keys[q0_row].split("_")[0]

        t0 = time.perf_counter()
        toploc.search_first_turn(query_matrix[q0_row : q0_row + 1], conv_id, k)
        first_total_ms += (time.perf_counter() - t0) * 1000

        if fu_rows:
            t0 = time.perf_counter()
            toploc.search_followup(query_matrix[fu_rows], conv_id, k)
            followup_total_ms += (time.perf_counter() - t0) * 1000

    return first_total_ms, followup_total_ms


def latency_stats(times_ms):
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


print(
    f"\nRunning {BATCH_WARMUP_RUNS} warmup + {BATCH_TIMED_RUNS} timed per-conversation sweeps..."
)
for _ in range(BATCH_WARMUP_RUNS):
    timed_sweep()

first_times, followup_times = [], []
for run_i in range(BATCH_TIMED_RUNS):
    f_ms, u_ms = timed_sweep()
    first_times.append(f_ms)
    followup_times.append(u_ms)
    f_pq = f_ms / first_n if first_n else float("nan")
    u_pq = u_ms / followup_n if followup_n else float("nan")
    print(
        f"  run {run_i + 1}: first-turn total={f_ms:7.2f} ms (per-query={f_pq:.3f}) | follow-up total={u_ms:7.2f} ms (per-query={u_pq:.3f})"
    )

# ================= RESULTS =================
f_min, f_med, f_mean = latency_stats(first_times)
u_min, u_med, u_mean = latency_stats(followup_times)
overall_times = [f + u for f, u in zip(first_times, followup_times)]
o_min, o_med, o_mean = latency_stats(overall_times)


def per_query_line(stats, n):
    if not n:
        return "n/a"
    return f"{stats[0] / n:8.3f}  /  {stats[1] / n:8.3f}  /  {stats[2] / n:8.3f}  ms"


print("\n" + "=" * 70)
print(f"TOPLOC IVF EVALUATION RESULTS ({model_name})")
print("=" * 70)
print(f"NDCG@3:  {results[nDCG @ 3]:.4f}")
print(f"NDCG@10: {results[nDCG @ k]:.4f}")
print(f"MRR@10:  {results[RR @ k]:.4f}")
print()
print(f"Latency (min / median / mean over {BATCH_TIMED_RUNS} sweeps):")
print(f"  first-turn per query: {per_query_line((f_min, f_med, f_mean), first_n)}")
print(f"  follow-up  per query: {per_query_line((u_min, u_med, u_mean), followup_n)}")
print(f"  overall    per query: {per_query_line((o_min, o_med, o_mean), N)}")
print("=" * 70)
