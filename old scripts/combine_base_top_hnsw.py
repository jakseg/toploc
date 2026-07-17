#!/usr/bin/env python3
"""
combine_base_top_hnsw.py
========================
Baseline HNSW + TopLoc-HNSW in ONE script.

Purpose
-------
This is the HNSW analogue of combine_base_top_ivf.py:
  * load the huge FAISS HNSW index once
  * load/encode all evaluation queries once
  * run baseline HNSW and TopLoc-HNSW under the same process/settings
  * print side-by-side metrics and latency
  * optionally sweep efSearch values and write a CSV

TopLoc-HNSW logic
-----------------
For each conversation:
  1. q0 is searched with normal FAISS HNSW, using efSearch * up.
  2. The best q0 result(s) become privileged entry point(s).
  3. Follow-up turns are searched with a custom level-0 beam search seeded from
     the privileged entry point(s).

Important
---------
Real latency needs the C++ pybind11 module `toploc_hnsw_search`.
Build it first, for example:

    mkdir -p build_hnsw_release
    cd build_hnsw_release
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j
    cd ..

Then run with PYTHONPATH pointing to the build directory:

    PYTHONPATH=build_hnsw_release \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -u combine_base_top_hnsw.py snowflake hnsw --threads 1

Examples
--------
Single efSearch, default EF_SEARCH env var or 64:
    python -u combine_base_top_hnsw.py snowflake hnsw --threads 1

Single efSearch explicitly:
    python -u combine_base_top_hnsw.py snowflake hnsw --ef-search 128 --up 2 --threads 1

Sweep:
    python -u combine_base_top_hnsw.py snowflake hnsw --sweep --threads 1

Custom sweep:
    python -u combine_base_top_hnsw.py snowflake hnsw --sweep 16,32,64,128 --threads 1

Debug small run:
    python -u combine_base_top_hnsw.py snowflake hnsw --max-turns 20 --threads 1
"""

from __future__ import annotations

import argparse
import csv
import heapq
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import ir_measures
import numpy as np
from ir_measures import RR, nDCG


# ================= CONFIGURATION =================
CACHE_BASE = os.environ.get("CACHE_BASE", "/home/toploc1/Datasets/toploc2")
DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/home/toploc1/Datasets/conversational/CAST2019/topics"
)
CACHE_DIRS = {
    "snowflake": os.path.join(CACHE_BASE, "snowflake"),
    "dragon": os.path.join(CACHE_BASE, "dragon"),
}

USE_MMAP = os.environ.get("MMAP", "0") == "1"
BATCH_WARMUP_RUNS = int(os.environ.get("BATCH_WARMUP_RUNS", 2))
BATCH_TIMED_RUNS = int(os.environ.get("BATCH_TIMED_RUNS", 5))


# ================= OPTIONAL C++ BACKEND =================
def import_cpp_backend():
    try:
        from toploc_hnsw_search import toploc_hnsw_level0_search_ptr

        return toploc_hnsw_level0_search_ptr
    except ImportError as exc:
        raise ImportError(
            "Could not import C++ module `toploc_hnsw_search`.\n"
            "Build it first and set PYTHONPATH, for example:\n"
            "  mkdir -p build_hnsw_release && cd build_hnsw_release\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && cd ..\n"
            "  PYTHONPATH=build_hnsw_release python -u combine_base_top_hnsw.py snowflake hnsw --threads 1\n"
            "For correctness-only/debugging, you can run with: --backend python"
        ) from exc


def faiss_index_ptr(index) -> int:
    """Return raw C++ pointer from a FAISS SWIG Python object."""
    try:
        return int(index.this)
    except Exception:
        return int(faiss.downcast_index(index).this)


# ================= QUERY ENCODER =================
def load_query_encoder(model_name: str):
    """Return a function: list[str] -> np.ndarray of shape (N, dim)."""
    if model_name == "snowflake":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

        def encode_batch(queries: Sequence[str]) -> np.ndarray:
            return model.encode(
                list(queries),
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=True,
            ).astype("float32")

        return encode_batch

    if model_name == "dragon":
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
        model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
        model.eval()

        def encode_batch(queries: Sequence[str], chunk: int = 32) -> np.ndarray:
            outs = []
            for i in range(0, len(queries), chunk):
                batch = list(queries[i : i + chunk])
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = model(**tokens)
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.append(emb.cpu().numpy())
            return np.vstack(outs).astype("float32")

        return encode_batch

    raise ValueError(f"Unknown model: {model_name}")


def load_precomputed_query_embeddings(model_name: str, keys: Sequence[str]) -> Optional[np.ndarray]:
    """Try to load precomputed topic embeddings aligned to `keys`.

    Returns an (N, dim) float32 matrix aligned to keys on full success,
    otherwise None.
    """
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    emb_path = os.path.join(DATASET_DIR, f"topics_{model_name}_embeddings.parquet")
    if not os.path.exists(emb_path):
        return None

    table = pq.read_table(emb_path)
    cols = table.column_names
    id_col = next((c for c in ("id", "qid", "turn_id", "topic_id", "tid") if c in cols), None)
    emb_col = next((c for c in ("embedding", "embeddings", "vector", "emb") if c in cols), None)
    if id_col is None or emb_col is None:
        print(
            f"  WARN: {os.path.basename(emb_path)} has columns {cols}; "
            "could not find id/embedding columns — encoding instead.",
            flush=True,
        )
        return None

    emb_map = {
        str(i): e
        for i, e in zip(table.column(id_col).to_pylist(), table.column(emb_col).to_pylist())
    }
    missing = [k for k in keys if k not in emb_map]
    if missing:
        print(
            f"  WARN: {len(missing)}/{len(keys)} eval turns missing from "
            f"{os.path.basename(emb_path)} — encoding instead.",
            flush=True,
        )
        return None

    matrix = np.ascontiguousarray([emb_map[k] for k in keys], dtype="float32")
    faiss.normalize_L2(matrix)
    print(f"  Loaded precomputed query embeddings {matrix.shape} from {os.path.basename(emb_path)}")
    return matrix


# ================= FILE LOADERS =================
def load_topics(path: str) -> Dict[str, str]:
    """Load topics.tsv. In this project it is comma-separated: turn_id,query."""
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


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels.qrel. In this project it is comma-separated: qid,iter,pid,score."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            qid, _, pid, score = parts
            qid = qid.strip()
            pid = pid.strip()
            try:
                score_int = int(score)
            except ValueError:
                continue
            if score_int > 0:
                qrels[qid][pid] = score_int
    return qrels


def turn_sort_key(turn_key: str) -> Tuple[int, ...]:
    """Sort keys like '31_1', '31_2', ... in conversation order."""
    numbers = re.findall(r"\d+", turn_key)
    return tuple(int(x) for x in numbers) if numbers else (10**9,)


# ================= HNSW GRAPH ACCESS =================
def load_hnsw_level0_graph(index):
    hnsw = index.hnsw
    offsets = faiss.vector_to_array(hnsw.offsets).astype("int64", copy=False)
    neighbors = faiss.vector_to_array(hnsw.neighbors).astype("int64", copy=False)
    degree0 = int(hnsw.nb_neighbors(0))
    return offsets, neighbors, degree0


def level0_neighbors(graph, node_id: int) -> np.ndarray:
    offsets, neighbors, degree0 = graph
    start = int(offsets[int(node_id)])
    block = neighbors[start : start + degree0]
    return block[block >= 0]


def reconstruct_batch_safe(index, node_ids: Sequence[int]) -> np.ndarray:
    ids = np.asarray(node_ids, dtype="int64")
    try:
        return index.reconstruct_batch(ids).astype("float32")
    except Exception:
        vecs = np.empty((len(ids), index.d), dtype="float32")
        for i, node_id in enumerate(ids):
            index.reconstruct(int(node_id), vecs[i])
        return vecs


# ================= TOPLOC-HNSW SEARCH BACKENDS =================
def cpp_level0_search(cpp_func, index_ptr, graph, q_embs, entry_points, k: int, ef_search: int):
    offsets, neighbors, degree0 = graph
    q_c = np.ascontiguousarray(q_embs, dtype="float32")
    ep_c = np.ascontiguousarray(entry_points, dtype="int64")
    off_c = np.ascontiguousarray(offsets, dtype="int64")
    nei_c = np.ascontiguousarray(neighbors, dtype="int64")
    return cpp_func(index_ptr, q_c, ep_c, off_c, nei_c, degree0, k, ef_search)


def python_level0_search_one(index, graph, q_emb, entry_points, k: int, ef_search: int):
    """Pure-Python fallback for correctness/debug only. Slow."""
    query_vec = q_emb.reshape(-1).astype("float32")
    candidates = []  # (-score, node_id)
    results = []  # (score, node_id); min heap
    visited = set()

    def add_candidates(node_ids: Iterable[int]):
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
    visited_counts = np.array([len(visited)], dtype="int64")
    return scores, indices, visited_counts


def python_level0_search_batch(index, graph, q_embs, entry_points, k: int, ef_search: int):
    scores_all, indices_all, visited_all = [], [], []
    for i in range(q_embs.shape[0]):
        s, ids, v = python_level0_search_one(index, graph, q_embs[i : i + 1], entry_points, k, ef_search)
        scores_all.append(s[0])
        indices_all.append(ids[0])
        visited_all.append(v[0])
    return (
        np.asarray(scores_all, dtype="float32"),
        np.asarray(indices_all, dtype="int64"),
        np.asarray(visited_all, dtype="int64"),
    )


# ================= UTILS =================
def latency_stats(times_ms: Sequence[float]) -> Tuple[float, float, float]:
    if not times_ms:
        return float("nan"), float("nan"), float("nan")
    return min(times_ms), float(np.median(times_ms)), float(np.mean(times_ms))


def add_to_run(run, turn_key: str, scores_row, indices_row, id_map: Dict[str, str]):
    for idx, score in zip(indices_row, scores_row):
        idx_int = int(idx)
        if idx_int < 0:
            continue
        pid = id_map.get(str(idx_int))
        if pid is not None:
            run[turn_key][pid] = float(score)


def metric_values(metrics, k: int):
    return metrics[nDCG @ 3], metrics[nDCG @ k], metrics[RR @ k]


# ================= EVALUATION OBJECT =================
class HnswCombinedEvaluator:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        self.index_type = args.index_type
        self.cache_dir = CACHE_DIRS[self.model_name]
        self.index_path = os.path.join(self.cache_dir, f"{self.index_type}_index.index")
        self.ids_path = os.path.join(self.cache_dir, f"{self.index_type}_ids.npy")
        self.topics_path = os.path.join(DATASET_DIR, "topics.tsv")
        self.qrels_path = os.path.join(DATASET_DIR, "qrels.qrel")
        self.k = args.k
        self.cpp_func = None
        self.index_ptr = None

        if self.index_type != "hnsw":
            raise ValueError("This script is for HNSW only. Use index_type='hnsw'.")

        if args.backend == "cpp":
            self.cpp_func = import_cpp_backend()

        faiss.omp_set_num_threads(args.threads)

        print(
            f"Evaluating {self.model_name} ({self.index_type}) | "
            f"backend={args.backend} | mode={'SWEEP' if args.sweep_values else 'SINGLE'}",
            flush=True,
        )
        print(f"FAISS threads: {faiss.omp_get_max_threads()}", flush=True)
        print("Loading HNSW index once...", flush=True)
        t0 = time.perf_counter()
        if USE_MMAP:
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
        else:
            self.index = faiss.read_index(self.index_path)
        if not hasattr(self.index, "hnsw"):
            raise TypeError("Loaded index is not an HNSW index.")
        if self.args.backend == "cpp":
            self.index_ptr = faiss_index_ptr(self.index)
        print(
            f"Index loaded in {time.perf_counter() - t0:.1f}s: "
            f"ntotal={self.index.ntotal:,}, dim={self.index.d}, mmap={USE_MMAP}",
            flush=True,
        )

        print("Loading HNSW level-0 graph arrays once...", flush=True)
        self.graph = load_hnsw_level0_graph(self.index)
        print(f"Level-0 degree slots: {self.graph[2]}", flush=True)

        id_array = np.load(self.ids_path, allow_pickle=True)
        self.id_map = {str(i): str(pid) for i, pid in enumerate(id_array)}
        indexed_pids = set(self.id_map.values())
        print(f"ID map loaded: {len(self.id_map):,} passages", flush=True)

        self.topics = load_topics(self.topics_path)
        qrels = load_qrels(self.qrels_path)
        self.filtered_qrels = {
            qid: {pid: s for pid, s in pid_scores.items() if pid in indexed_pids}
            for qid, pid_scores in qrels.items()
        }
        self.filtered_qrels = {qid: vals for qid, vals in self.filtered_qrels.items() if vals}
        if not self.topics:
            raise RuntimeError("Parsed 0 topics. Check topics.tsv delimiter/format.")
        if not self.filtered_qrels:
            raise RuntimeError("No qrels survived filtering against indexed passage ids.")

        print(f"Topics loaded: {len(self.topics):,} turns", flush=True)
        print(f"Turns with relevant passages in index: {len(self.filtered_qrels):,}/{len(qrels):,}", flush=True)

        self._prepare_eval_keys_and_embeddings()

    def _prepare_eval_keys_and_embeddings(self):
        conversations = defaultdict(list)
        for turn_key in self.topics:
            conversations[turn_key.split("_")[0]].append(turn_key)
        for conv_id in conversations:
            conversations[conv_id].sort(key=turn_sort_key)

        # Same eval-set style as the baseline: only judged turns whose qrels survive filtering.
        eval_keys_all = [k0 for k0 in self.topics if k0 in self.filtered_qrels]
        if self.args.max_turns and self.args.max_turns > 0:
            eval_key_set = set(eval_keys_all[: self.args.max_turns])
        else:
            eval_key_set = set(eval_keys_all)

        self.conv_valid = {}
        self.eval_keys = []
        self.eval_queries = []
        for conv_id, turns in conversations.items():
            valid = [t for t in turns if t in self.filtered_qrels and t in eval_key_set]
            if not valid:
                continue
            self.conv_valid[conv_id] = valid
            for t in valid:
                self.eval_keys.append(t)
                self.eval_queries.append(self.topics[t])

        N = len(self.eval_keys)
        print(f"Eval set: {N:,} turns across {len(self.conv_valid):,} conversations", flush=True)
        print(f"\nObtaining query embeddings for {N:,} queries once...", flush=True)
        query_matrix = load_precomputed_query_embeddings(self.model_name, self.eval_keys)
        if query_matrix is None:
            print(f"Loading {self.model_name} query encoder...", flush=True)
            encode_batch = load_query_encoder(self.model_name)
            t0 = time.perf_counter()
            query_matrix = encode_batch(self.eval_queries)
            enc_ms = (time.perf_counter() - t0) * 1000
            print(
                f"Query encoding done in {enc_ms:.1f} ms total "
                f"({enc_ms / max(N, 1):.2f} ms/query) — shape={query_matrix.shape}",
                flush=True,
            )

        self.query_matrix = np.ascontiguousarray(query_matrix, dtype="float32")
        self.emb_lookup = {k0: self.query_matrix[i : i + 1] for i, k0 in enumerate(self.eval_keys)}

    def _toploc_search(self, q_embs: np.ndarray, entry_points: np.ndarray, ef_search: int):
        if self.args.backend == "cpp":
            return cpp_level0_search(
                self.cpp_func,
                self.index_ptr,
                self.graph,
                q_embs,
                entry_points,
                self.k,
                ef_search,
            )
        return python_level0_search_batch(self.index, self.graph, q_embs, entry_points, self.k, ef_search)

    def eval_baseline(self, ef_search: int):
        self.index.hnsw.efSearch = ef_search

        # Metrics: one batched native FAISS search over all eval turns.
        run = defaultdict(dict)
        all_scores, all_indices = self.index.search(self.query_matrix, self.k)
        for row, turn_key in enumerate(self.eval_keys):
            add_to_run(run, turn_key, all_scores[row], all_indices[row], self.id_map)
        metrics = ir_measures.calc_aggregate(
            [nDCG @ 3, nDCG @ self.k, RR @ self.k],
            dict(self.filtered_qrels),
            dict(run),
        )

        # Timing: same per-conversation style as TopLoc and IVF combined script.
        conv_rows = []
        key_to_row = {k0: i for i, k0 in enumerate(self.eval_keys)}
        for conv_id, valid_turns in self.conv_valid.items():
            rows = [key_to_row[t] for t in valid_turns]
            conv_rows.append(rows)
        first_n = len(conv_rows)
        followup_n = sum(max(0, len(rows) - 1) for rows in conv_rows)

        def sweep():
            first_ms = 0.0
            followup_ms = 0.0
            for rows in conv_rows:
                q0_row = rows[0]
                t0 = time.perf_counter()
                self.index.search(self.query_matrix[q0_row : q0_row + 1], self.k)
                first_ms += (time.perf_counter() - t0) * 1000
                if len(rows) > 1:
                    fu_rows = rows[1:]
                    t0 = time.perf_counter()
                    self.index.search(self.query_matrix[fu_rows], self.k)
                    followup_ms += (time.perf_counter() - t0) * 1000
            return first_ms, followup_ms

        for _ in range(BATCH_WARMUP_RUNS):
            sweep()
        first_times, followup_times = [], []
        for _ in range(BATCH_TIMED_RUNS):
            f_ms, u_ms = sweep()
            first_times.append(f_ms)
            followup_times.append(u_ms)

        f_mean = latency_stats(first_times)[2]
        u_mean = latency_stats(followup_times)[2]
        fu_pq = u_mean / followup_n if followup_n else float("nan")
        overall_pq = (f_mean + u_mean) / len(self.eval_keys) if self.eval_keys else float("nan")
        return metrics, fu_pq, overall_pq, first_n, followup_n

    def eval_toploc(self, ef_search: int, up: int, entry_points_n: int):
        k = self.k
        old_ef = self.index.hnsw.efSearch

        # Build TopLoc run + cache structures for timing.
        run = defaultdict(dict)
        conv_q0_emb = {}
        conv_entry_points = {}
        conv_fu_embs = {}
        conv_fu_keys = {}
        visited_counts = []

        for conv_i, (conv_id, valid_turns) in enumerate(self.conv_valid.items(), start=1):
            q0_key = valid_turns[0]
            q0_emb = self.emb_lookup[q0_key]

            self.index.hnsw.efSearch = ef_search * up
            q0_scores, q0_indices = self.index.search(q0_emb, max(k, entry_points_n))
            self.index.hnsw.efSearch = old_ef

            eps = np.asarray(
                [int(x) for x in q0_indices[0][:entry_points_n] if int(x) >= 0],
                dtype="int64",
            )
            if len(eps) == 0:
                continue

            conv_q0_emb[conv_id] = q0_emb
            conv_entry_points[conv_id] = eps
            add_to_run(run, q0_key, q0_scores[0][:k], q0_indices[0][:k], self.id_map)

            followup_keys = valid_turns[1:]
            if followup_keys:
                fu_embs = np.vstack([self.emb_lookup[t] for t in followup_keys]).astype("float32")
                conv_fu_embs[conv_id] = fu_embs
                conv_fu_keys[conv_id] = followup_keys

                scores, indices, visited = self._toploc_search(fu_embs, eps, ef_search)
                visited_counts.extend([int(x) for x in np.asarray(visited)])
                for row, turn_key in enumerate(followup_keys):
                    add_to_run(run, turn_key, scores[row], indices[row], self.id_map)

            if self.args.progress and (conv_i % 5 == 0 or conv_i == len(self.conv_valid)):
                print(f"  TopLoc run build: conv {conv_i}/{len(self.conv_valid)}", flush=True)

        self.index.hnsw.efSearch = old_ef
        first_n = len(conv_q0_emb)
        followup_n = sum(len(v) for v in conv_fu_keys.values())

        metrics = ir_measures.calc_aggregate(
            [nDCG @ 3, nDCG @ k, RR @ k],
            dict(self.filtered_qrels),
            dict(run),
        )

        def sweep():
            first_ms = 0.0
            followup_ms = 0.0
            old = self.index.hnsw.efSearch
            for conv_id in conv_q0_emb:
                q0_emb = conv_q0_emb[conv_id]
                self.index.hnsw.efSearch = ef_search * up
                t0 = time.perf_counter()
                self.index.search(q0_emb, max(k, entry_points_n))
                first_ms += (time.perf_counter() - t0) * 1000
                self.index.hnsw.efSearch = old

                if conv_id in conv_fu_embs:
                    t0 = time.perf_counter()
                    self._toploc_search(conv_fu_embs[conv_id], conv_entry_points[conv_id], ef_search)
                    followup_ms += (time.perf_counter() - t0) * 1000
            self.index.hnsw.efSearch = old
            return first_ms, followup_ms

        for _ in range(BATCH_WARMUP_RUNS):
            sweep()
        first_times, followup_times = [], []
        for _ in range(BATCH_TIMED_RUNS):
            f_ms, u_ms = sweep()
            first_times.append(f_ms)
            followup_times.append(u_ms)

        f_mean = latency_stats(first_times)[2]
        u_mean = latency_stats(followup_times)[2]
        fu_pq = u_mean / followup_n if followup_n else float("nan")
        overall_pq = (f_mean + u_mean) / (first_n + followup_n) if (first_n + followup_n) else float("nan")
        avg_visited = float(np.mean(visited_counts)) if visited_counts else float("nan")
        return metrics, fu_pq, overall_pq, first_n, followup_n, avg_visited


# ================= CLI =================
def parse_args():
    parser = argparse.ArgumentParser(description="Combined baseline HNSW + TopLoc-HNSW evaluation")
    parser.add_argument("model", choices=["snowflake", "dragon"], nargs="?", default="snowflake")
    parser.add_argument("index_type", choices=["hnsw"], nargs="?", default="hnsw")
    parser.add_argument("--ef-search", type=int, default=int(os.environ.get("EF_SEARCH", 64)))
    parser.add_argument("--up", type=int, default=int(os.environ.get("UP", 2)))
    parser.add_argument(
        "--up-values",
        default=None,
        help="Optional comma-list for TopLoc-HNSW up sweep, e.g. --up-values 2,4,8,16",
    )
    parser.add_argument("--entry-points", type=int, default=int(os.environ.get("ENTRY_POINTS", 1)))
    parser.add_argument("--k", type=int, default=int(os.environ.get("K", 10)))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 1)))
    parser.add_argument("--max-turns", type=int, default=0, help="Debug limit. 0 means all judged turns.")
    parser.add_argument("--backend", choices=["cpp", "python"], default=os.environ.get("BACKEND", "cpp"))
    parser.add_argument("--progress", action="store_true", help="Print per-conversation TopLoc build progress.")
    parser.add_argument(
        "--sweep",
        nargs="?",
        const="1,2,4,8,16,32,64,128,256,512,1024",
        help="Sweep efSearch values. Optional comma-list, e.g. --sweep 16,32,64,128",
    )
    args = parser.parse_args()

    args.sweep_values = None
    if args.sweep is not None:
        args.sweep_values = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]

    args.up_values = None
    if args.up_values is not None:
        args.up_values = [int(x.strip()) for x in args.up_values.split(",") if x.strip()]

    return args


# ================= MAIN =================
def main():
    args = parse_args()
    evaluator = HnswCombinedEvaluator(args)

    # IMPORTANT:
    # The index and query embeddings are loaded/created once above.
    # These loops only run searches with different efSearch/up values.
    ef_values = args.sweep_values if args.sweep_values else [args.ef_search]
    up_values = args.up_values if args.up_values else [args.up]
    rows = []

    print("\n" + "=" * 130)
    header = (
        f"{'ef':>6} {'up':>5} | "
        f"{'base ms':>9} {'topl ms':>9} {'speedup':>8} | "
        f"{'b_NDCG3':>8} {'t_NDCG3':>8} | "
        f"{'b_NDCG10':>9} {'t_NDCG10':>9} | "
        f"{'b_MRR10':>8} {'t_MRR10':>8} | "
        f"{'visited':>8}"
    )
    print(header)
    print("-" * 130)

    # Baseline HNSW depends on efSearch, but not on up.
    # Therefore we compute baseline once per efSearch and reuse it for all up values.
    for ef in ef_values:
        b_metrics, b_fu_pq, b_overall, b_first_n, b_followup_n = evaluator.eval_baseline(ef)
        b_n3, b_n10, b_mrr = metric_values(b_metrics, args.k)

        for up in up_values:
            t_metrics, t_fu_pq, t_overall, t_first_n, t_followup_n, avg_visited = evaluator.eval_toploc(
                ef, up, args.entry_points
            )

            speedup = (
                b_fu_pq / t_fu_pq
                if np.isfinite(t_fu_pq) and t_fu_pq > 0
                else float("nan")
            )
            t_n3, t_n10, t_mrr = metric_values(t_metrics, args.k)

            print(
                f"{ef:>6} {up:>5} | "
                f"{b_fu_pq:>9.3f} {t_fu_pq:>9.3f} {speedup:>7.2f}x | "
                f"{b_n3:>8.4f} {t_n3:>8.4f} | "
                f"{b_n10:>9.4f} {t_n10:>9.4f} | "
                f"{b_mrr:>8.4f} {t_mrr:>8.4f} | "
                f"{avg_visited:>8.1f}",
                flush=True,
            )

            rows.append(
                {
                    "model": args.model,
                    "index": args.index_type,
                    "method_group": "hnsw",
                    "ef_search": ef,
                    "up": up,
                    "entry_points": args.entry_points,
                    "backend": args.backend,
                    "threads": args.threads,
                    "mmap": int(USE_MMAP),

                    "baseline_followup_ms_per_query": round(b_fu_pq, 4),
                    "toploc_followup_ms_per_query": round(t_fu_pq, 4),
                    "speedup_followup": round(speedup, 3),

                    "baseline_overall_ms_per_query": round(b_overall, 4),
                    "toploc_overall_ms_per_query": round(t_overall, 4),

                    "baseline_NDCG@3": round(b_n3, 4),
                    "toploc_NDCG@3": round(t_n3, 4),

                    "baseline_NDCG@10": round(b_n10, 4),
                    "toploc_NDCG@10": round(t_n10, 4),

                    "baseline_MRR@10": round(b_mrr, 4),
                    "toploc_MRR@10": round(t_mrr, 4),

                    "avg_visited_nodes_toploc": (
                        round(avg_visited, 2) if np.isfinite(avg_visited) else ""
                    ),

                    "baseline_first_turns": b_first_n,
                    "baseline_followup_turns": b_followup_n,
                    "toploc_first_turns": t_first_n,
                    "toploc_followup_turns": t_followup_n,
                }
            )

    print("=" * 130)
    print(
        f"\nef_values={ef_values} | up_values={up_values} | "
        f"entry_points={args.entry_points} | "
        f"warmup={BATCH_WARMUP_RUNS} timed={BATCH_TIMED_RUNS} | "
        f"threads={args.threads} | backend={args.backend} | mmap={int(USE_MMAP)}"
    )
    print("Times shown in the table are FOLLOW-UP per-query latency in ms.")
    print("Overall per-query latency is written to CSV.")

    # Write CSV when we are doing either efSearch sweep or up sweep.
    if args.sweep_values or args.up_values:
        os.makedirs("results/raw/hnsw", exist_ok=True)

        ef_tag = (
            f"ef{min(ef_values)}-{max(ef_values)}"
            if len(ef_values) > 1
            else f"ef{ef_values[0]}"
        )
        up_tag = (
            f"up{min(up_values)}-{max(up_values)}"
            if len(up_values) > 1
            else f"up{up_values[0]}"
        )

        out_csv = (
            f"results/raw/hnsw/"
            f"hnsw_{args.model}_{ef_tag}_{up_tag}_ep{args.entry_points}_mmap{int(USE_MMAP)}.csv"
        )

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nCSV written: {out_csv} ({len(rows)} rows)")

    if args.backend == "python":
        print("\nNOTE: --backend python is correctness/debug only. Use --backend cpp for real latency.")


if __name__ == "__main__":
    main()
