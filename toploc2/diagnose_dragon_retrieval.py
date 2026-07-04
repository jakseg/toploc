#!/usr/bin/env python3
"""Why is dragon NDCG/MRR/Accuracy@10 ~0 for --dataset msmarco-on-cast, while
routing looks sane? Two stages, split by RAM cost:

  --stage coverage (default, ~5 GB RAM, safe to run anytime):
    1. qrels / dev / groundtruth coverage - is N (eval queries with qrels)
       actually healthy, or did coverage silently collapse?
    2. hnsw_ids.npy sanity (length vs the known 38,636,446-passage count,
       mtime vs the index file) - a STALE ids file left over from before the
       dragon dot-product rebuild would map every retrieved internal node id
       to the WRONG pid string. Routing never touches pids, so this kind of
       mismatch would not show up there at all, only in NDCG/MRR/Accuracy.
       NOTE: this only catches a COUNT/staleness mismatch, not a reordering
       that preserves the count - the --stage search check below is the only
       way to rule that out for real.

  --stage search (needs the real HNSW index resident - see RAM note below):
    3. real retrieval spot check - run actual HNSW search for a handful of
       dev queries that DO have qrels, and check whether ANY relevant pid
       shows up in the top-100. If the run dict never contains a relevant
       pid, the problem is in query/doc embedding alignment or the id_map,
       not in QLR itself.

RAM NOTE:
  --stage coverage: ~6-9 GB (dominated by the 38.6M-string hnsw_ids.npy + a
    set() of it; the dev embeddings are a few MB). Safe with 110 GB free.
  --stage search:  faiss's IO_FLAG_MMAP does NOT reliably keep IndexHNSWFlat
    off the resident set - the graph + flat storage get read into RAM close
    to in full. So this needs ~the index file size (~129 GB) + the ids set
    (~7 GB) ~= 135-140 GB resident MINIMUM. On a shared box leave headroom;
    ~160-200 GB free is comfortable. Run `free -h` first. If tight, run only
    --stage coverage now - with the index file-size check added there it can
    already catch a wrong/partial index (the most likely cause given the
    absurdly low baseline latency) WITHOUT loading the index at all.

Run from inside toploc2/ (same directory as toploc2_hnsw_pure_python.py):
    python diagnose_dragon_retrieval.py                  # coverage only
    python diagnose_dragon_retrieval.py --stage search   # + real HNSW search
"""
import argparse
import os

import numpy as np

import toploc2_hnsw_pure_python as qlr

MODEL = "dragon"
DATASET = "msmarco-on-cast"
K = 10
EXPECTED_NTOTAL = 38_636_446  # CAST2019 collection size (see CLAUDE.md)
# Full HNSWFlat over 38.6M x 768 float32 ~ 118 GB flat + ~10 GB graph; CLAUDE.md
# records the built dragon index at 129 GB. Warn well below that.
EXPECTED_INDEX_GB = 129
MIN_INDEX_GB = 100

cache_dir = qlr.CACHE_DIRS[MODEL]
index_path = os.path.join(cache_dir, "hnsw_index.index")
ids_path = os.path.join(cache_dir, "hnsw_ids.npy")


def stage_coverage():
    print("=== 1. qrels / dev / groundtruth coverage ===", flush=True)
    qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
    id_array = np.load(ids_path, allow_pickle=True)
    indexed_pids = set(str(x) for x in id_array)

    filtered_qrels = {}
    for qid, pid_scores in qrels.items():
        valid = {p: s for p, s in pid_scores.items() if p in indexed_pids}
        if valid:
            filtered_qrels[qid] = valid
    print(f"qrels queries total: {len(qrels):,} | with >=1 indexed relevant pid: "
          f"{len(filtered_qrels):,}", flush=True)

    normalize = MODEL != "dragon"
    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, _ = qlr.load_parquet_embeddings(dev_dir, normalize=normalize)
    eval_keys = [qid for qid in q_ids if qid in filtered_qrels]
    print(f"dragon dev queries: {len(q_ids):,} | with qrels (N): {len(eval_keys):,}",
          flush=True)

    sf_dir = qlr.MSMARCO_DEV_QUERY_DIRS["snowflake"]
    sf_ids, _ = qlr.load_parquet_embeddings(sf_dir, normalize=True)
    if set(q_ids) == set(sf_ids):
        print(f"parity vs snowflake dev ids: IDENTICAL ({len(set(q_ids)):,} ids)", flush=True)
    else:
        print(f"parity vs snowflake dev ids: DIFFERS "
              f"(dragon-only={len(set(q_ids) - set(sf_ids))}, "
              f"snowflake-only={len(set(sf_ids) - set(q_ids))})", flush=True)

    gt_path = qlr.groundtruth_path(cache_dir, MODEL, DATASET, K)
    gt = qlr.load_groundtruth(gt_path)
    if gt is None:
        print(f"groundtruth file MISSING: {gt_path}", flush=True)
    else:
        n_gt = sum(1 for k in eval_keys if gt.get(k))
        print(f"groundtruth entries covering eval set: {n_gt:,}/{len(eval_keys):,}  "
              f"({gt_path})", flush=True)

    print("\n=== 2. hnsw_ids.npy + index-file sanity (no index load) ===", flush=True)
    ok_count = len(id_array) == EXPECTED_NTOTAL
    print(f"len(hnsw_ids.npy) = {len(id_array):,}  (expected {EXPECTED_NTOTAL:,}) "
          f"-> {'OK' if ok_count else 'MISMATCH!!'}", flush=True)

    idx_gb = os.path.getsize(index_path) / 1e9
    ids_gb = os.path.getsize(ids_path) / 1e9
    print(f"index file: {idx_gb:.1f} GB  mtime={os.path.getmtime(index_path)}", flush=True)
    print(f"ids file:   {ids_gb:.3f} GB  mtime={os.path.getmtime(ids_path)}", flush=True)

    # The absurd baseline latency in the CSV (0.03 ms/q vs snowflake's 0.39 ms/q)
    # is the loudest clue: it implies the loaded dragon index is NOT the full
    # 38.6M collection. A full HNSWFlat over 38.6M x 768 float32 is ~118 GB of
    # flat storage + ~10 GB graph -> ~129 GB on disk (see CLAUDE.md). If the file
    # is much smaller, the index is partial/wrong and every metric collapses
    # regardless of QLR - catchable here WITHOUT loading 129 GB into RAM.
    if idx_gb < MIN_INDEX_GB:
        print(f"  WARNING: index file is only {idx_gb:.1f} GB, far below the "
              f"~{EXPECTED_INDEX_GB} GB expected for a full 38.6M x 768 HNSWFlat "
              f"-> this is very likely a partial/wrong index, which alone would "
              f"explain both the ~0 metrics AND the impossibly low latency. "
              f"Check `ls -la {index_path}` and how it was built.", flush=True)
    else:
        print(f"  index file size is in the expected range (>~{MIN_INDEX_GB} GB) "
              f"-> not an obviously-truncated index.", flush=True)

    if os.path.getmtime(ids_path) < os.path.getmtime(index_path):
        print("  WARNING: ids file is OLDER than the index file - it may be a "
              "stale copy from before the dragon dot-product rebuild. Every "
              "retrieved internal node id would then map to the WRONG pid "
              "string, which would zero out NDCG/MRR/Accuracy without "
              "affecting routing at all.", flush=True)
    else:
        print("  ids file is not older than the index file (no obvious staleness "
              "from mtimes alone - a same-count reordering would NOT be caught "
              "here; run --stage search to rule that out for real).", flush=True)


def stage_search():
    import time

    import faiss  # only needed here, keeps the coverage stage import-light

    print("=== 3. real retrieval spot check (needs the full HNSW index resident) ===",
          flush=True)
    print(f"Loading {index_path} ({os.path.getsize(index_path) / 1e9:.1f} GB on disk) "
          f"- check `free -h` first if RAM is tight.", flush=True)
    index = faiss.read_index(index_path)  # mmap does not reliably help for HNSWFlat
    print(f"Loaded: ntotal={index.ntotal:,}, dim={index.d}", flush=True)

    id_array = np.load(ids_path, allow_pickle=True)
    print(f"index.ntotal={index.ntotal:,} vs len(hnsw_ids.npy)={len(id_array):,} "
          f"-> {'OK' if index.ntotal == len(id_array) else 'MISMATCH!!'}", flush=True)

    qrels = qlr.load_qrels(qlr.MSMARCO_QRELS, pid_prefix="MARCO_")
    indexed_pids = set(str(x) for x in id_array)
    filtered_qrels = {qid: {p: s for p, s in d.items() if p in indexed_pids}
                      for qid, d in qrels.items()}
    filtered_qrels = {qid: d for qid, d in filtered_qrels.items() if d}

    normalize = MODEL != "dragon"
    dev_dir = qlr.MSMARCO_DEV_QUERY_DIRS[MODEL]
    q_ids, q_emb = qlr.load_parquet_embeddings(dev_dir, normalize=normalize)
    eval_keys = [qid for qid in q_ids if qid in filtered_qrels]

    # ground truth (exact top-k), same pid space as the run (MARCO_/CAR_). Lets us
    # reproduce the exact Accuracy@10=0.0 symptom and see WHERE it breaks: if HNSW
    # top-10 has zero overlap with exact top-10, either the search is degenerate
    # (wrong/partial index) or the query/doc embeddings live in different spaces.
    gt = qlr.load_groundtruth(qlr.groundtruth_path(cache_dir, MODEL, DATASET, K))
    if gt is None:
        print("  (no groundtruth file -> Accuracy comparison skipped)", flush=True)

    id_map = {i: str(pid) for i, pid in enumerate(id_array)}
    row_of = {qid: i for i, qid in enumerate(q_ids)}
    sample = eval_keys[:15]
    qv = np.ascontiguousarray(q_emb[[row_of[k] for k in sample]], dtype="float32")
    index.hnsw.efSearch = 200
    t0 = time.perf_counter()
    scores, idxs = index.search(qv, 100)
    ms_per_q = (time.perf_counter() - t0) * 1000 / len(sample)
    print(f"HNSW search latency: {ms_per_q:.3f} ms/query at ef=200 (snowflake was "
          f"~2.5 ms/q; a value in the tens of microseconds means the index is not "
          f"really a 38.6M search).\n", flush=True)

    hits = gt_overlap_any = 0
    for row, qid in enumerate(sample):
        rel_pids = set(filtered_qrels[qid].keys())
        ret_pids = [id_map[i] for i in idxs[row] if i != -1]
        top10 = set(ret_pids[:K])
        overlap = rel_pids & set(ret_pids)
        if overlap:
            hits += 1
        gt_top = set((gt.get(qid) or [])[:K]) if gt else set()
        gt_hit = len(top10 & gt_top) if gt else "n/a"
        if gt and (top10 & gt_top):
            gt_overlap_any += 1
        print(f"  qid={qid:>10}  qrels_hit_top100={bool(overlap)}  "
              f"acc@10_overlap={gt_hit}  top3_ret={ret_pids[:3]}  "
              f"gt_top3={list(gt_top)[:3] if gt else 'n/a'}  "
              f"top3_scores={scores[row][:3].round(3).tolist()}", flush=True)

    print(f"\n{hits}/{len(sample)} sample queries had >=1 qrels pid in top-100.",
          flush=True)
    if gt:
        print(f"{gt_overlap_any}/{len(sample)} had >=1 EXACT-topk pid in HNSW top-10 "
              f"(this is the Accuracy@10 signal).", flush=True)
        print("If qrels_hit is healthy but acc@10_overlap is 0 -> the run and the "
              "groundtruth pids don't line up (id_map / pid-format mismatch). If "
              "BOTH are ~0 with sane scores -> query/doc embeddings are misaligned "
              "(e.g. DRAGON docs must be encoded with the CONTEXT encoder, queries "
              "with the QUERY encoder). If scores are near-degenerate -> wrong/"
              "partial index. None of these is a QLR bug.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["coverage", "search"], default="coverage")
    args = ap.parse_args()
    if args.stage == "coverage":
        stage_coverage()
    else:
        stage_search()
