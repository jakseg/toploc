#!/usr/bin/env bash
# Reproduce the QLR (toploc2) experiments on msmarco-on-cast, pure Python (no C++).
#
#
#   bash run_qlr_experiments.sh snowflake
#   LOG_LIMIT=0 bash run_qlr_experiments.sh snowflake     # full ~808k log
#   bash run_qlr_experiments.sh dragon
#
# Produces, under results_<model>_<dataset>_<ts>/:
#   groundtruth.log, baseline_sweep.csv, qlr_sweep.csv (+ logs)
set -euo pipefail

MODEL="${1:-snowflake}"               # snowflake | dragon
DATASET="${2:-msmarco-on-cast}"
LOG_LIMIT="${LOG_LIMIT:-100000}"      # cap |Q_L| for a first pass; 0 = full log
PY="${PY:-python}"

# mmap only affects ACCESS SPEED, not results — NDCG/MRR/Accuracy@10/avg_visited are
# identical either way. combine_base_top_hnsw.py (the TopLoc-HNSW comparison) runs
# WITHOUT mmap by default, so default to that here to stay comparable: full RAM load
# (the ~157 GiB snowflake index then needs that much free RAM; slower to load, but
# latency-comparable). Set MMAP=1 if RAM is tight — safe for accuracy/visited; only a
# future real-latency comparison would then not match.
export MMAP="${MMAP:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="results_${MODEL}_${DATASET}_${TS}"
mkdir -p "$OUT"
echo "model=$MODEL dataset=$DATASET log_limit=$LOG_LIMIT -> $OUT/"

# 0. dragon only: produce dev embeddings (snowflake's are precomputed).
if [ "$MODEL" = "dragon" ]; then
  echo "== dragon: encode dev embeddings =="
  $PY -u encode_msmarco_dev_dragon.py 2>&1 | tee "$OUT/encode_dev.log"
fi

# 1. Ground truth for Accuracy@10 (exact top-10). Loads only the doc embeddings,
#    NOT the HNSW index, so it does not compete for RAM with the eval run. Threads
#    the GEMM via OMP_NUM_THREADS (the matmul uses numpy/BLAS, not faiss threads).
echo "== 1/2 ground truth (exact top-10) =="
OMP_NUM_THREADS="${THREADS:-14}" \
$PY -u compute_groundtruth.py "$MODEL" --dataset "$DATASET" --method stream \
  2>&1 | tee "$OUT/groundtruth.log"

# 2. Baseline + QLR in ONE process so the ~157 GiB index loads only ONCE.
#    --mode both runs the plain-HNSW efSearch sweep AND the QLR sweep
#    (th x k' x ef-search x PCA), writing both into one CSV (the 'mode' column
#    separates them). EP/s_max are built once and reused across the QLR grid.
#    --threads 1 is MANDATORY for paper-faithful latency (the paper searches
#    single-threaded); latency_ms_per_q is then the amortized, comparable number.
#    (GT above keeps many threads for its matmul — it doesn't touch the index.)
echo "== 2/2 baseline + QLR sweep (single index load, threads=1 for faithful latency) =="
$PY -u qlr.py "$MODEL" --dataset "$DATASET" \
  --mode both --sweep --log-limit "$LOG_LIMIT" --threads 1 \
  --out "$OUT/sweep.csv" \
  2>&1 | tee "$OUT/sweep.log"

echo "Done. In $OUT/sweep.csv compare mode=baseline vs mode=qlr at MATCHED Accuracy@10:"
echo "  baseline latency_ms_per_q  vs  qlr latency_ms_per_q  ->  speedup (grows with the target)."
