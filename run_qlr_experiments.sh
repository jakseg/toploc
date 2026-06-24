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

# 0. dragon only: produce dev embeddings (snowflake's are precomputed) and validate.
if [ "$MODEL" = "dragon" ]; then
  echo "== dragon: encode dev embeddings + smoke test =="
  $PY -u encode_msmarco_dev_dragon.py        2>&1 | tee "$OUT/encode_dev.log"
  $PY -u smoke_test_msmarco_on_cast.py dragon 2>&1 | tee "$OUT/smoke.log"
fi

# 1. Ground truth for Accuracy@10 (exact top-10). Loads only the doc embeddings,
#    NOT the HNSW index, so it does not compete for RAM with the eval runs.
echo "== 1/3 ground truth (exact top-10) =="
$PY -u compute_groundtruth.py "$MODEL" --dataset "$DATASET" --method stream \
  2>&1 | tee "$OUT/groundtruth.log"

# 2. Plain-HNSW baseline curve — sweep efSearch 10..200 (the paper's comparison).
echo "== 2/3 baseline sweep (plain HNSW) =="
$PY -u toploc2_hnsw_pure_python.py "$MODEL" --dataset "$DATASET" \
  --mode baseline --sweep --out "$OUT/baseline_sweep.csv" \
  2>&1 | tee "$OUT/baseline.log"

# 3. QLR sweep — th x k' x ef-search x PCA(0, dim/4). EP/s_max built once, reused.
echo "== 3/3 QLR sweep =="
$PY -u toploc2_hnsw_pure_python.py "$MODEL" --dataset "$DATASET" \
  --sweep --log-limit "$LOG_LIMIT" --out "$OUT/qlr_sweep.csv" \
  2>&1 | tee "$OUT/qlr.log"

echo "Done. Compare baseline_sweep.csv vs qlr_sweep.csv (Accuracy@10 at matched cost) in $OUT/"
