#!/bin/bash
# ================================================================
# Run combined HNSW evaluation:
# Baseline HNSW + TopLoc-HNSW in ONE script.
#
# IMPORTANT — latency backend:
# The default backend is now `faiss` (native search_level_0, globally batched):
# real, paper-faithful latency, no C++ module to build. The cpp/python beam
# kernels are correctness-only and make TopLoc look slower than it is.
#
# IMPORTANT — MMAP:
# Run with MMAP=0 (full RAM load). The snowflake HNSW index (~157 GiB) fits in
# the ~250 GB free on big-dama-3, and mmap distorts the microsecond latency we
# are measuring. Only fall back to MMAP=1 if the other group is using the node
# and free RAM drops below ~180 GB (metrics are identical either way).
#
# Usage:
#   chmod +x run_combined_hnsw_snowflake.sh
#   ./run_combined_hnsw_snowflake.sh
# ================================================================

mkdir -p results

# Faithful latency: full RAM load (see note above). Set MMAP=1 only as a fallback.
MMAP=0

MODEL="snowflake"
INDEX="hnsw"

EF_SEARCH=512
UP=2
ENTRY_POINTS=1
THREADS=1

OUT="results/combined_hnsw_${MODEL}_ef${EF_SEARCH}_up${UP}_ep${ENTRY_POINTS}_mmap.txt"

echo "========================================================"
echo "  STARTING COMBINED HNSW RUN"
echo "========================================================"
echo "Model:        ${MODEL}"
echo "Index:        ${INDEX}"
echo "efSearch:     ${EF_SEARCH}"
echo "up:           ${UP}"
echo "entry_points: ${ENTRY_POINTS}"
echo "threads:      ${THREADS}"
echo "MMAP:         ${MMAP}"
echo "Output:       ${OUT}"
echo "========================================================"

export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}
export THREADS=${THREADS}

MMAP=${MMAP} \
python -u combine_base_top_hnsw_test.py ${MODEL} ${INDEX} \
  --backend faiss \
  --ef-search ${EF_SEARCH} \
  --up ${UP} \
  --entry-points ${ENTRY_POINTS} \
  --threads ${THREADS} \
  > "${OUT}" 2>&1

echo ""
echo "========================================================"
echo "  FINISHED"
echo "========================================================"
echo "Saved to: ${OUT}"
echo ""
echo "Last lines:"
tail -80 "${OUT}"