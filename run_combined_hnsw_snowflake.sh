#!/bin/bash
# ================================================================
# Run combined HNSW evaluation:
# Baseline HNSW + TopLoc-HNSW in ONE script.
#
# IMPORTANT:
# Use MMAP=1 because loading the full Snowflake HNSW index into RAM
# caused memory/swap problems on big-dama-3.
#
# Usage:
#   chmod +x run_combined_hnsw_snowflake.sh
#   ./run_combined_hnsw_snowflake.sh
# ================================================================

mkdir -p results

# Safer on current server
MMAP=1

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
PYTHONPATH=build_hnsw_release \
python -u combine_base_top_hnsw.py ${MODEL} ${INDEX} \
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