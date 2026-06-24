#!/bin/bash
# ================================================================
# Grid search using toploc_combined.py
# Loads index + model ONCE per (model, H, NP, ALPHA) combination
# then runs both IVF and IVF+ evals back to back.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
# ================================================================

mkdir -p results

# MMAP=0 → load full index into RAM (faster, needs lots of RAM)
# MMAP=1 → memory-map the index (slower, saves RAM if you get "Killed")
MMAP=0

MODELS=("snowflake" "dragon")
INDEX="ivf"

# Paper grid search ranges
H_VALUES=(512 1024 4096 8192)
NP_VALUES=(1 2 4 8 16 32 64 128)
ALPHA_VALUES=(0.0 0.05 0.1 0.2)

echo "========================================================"
echo "  STARTING COMBINED GRID SEARCH"
echo "  Each run loads index+model ONCE, runs IVF + IVF+"
echo "========================================================"

for model in "${MODELS[@]}"; do
    for h in "${H_VALUES[@]}"; do
        for np in "${NP_VALUES[@]}"; do
            for alpha in "${ALPHA_VALUES[@]}"; do

                out="results/combined_${model}_H${h}_NP${np}_A${alpha}.txt"

                echo "Running: model=${model}  H=${h}  NP=${np}  ALPHA=${alpha}"

                MMAP=$MMAP \
                H_CACHED=$h \
                NP=$np \
                ALPHA=$alpha \
                python3 toploc_combined.py $model $INDEX > "$out" 2>&1

                grep -E "NDCG@10|Avg Time|refreshes" "$out"
                echo "  → saved to $out"
                echo ""

            done
        done
    done
done

# ================================================================
# SUMMARY
# ================================================================
echo "========================================================"
echo "  SUMMARY — sorted by NDCG@10 (filename tells you config)"
echo "========================================================"
grep -r "NDCG@10" results/combined_*.txt 2>/dev/null \
    | sort -t: -k2 -rn \
    | head -20

echo ""
echo "All done. Full results in results/ folder."