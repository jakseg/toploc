#!/bin/bash
# ================================================================
# Full grid search script for TopLoc-IVF and TopLoc-IVF+
# Based on paper hyperparameter ranges (Section 3)
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# All results are saved to results/ folder as .txt files
# so you can compare them afterwards.
# ================================================================

mkdir -p results

# ================================================================
# ABOUT MMAP:
#   MMAP=0  → load full index into RAM (faster, needs ~100GB+ RAM)
#   MMAP=1  → memory-map the index   (slower, saves RAM)
#   Only set MMAP=1 if the process gets "Killed" = out of memory
# ================================================================
MMAP=0


# ================================================================
# MODELS and INDEX TYPE (paper tests both)
# ================================================================
MODELS=("snowflake" "dragon")
INDEX="ivf"


# ================================================================
# PAPER GRID SEARCH PARAMETERS
#
# H_CACHED ∈ {512, 1024, 4096, 8192}   — cached centroids
# NP       ∈ {1,2,4,8,16,32,64,128}    — nprobe (powers of 2)
# ALPHA    ∈ {0.0, 0.05, 0.1, 0.2}     — IVF+ only, drift threshold
# ================================================================
H_VALUES=(512 1024 4096 8192)
NP_VALUES=(1 2 4 8 16 32 64 128)
ALPHA_VALUES=(0.0 0.05 0.1 0.2)


# ================================================================
# TOPLOC-IVF GRID SEARCH
# Parameters: model x H_CACHED x NP
# ================================================================
echo "========================================================"
echo "  STARTING TopLoc-IVF GRID SEARCH"
echo "========================================================"

for model in "${MODELS[@]}"; do
    for h in "${H_VALUES[@]}"; do
        for np in "${NP_VALUES[@]}"; do

            out="results/ivf_${model}_H${h}_NP${np}.txt"

            echo "Running: model=${model}  H=${h}  NP=${np}"

            MMAP=$MMAP \
            H_CACHED=$h \
            NP=$np \
            python3 toploc_ivf_2.py $model $INDEX > "$out" 2>&1

            # Print the key metrics line to terminal as it finishes
            grep -E "NDCG|MRR|Avg Time" "$out" | head -4
            echo "  → saved to $out"
            echo ""

        done
    done
done


# ================================================================
# TOPLOC-IVF+ GRID SEARCH
# Parameters: model x H_CACHED x NP x ALPHA
# ================================================================
echo "========================================================"
echo "  STARTING TopLoc-IVF+ GRID SEARCH"
echo "========================================================"

for model in "${MODELS[@]}"; do
    for h in "${H_VALUES[@]}"; do
        for np in "${NP_VALUES[@]}"; do
            for alpha in "${ALPHA_VALUES[@]}"; do

                out="results/ivf_plus_${model}_H${h}_NP${np}_A${alpha}.txt"

                echo "Running: model=${model}  H=${h}  NP=${np}  ALPHA=${alpha}"

                MMAP=$MMAP \
                H_CACHED=$h \
                NP=$np \
                ALPHA=$alpha \
                python3 toploc_ivf_plus.py $model $INDEX > "$out" 2>&1

                grep -E "NDCG|MRR|Avg Time|refreshes" "$out" | head -5
                echo "  → saved to $out"
                echo ""

            done
        done
    done
done


# ================================================================
# SUMMARY — print best NDCG@10 for each file
# ================================================================
echo "========================================================"
echo "  SUMMARY — sorted by NDCG@10"
echo "========================================================"

echo ""
echo "--- TopLoc-IVF ---"
grep -h "NDCG@10" results/ivf_snowflake_*.txt results/ivf_dragon_*.txt 2>/dev/null \
    | sort -t: -k2 -rn \
    | head -20

echo ""
echo "--- TopLoc-IVF+ ---"
grep -h "NDCG@10" results/ivf_plus_snowflake_*.txt results/ivf_plus_dragon_*.txt 2>/dev/null \
    | sort -t: -k2 -rn \
    | head -20

echo ""
echo "All done. Full results in results/ folder."

#======================================
