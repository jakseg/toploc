#!/usr/bin/env bash
# Set up the demo environment and compile the real toploc_search C++ kernel.
#
# One-time setup on any machine (after installing Miniconda/Conda):
#     ./demo/build_demo.sh   # run from the repo root
# Then run the demo:
#     conda activate toploc-demo
#     streamlit run demo/demo_app.py
#
# The compiled module (toploc_search.*.so) is platform-specific, so each
# machine runs this once. The demo data lives in demo/data/ (copy it over
# separately — it is not built here and not committed to git).
# Note: no 'set -u' — conda's compiler activation scripts reference unbound
# variables (e.g. AR) and would abort the script otherwise.
set -eo pipefail

ENV_NAME=toploc-demo
# This script lives in demo/, next to environment.yml; the repo root is one level
# up. CMakeLists.txt and toploc_search.cpp live in 'old scripts/' (the kernel is
# shared with toploc_ivf.py).
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/old scripts"
cd "$REPO_ROOT"

# 1. Create (or update) the conda environment from the recipe.
eval "$(conda shell.bash hook)"
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo ">> Updating existing env '${ENV_NAME}'..."
    conda env update -n "${ENV_NAME}" -f "$DEMO_DIR/environment.yml" --prune
else
    echo ">> Creating env '${ENV_NAME}'..."
    conda env create -f "$DEMO_DIR/environment.yml"
fi
conda activate "${ENV_NAME}"

# 2. Compile toploc_search.cpp against this env's faiss + pybind11.
echo ">> Building toploc_search..."
rm -rf build
cmake -S "$SRC_DIR" -B build \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DPython3_EXECUTABLE="${CONDA_PREFIX}/bin/python" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. Place the module at the repo root so demo_app.py / toploc_ivf.py can import it.
cp build/toploc_search*.so "${REPO_ROOT}/"

echo ""
echo ">> Done. Built: $(ls toploc_search*.so)"
echo ">> Run the demo with:"
echo "     conda activate ${ENV_NAME}"
echo "     streamlit run demo/demo_app.py"
