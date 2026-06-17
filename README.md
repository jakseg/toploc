# TopLoc Baseline Implementation

Reimplementation of the baseline retrieval methods from the [TopLoc paper](https://arxiv.org/abs/2501.01onal) (SIGIR '25), using the TREC CAsT 2019/2020 datasets.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

### 1. Create Embeddings

```bash
python create_embeddings_snowflake.py   # Snowflake Arctic Embed (1024-dim)
python create_embeddings_dragon.py      # Dragon+ (768-dim)
```

Outputs are saved to `data/snowflake/` and `data/dragon/` respectively.

### 2. Build Index

```bash
python create_index.py <model> <index_type> [dataset]
```

- `model`: `snowflake` or `dragon`
- `index_type`: `exact`, `ivf`, `hnsw`, a comma-separated list (e.g. `exact,ivf`), or `all`
- `dataset`: `cast2019` (default) or `msmarco` (the QLR/toploc2 collection; lands in
  its own `toploc2/msmarco/<model>/` cache and builds HNSW with `ef_construction=500`)

Examples:
```bash
python create_index.py snowflake ivf            # CAST2019 (default)
python create_index.py snowflake exact,ivf      # build two sequentially
python create_index.py snowflake all            # exact + ivf + hnsw
python create_index.py snowflake hnsw msmarco   # QLR document index (msmarco)
```

The build streams embeddings from parquet and checkpoints every 50 files
(index + ids + checkpoint metadata). If a build is interrupted, rerunning the
same command resumes from the last checkpoint. Per-index logs are written to
the cache directory (e.g. `ivf_indexCreation.log`).

### 3. Evaluate

```bash
python evaluate_baseline.py <model> <index_type>
```

Examples:
```bash
python evaluate_baseline.py snowflake ivf
python evaluate_baseline.py dragon exact
python evaluate_baseline.py snowflake hnsw
```

Reports NDCG@10, MRR@10, and average query latency.

### Test Pipeline

Quick sanity check with a single query:

```bash
python test_ivf_pipeline.py <model>
```

## Interactive Demo (Streamlit)

A small Streamlit app that runs conversational retrieval over a compact subset
(all judged passages + distractors) so it fits on a laptop. Per query it shows
the retrieved passages, the number of vectors actually compared (a
scale-independent efficiency proxy), the speedup vs exact search, and — for
known CAsT turns — live NDCG/MRR. Exact and IVF use plain FAISS; **TopLoc IVF
calls the real `toploc_search` C++ kernel** (the same one used in the benchmark
scripts), with a pure-Python fallback if the module is not compiled.

### 1. Get the demo subset (`demo/data/`)

The demo needs a self-contained `demo/data/` folder (passages, FAISS indices,
query embeddings, topics, qrels, meta). It is a generated artifact and **not**
committed (gitignored). Two ways to obtain it:

**a) Download the prebuilt subset (recommended).** It is attached as an asset to
the `demo-data` release:

```bash
cd demo
gh release download demo-data -p demo_data.zip   # or download from the Releases page
unzip demo_data.zip                               # creates demo/data/
```

**b) Rebuild it (where the full data lives).**

```bash
python demo/build_demo_subset.py
```

The full 38M collection is only needed for rebuilding; the demo itself runs
entirely off `demo/data/`.

### 2. Set up the demo environment and build the C++ kernel

Running the real `toploc_search` kernel needs FAISS with C++ dev files, so the
demo uses a conda environment (`environment.yml`). A system C++ compiler is
required for the build (macOS: `xcode-select --install`; Linux: `build-essential`).

```bash
./build_demo.sh        # creates the 'toploc-demo' conda env and compiles toploc_search
```

This recreates the environment from `environment.yml` and builds
`toploc_search.*.so` (platform-specific, gitignored — run once per machine).

### 3. Run the app

```bash
conda activate toploc-demo
streamlit run demo/demo_app.py
```

Pick a known CAsT turn from the sidebar (precomputed embedding + metrics) or type
a free-text query, switch between **Exact / IVF / TopLoc IVF**, and tune
`nprobe` and the number of cached centroids `h`. In a conversation the first turn
seeds the TopLoc cache (standard IVF search); follow-up turns reuse it and scan
far fewer vectors.

## Data Structure

```
dataset/              Raw data (passages, topics, qrels)
data/snowflake/       Snowflake embeddings, id map, and indices
data/dragon/          Dragon embeddings, id map, and indices
```
