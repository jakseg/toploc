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

## QLR (Query Log Router — toploc2)

Reimplementation of "HNSW Graph Meets Query Logs": a lightweight router in front
of a standard HNSW document index. It runs on `msmarco-on-cast` — the CAST2019
HNSW index is the document index `I_D` (it already contains every msmarco passage
as `MARCO_<n>`), with the msmarco train split as the query log `Q_L` and the
msmarco dev queries + `qrels.dev.small.tsv` as the test set.

Driver: `toploc2_hnsw_pure_python.py` (pure Python, no C++). Two metrics are
reported: **Accuracy@10** (fraction of the *exhaustive* top-10 retrieved — the
paper's headline metric) and qrels-based **NDCG@3/10 + MRR@10**.

### 0. Local sanity check (no server needed)

```bash
conda activate toploc-demo        # any env with faiss + pyarrow + ir_measures
python test_qlr_pipeline.py       # synthetic, runs in seconds -> 14/14 passed
```

### Run order (on the server)

All commands use `--dataset msmarco-on-cast`.

**On `MMAP`:** memory-mapping only changes how fast vectors are read from disk, not
the results — NDCG/MRR/Accuracy@10 and `avg_visited` are identical with or without
it. To stay latency-comparable to `combine_base_top_hnsw.py` (the TopLoc-HNSW
comparison, which runs **without** mmap), leave `MMAP` unset (default off): the
index is loaded fully into RAM, so the ~157 GiB snowflake index needs that much
free RAM and loads slower. If RAM is tight, set `MMAP=1` — safe for accuracy/visited,
only a real-latency comparison would then not match. (`compute_groundtruth.py` never
loads the HNSW index, so mmap is irrelevant there.)

**One-shot wrapper** (runs steps 1–4 below and tees logs to
`results_<model>_<dataset>_<ts>/`):

```bash
bash run_qlr_experiments.sh snowflake               # snowflake (data validated)
bash run_qlr_experiments.sh dragon                  # encodes dev embeddings + smoke test first
LOG_LIMIT=0 bash run_qlr_experiments.sh snowflake   # full ~808k log (default caps |Q_L| at 100k)
```

Or the individual steps, in order:

1. **(dragon only)** encode the dev queries and validate the data:
   ```bash
   python encode_msmarco_dev_dragon.py
   python smoke_test_msmarco_on_cast.py dragon       # expect ~100% qrels coverage, READY
   ```

2. **ground truth for Accuracy@10** (one-time exact top-10; streams the doc
   embeddings, does not load the HNSW index):
   ```bash
   python compute_groundtruth.py <model> --dataset msmarco-on-cast   # --method stream (default)
   ```

3. **plain-HNSW baseline** (the comparison) — sweep efSearch 10..200:
   ```bash
   python -u toploc2_hnsw_pure_python.py <model> --dataset msmarco-on-cast \
       --mode baseline --sweep --out baseline_sweep.csv
   ```

4. **QLR sweep** (`th × k' × ef × PCA`; EP/s_max built once and reused):
   ```bash
   python -u toploc2_hnsw_pure_python.py <model> --dataset msmarco-on-cast \
       --sweep --log-limit 100000 --out qlr_sweep.csv
   ```

Each sweep writes a CSV (and prints a markdown table). Compare `baseline_sweep.csv`
vs `qlr_sweep.csv`: QLR should reach about the **same Accuracy@10 as the baseline**
on the identical index. A quick single-config first run:

```bash
python -u toploc2_hnsw_pure_python.py snowflake --dataset msmarco-on-cast \
    --log-limit 2000 --max-turns 50
```

### Real routed latency (FAISS `search_level_0`)

The routed level-0 beam search runs two ways: the pure-Python loop (validates
correctness) or FAISS' built-in `search_level_0` — the *same* seeded level-0
search, but executed in compiled C++, so it gives **real latency** with identical
top-k. FAISS exposes the seeded entry points directly (`search_type=2`), so no
custom C++ kernel is needed. Select the backend per run:

```bash
# whole run uses the FAISS backend -> routed_ms_per_q is real C++ latency
python -u toploc2_hnsw_pure_python.py <model> --dataset msmarco-on-cast \
    --sweep --level0-backend faiss --out qlr_sweep.csv

# OR run BOTH and put them side by side in the CSV (verify before trusting)
python -u toploc2_hnsw_pure_python.py <model> --dataset msmarco-on-cast \
    --sweep --compare-backends --log-limit 2000 --max-turns 200
```

`--compare-backends` adds three columns: `routed_ms_faiss_per_q` (FAISS, real)
next to `routed_ms_per_q` (Python), `speedup_routed` (their ratio), and
`topk_match` — `1.0` means both backends return identical results, so the speedup
is honest. Validate on a small `--log-limit`/`--max-turns` first; the speedup only
shows on the full 38.6M index (on a tiny index the Python marshaling overhead
dominates). The **IVF** path already runs in FAISS C++ via `search_preassigned`,
so this backend swap applies only to the HNSW/QLR path.

Notes:
- The pure-Python routed latency (default `--level0-backend python`) is **not
  real** — `avg_visited` (nodes scanned) is the implementation-independent
  efficiency proxy, unaffected by mmap or thread count. For real routed latency use
  `--level0-backend faiss` / `--compare-backends` (above); for a clean latency run
  match `combine_base_top_hnsw.py` (no mmap, single-thread:
  `OMP_NUM_THREADS=1 ... --threads 1`).
- snowflake `th` is cosine (default 0.5); dragon is raw dot product, so its `th`
  is calibrated from the data when unset.

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
