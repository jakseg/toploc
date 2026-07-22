# TopLoc & QLR Reimplementation

Reimplementation of two retrieval papers on shared infrastructure:

- **TopLoc** (SIGIR '25) — conversational retrieval (Exact / IVF / HNSW baselines
  plus the TopLoc IVF/HNSW speedups) on TREC CAsT **2019 and 2020**.
- **"HNSW Graph Meets Query Logs"** — the Query Log Router (QLR), a lightweight
  router in front of an HNSW document index, on MS MARCO.

Both run with the **Snowflake** and **Dragon** encoders.

## Setup

```bash
# For toploc IVF/IVF+:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For toploc HNSW / QLR
```

## Pipeline

### 1. Create Embeddings

The finished embeddings were provided by our supervisor; these scripts reproduce
them and are paper-faithful (toploc.pdf §Models). They stream any input and write
sharded parquet with columns `id`, `embedding` — the format `create_index.py` and
the QLR driver consume. Both are model-parametrized (`snowflake` / `dragon`).

Run from the repo root, in the project venv. Set the model once — every command
below is model-parametrized:

```bash
source venv/bin/activate

D=/home/toploc2/Datasets/conversational
M=snowflake          # or: M=dragon
```

Every command is a **full run**. Append `--limit N --shard-size 0` to write a single
small shard instead.

#### Document collection

The CAsT2019 collection, 38.6M passages:

```bash
python create_embeddings/create_document_embeddings.py $M \
  --input   $D/CAST2019/CAST2019collection.tsv \
  --out-dir $D/CAST2019/${M}_embeddings
```

#### QLR queries

QLR searches the same document index built from the collection above (`I_D`), so it
needs no new document embeddings — only queries: the historical query log `Q_L`
(msmarco train, ~808k) and the test queries (msmarco dev.small).

```bash
# historical query log Q_L
python create_embeddings/create_query_embeddings.py $M \
  --input   $D/msmarco/msmarco_train_queries.jsonl --input-format jsonl \
  --out-dir $D/msmarco/$M

# test queries
python create_embeddings/create_query_embeddings.py $M \
  --input   $D/CAST2019/msmarco/msmarco_queries/dev_queries.jsonl --input-format jsonl \
  --out-dir $D/CAST2019/msmarco/msmarco_embeddings/dev_query     # dragon: dev_query_dragon
```

#### Observations

Checked against the supplied embeddings, per-id cosine ≥ 0.9999:

- **Documents: snowflake normalised, dragon raw (norm ~65)** — each model is stored
  the way it scores (dragon-plus uses raw dot product). `create_index.py`
  L2-normalises both at build time, which is what the graph needs.
- **The supplied dragon `Q_L` is un-normalised, ours is normalised** — same vectors,
  different scale. `qlr.py` normalises at load, so both work.

### 2. Build Index

```bash
python create_index.py <model> <index_type> [dataset] [--param value ...]
```

- `model`: `snowflake` or `dragon`
- `index_type`: `exact`, `ivf`, `hnsw`, a comma-separated list (e.g. `exact,ivf`), or `all`
- `dataset`: `cast2019` (default). QLR does **not** need a separate index — it reuses
  the CAST2019 index as its document index `I_D` (see the QLR section). A legacy
  `msmarco` dataset also exists but is not used by the final pipeline.

Examples:
```bash
python create_index.py snowflake ivf            # CAST2019 (default)
python create_index.py snowflake exact,ivf      # build two sequentially
python create_index.py snowflake all            # exact + ivf + hnsw
```

#### Choosing the index parameters

Every index parameter is exposed as an optional flag. **Omit a flag and the build
uses the paper-faithful per-model / per-dataset default**, so the positional-only form
above reproduces the paper. Pass a flag to override it. The values tested in the two
papers are listed with `python create_index.py --help`; the same grid in short:

| Flag | Applies to | Paper values | Default |
|------|-----------|--------------|---------|
| `--num-centroids` | IVF | `{32768, 65536, 131072, 262144}` = 2¹⁵–2¹⁸ (TopLoc) | dragon 262144, snowflake 32768 |
| `--nprobe` | IVF | 1…4096 in powers of 2 (TopLoc) | 128 |
| `--kmeans-niter` | IVF | 25 (FAISS default), 10 (project) | 25 |
| `--train-sample-size` | IVF | ≥ 39 × `num_centroids` (FAISS heuristic) | auto (≥ 40 × centroids) |
| `--M` | HNSW | `{16, 32, 64}` (TopLoc); 32 (QLR) | 32 |
| `--ef-construction` | HNSW | 500 (QLR); 200 (project default) | 200 (cast2019) / 500 (msmarco) |
| `--ef-search` | HNSW | 1…4096 pow2 (TopLoc); 10…200 step 10 (QLR) | 64 |
| `--normalize / --no-normalize` | both | Dragon L2-normalized for cosine (TopLoc) | `--normalize` |

`--nprobe` and `--ef-search` are search-time knobs baked into the saved index as
defaults; the evaluation scripts sweep them. Keep `--normalize` on: an un-normalized
HNSW/IVF graph on Dragon degenerates (norm-bias hubs → ~0 recall).

Examples:
```bash
# reproduce the paper build (no flags needed)
python create_index.py dragon ivf                                   # 2^18 centroids
python create_index.py snowflake ivf                                # 2^15 centroids
# pick your own parameters
python create_index.py dragon ivf  --num-centroids 131072 --kmeans-niter 10
python create_index.py snowflake hnsw --M 64 --ef-construction 500
```

The build streams embeddings from parquet and checkpoints every 50 files
(index + ids + checkpoint metadata). If a build is interrupted, rerunning the
same command resumes from the last checkpoint. Per-index logs are written to
the cache directory (e.g. `ivf_indexCreation.log`).

### 3. Toploc HNSW/IVF/IVF+ (CAsT 2019 / 2020)

Conversational retrieval and metrics run through `combine_hnsw.py` (HNSW, incl. the
TopLoc-HNSW speedup) and `combine_IVF.py` (IVF). Each loads the document index from
the model cache and reads the topics/qrels from `DATASET_DIR` (default = CAsT 2019).
`--sweep` runs the full efSearch / nprobe grid; single-thread latency is the default.
CAsT **2020 reuses the exact same index** — only `DATASET_DIR` changes.

```bash
# CAsT 2019 (default DATASET_DIR) or CAsT 2020 

# Running Toploc IVF and IVF+:

# Snowflake:

tmux new-session -s snow2020

taskset -c 0-13 env NUM_THREADS=14 OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
DATASET_DIR=/home/toploc2/Datasets/conversational/CAST2020/topics \
python3 -u combine_IVF.py snowflake ivf --sweep 2>&1 | tee results/snow_cast2020.log

# Dragon:

tmux new-session -s drag2020

taskset -c 0-13 env NUM_THREADS=14 OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
DATASET_DIR=/home/toploc2/Datasets/conversational/CAST2020/topics \
ALPHA_GRID="0.0,0.05,0.1,0.2" \
python3 -u combine_IVF.py dragon ivf --sweep 16,32,64,128,256,512,1024,2048,4096 2>&1 | tee results/drag_cast2020.log



# Running Toploc HNSW:

#Snowflake:

tmux new -s snowflake_cast2020_hnsw

CACHE_BASE=/home/toploc2/Datasets/toploc2 \
DATASET_DIR=/home/toploc2/Datasets/conversational/CAST2020/topics \
MMAP=0 \
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 \
taskset -c 0-13 python -u combine_hnsw.py snowflake hnsw \
  --backend faiss --threads 14 --entry-points 1 --sweep \
  2>&1 | tee results/snowflake_hnsw_cast2020_sweep_t14.log

#Dragon:

tmux new -s dragon_cast2020_hnsw

CACHE_BASE=/home/toploc2/Datasets/toploc2 \
DATASET_DIR=/home/toploc2/Datasets/conversational/CAST2020/topics \
MMAP=0 \
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 \
taskset -c 0-13 python -u combine_hnsw.py dragon hnsw \
  --backend faiss --threads 14 --entry-points 1 --sweep \
  2>&1 | tee results/dragon_hnsw_cast2020_sweep_t14.log

```

Reports NDCG@3/10, MRR@10 and single-thread per-query latency. Startup sanity:
"Turns with relevant passages in index" must be > 0. On CAsT 2019 (paper Table 1)
expect snowflake HNSW NDCG@10 ≈ 0.500, dragon ≈ 0.466 — validate 2019 before
trusting 2020.

## QLR (Query Log Router — toploc2)

Reimplementation of "HNSW Graph Meets Query Logs": a lightweight router in front
of a standard HNSW document index. It runs on `msmarco-on-cast` — the CAST2019
HNSW index is the document index `I_D` (it already contains every msmarco passage
as `MARCO_<n>`), with the msmarco train split as the query log `Q_L` and the
msmarco dev queries + `qrels.dev.small.tsv` as the test set.

Driver: `qlr.py` (pure Python, no C++). Two metrics are
reported: **Accuracy@10** (fraction of the *exhaustive* top-10 retrieved — the
paper's headline metric) and qrels-based **NDCG@3/10 + MRR@10**.

All QLR scripts live in the **`toploc2/`** directory — `cd toploc2` first; the
commands below are run from there (they import the driver as a sibling module).

### 0. Local sanity check (no server needed)

```bash
cd toploc2
conda activate toploc-demo        # any env with faiss + pyarrow + ir_measures
python test_qlr_pipeline.py       # synthetic, runs in seconds -> 16/16 passed
```

### Run order (on the server)

All commands use `--dataset msmarco-on-cast`.

**On `MMAP`:** memory-mapping only changes how fast vectors are read from disk, not
the results — NDCG/MRR/Accuracy@10 and `avg_visited` are identical with or without
it. To stay latency-comparable to `combine_hnsw.py` (the TopLoc-HNSW
comparison, which runs **without** mmap), leave `MMAP` unset (default off): the
index is loaded fully into RAM, so the ~157 GiB snowflake index needs that much
free RAM and loads slower. If RAM is tight, set `MMAP=1` — safe for accuracy/visited,
only a real-latency comparison would then not match. (`compute_groundtruth.py` never
loads the HNSW index, so mmap is irrelevant there.)

**One-shot wrapper** (runs steps 1–4 below and tees logs to
`results_<model>_<dataset>_<ts>/`):

```bash
bash run_qlr_experiments.sh snowflake               # snowflake (data validated)
bash run_qlr_experiments.sh dragon                  # encodes dev embeddings first
LOG_LIMIT=0 bash run_qlr_experiments.sh snowflake   # full ~808k log (default caps |Q_L| at 100k)
```

Or the individual steps, in order:

1. **(dragon only)** encode the dev queries:
   ```bash
   python encode_msmarco_dev_dragon.py
   ```

2. **ground truth for Accuracy@10** (one-time exact top-10; streams the doc
   embeddings, does not load the HNSW index). The matmul uses numpy/BLAS, so
   thread it via `OMP_NUM_THREADS`, not `--threads`:
   ```bash
   OMP_NUM_THREADS=14 python compute_groundtruth.py <model> --dataset msmarco-on-cast
   ```

3. **baseline + QLR in ONE run** — `--mode both` loads the ~157 GiB index *once*,
   then runs the plain-HNSW efSearch sweep **and** the QLR sweep (`th × k' × ef ×
   PCA`), writing both to one CSV (the `mode` column separates them; EP/s_max are
   built once and reused across the QLR grid):
   ```bash
   python -u qlr.py <model> --dataset msmarco-on-cast \
       --mode both --sweep --log-limit 100000 --threads 14 --out sweep.csv
   ```

In `sweep.csv` compare `mode=baseline` vs `mode=qlr`: QLR should reach about the
**same Accuracy@10 as the baseline** on the identical index. A quick first run
(also one index load):

```bash
python -u qlr.py snowflake --dataset msmarco-on-cast \
    --mode both --log-limit 2000 --max-turns 50
```

### Best configuration (per model)

Reading the full `--sweep --mode both` runs at the paper's **iso-accuracy**
protocol (fastest config of each method that reaches an Accuracy@10 target;
`latency_ms_per_q` = faithful batched, single thread):

| Model     | Best QLR config                | Acc@10 | QLR lat | plain-HNSW @ same Acc | Speedup |
|-----------|--------------------------------|--------|---------|-----------------------|---------|
| snowflake | `th=0.6 k'=10 pca=256 ef=70`   | 0.935  | 1.15 ms | 1.47 ms (ef=100)      | **1.28×** |
| dragon    | `th=0.7 k'=10 pca=192 ef=110`  | 0.931  | 0.72 ms | 1.09 ms (ef=170)      | **1.51×** |

`th`, `k'`, `pca` are the fixed best settings; **`ef` (`--ef-default`) is the
operating-point dial** — raise it for higher Accuracy@10 at more latency. The
speedup *grows* with the accuracy target (the paper's headline property), and
dragon's plain-HNSW baseline caps at Acc@10 ≈ 0.938, so at Acc 0.95 QLR wins
outright (no baseline config reaches it).

Start QLR directly with the best config (real routed latency, single thread, full
query log). Accuracy@10 needs the ground truth from step 2 above; without it that
column is `nan` (routing/latency still run):

```bash
cd toploc2
# snowflake
python -u qlr.py snowflake --dataset msmarco-on-cast \
    --mode qlr --threads 1 --log-limit 0 \
    --th 0.6 --k-prime 10 --pca-dim 256 --ef-default 70 --out snowflake_best.csv

# dragon (encode dev embeddings first, see step 1 above)
python -u qlr.py dragon --dataset msmarco-on-cast \
    --mode qlr --threads 1 --log-limit 0 \
    --th 0.7 --k-prime 10 --pca-dim 192 --ef-default 110 --out dragon_best.csv
```

The one-time EP build over the full ~808k log takes ~80 min; use `--log-limit
100000` for a faster, slightly less faithful run. For the full speedup *curve*
(every accuracy target side by side) use the `--sweep --mode both` command above.

### Routed latency (FAISS `search_level_0`)

The routed level-0 beam search is executed by FAISS' built-in `search_level_0` —
the seeded level-0 search in compiled C++ (FAISS exposes the seeded entry points
directly via `search_type=2`), so no custom C++ kernel is needed. All routed
queries are batched into ONE call so the `VisitedTable` (~38 MB on the 38.6M
index) is allocated once, not per query; the reported `latency_ms_per_q` is
therefore the real single-thread per-query latency. Run single-thread and without
mmap to stay comparable to `combine_hnsw.py` (`--threads 1`, `MMAP=0`).
A pure-Python beam and per-query variants are kept only inside `test_qlr_pipeline.py`
as the correctness reference — their top-k is asserted identical to the batched
FAISS path, which is why the batched latency can be trusted.

Notes:
- snowflake `th` is cosine (default 0.5); dragon is cosine too (L2-normalised for
  HNSW/IVF navigation) but its `th` is calibrated from the data when unset, because
  its query/context encoders are asymmetric.
- The **IVF** path already runs in FAISS C++ via `search_preassigned`.

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
./demo/build_demo.sh   # creates the 'toploc-demo' conda env and compiles toploc_search
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
