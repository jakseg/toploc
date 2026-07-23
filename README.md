# TopLoc & QLR Reimplementation

Reimplementation of two retrieval papers on shared infrastructure:

- **TopLoc** (SIGIR '25) — conversational retrieval (Exact / IVF / HNSW baselines
  plus the TopLoc IVF/HNSW speedups) on TREC CAsT **2019 and 2020**.
- **"HNSW Graph Meets Query Logs"** — the Query Log Router (QLR), a lightweight
  router in front of an HNSW document index, on MS MARCO.

Both run with the **Snowflake** and **Dragon** encoders.

## Setup

Two separate environments, one per part of the project — activate whichever the step
below belongs to:

```bash
# For toploc IVF/IVF+:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# venv-qlr -- for everything else (deactivate the other one first):
python3 -m venv venv-qlr
source venv-qlr/bin/activate
pip install -r requirements-qlr.txt
```

`venv-qlr` covers `create_embeddings/*`, `create_index.py`, `combine_hnsw.py` and
everything under `toploc2/` (QLR).

They are kept apart because QLR and `combine_hnsw.py --backend faiss` need
**faiss < 1.14**: 1.14 flipped the sign convention of the seed distances in
`search_level_0`, so the seeded level-0 search returns silently wrong results there.
`requirements-qlr.txt` pins that upper bound without imposing it on the IVF side.

## Pipeline

### 1. Create Embeddings

The finished embeddings were provided by our supervisor; these scripts reproduce
them and are paper-faithful (toploc.pdf §Models). They stream any input and write
sharded parquet with columns `id`, `embedding` — the format `create_index.py` and
the QLR driver consume. Both are model-parametrized (`snowflake` / `dragon`).

Run from the repo root. Set the model once — every command below is model-parametrized:

```bash
source venv-qlr/bin/activate

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
python create_index.py <model> <index_type> [--param value ...]
```

- `model`: `snowflake` or `dragon`
- `index_type`: `exact`, `ivf`, `hnsw`, a comma-separated list (e.g. `exact,ivf`), or `all`

The CAsT 2019 collection is the only one indexed — it contains MS MARCO v1 (as
`MARCO_<n>`), so CAsT 2020 and QLR reuse the same index. Paths come from `EMB_BASE`
(embeddings root, expects `<model>_embeddings/`) and `CACHE_BASE` (index output); both
default to the cluster locations, so point them elsewhere to build over a subset:

```bash
EMB_BASE=~/subset_check CACHE_BASE=~/subset_check/index python create_index.py snowflake hnsw
```

**What our indexes were built with** (CAsT 2019/2020 + QLR `I_D`): IVF `2^18` centroids
(dragon) / `2^15` (snowflake), `nprobe=128`, k-means `niter=10` on 40 × centroids
training samples · HNSW `M=32`, `efConstruction=200`, `efSearch=64` · Exact
`IndexFlatIP` · all L2-normalized, `METRIC_INNER_PRODUCT` (cosine).

#### Choosing the index parameters

Omit a flag and the build uses the paper-faithful per-model default — the
positional-only form reproduces the paper; pass a flag to override it. Full paper grids:
`python create_index.py --help`.

| Flag | Applies to | Paper values | Default |
|------|-----------|--------------|---------|
| `--num-centroids` | IVF | `{32768, 65536, 131072, 262144}` = 2¹⁵–2¹⁸ (TopLoc) | dragon 262144, snowflake 32768 |
| `--nprobe` | IVF | 1…4096 in powers of 2 (TopLoc) | 128 |
| `--kmeans-niter` | IVF | 25 (FAISS default), 10 (project) | 25 |
| `--train-sample-size` | IVF | ≥ 39 × `num_centroids` (FAISS heuristic) | auto (≥ 40 × centroids) |
| `--M` | HNSW | `{16, 32, 64}` (TopLoc); 32 (QLR) | 32 |
| `--ef-construction` | HNSW | 500 (QLR); TopLoc leaves it unspecified | 200 (what the built indexes use) |
| `--ef-search` | HNSW | 1…4096 pow2 (TopLoc); 10…200 step 10 (QLR) | 64 |
| `--normalize / --no-normalize` | both | Dragon L2-normalized for cosine (TopLoc) | `--normalize` |

`--nprobe` and `--ef-search` are search-time knobs baked into the saved index as
defaults; the evaluation scripts sweep them. Keep `--normalize` on: an un-normalized
HNSW/IVF graph on Dragon degenerates (norm-bias hubs → ~0 recall).

```bash
python create_index.py dragon ivf                    # paper build, no flags needed
python create_index.py snowflake exact,ivf           # several types sequentially
python create_index.py snowflake hnsw --M 64 --ef-construction 500   # your own params
```

The build streams embeddings from parquet, checkpoints every 50 files and resumes from
the last checkpoint if rerun; per-index logs land in the cache directory (e.g.
`ivf_indexCreation.log`). **To rebuild with different parameters**, delete the three
artifacts first (`<type>_index.index`, `<type>_ids.npy`, `<type>_checkpoint.json`) —
the filename does not encode the parameters, so an existing index is otherwise reused
and the flags are ignored (with a warning).

**Dragon only — one extra step after building.** `create_index.py` writes the Dragon
index to `<CACHE_BASE>/dragon/`, but for historical reasons the HNSW eval
(`combine_hnsw.py`) and QLR (`qlr.py`) read it from `<CACHE_BASE>/dragon/_cosine_old/`
(`combine_IVF.py` reads `dragon/` directly). After a fresh Dragon build, add one symlink
so all three resolve to the same index:

```bash
ln -sfn . "$CACHE_BASE/dragon/_cosine_old"      # Dragon only; Snowflake needs nothing
```

This is safe because `create_index.py` now L2-normalizes Dragon (the `_cosine_old`
indirection dates back to when `dragon/` held an un-normalized build). Snowflake builds
and reads the same `snowflake/` directory throughout — no symlink needed.

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

# OR to run using CasT 2019 (Default Dataset DIR)

taskset -c 0-13 env NUM_THREADS=14 OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
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

## QLR (Query Log Router)

Reimplementation of "HNSW Graph Meets Query Logs": a lightweight router in front
of a standard HNSW document index. It runs on `msmarco-on-cast` — the CAST2019
HNSW index is the document index `I_D` (it already contains every msmarco passage
as `MARCO_<n>`), with the msmarco train split as the query log `Q_L` and the
msmarco dev queries + `qrels.dev.small.tsv` as the test set.

Driver: `qlr.py` (pure Python, no C++). Two metrics are reported: **Accuracy@10**
(fraction of the *exhaustive* top-10 retrieved — the paper's headline metric) and
qrels-based **NDCG@3/10 + MRR@10**.

Run everything from `toploc2/`, inside `tmux` (the one-time EP build over the full
~808k log takes ~80 min), in `venv-qlr`. That env already covers the TopLoc-HNSW step,
so if you built it there just activate it — otherwise set it up (see Setup):

```bash
tmux new -s qlr
cd toploc2

source ../venv-qlr/bin/activate                 # if venv-qlr already exists
# --- otherwise create it first: ---
# python3 -m venv ../venv-qlr && source ../venv-qlr/bin/activate
# pip install -r ../requirements-qlr.txt
```

### 0. Sanity check (no server needed)

```bash
python test_qlr_pipeline.py       # synthetic, runs in seconds -> 16/16 passed
```

Run this first in any new environment: it asserts the seeded level-0 search against a
pure-Python reference. If `faiss_level0_matches_python` fails (15/16), the env has
faiss ≥ 1.14 (flipped seed-distance signs) — reinstall from `requirements-qlr.txt`,
which pins `faiss-cpu<1.14`.

### Run (per model)

All commands use `--dataset msmarco-on-cast`.

1. **(dragon only)** encode the dev queries:
   ```bash
   python encode_msmarco_dev_dragon.py
   ```

2. **ground truth for Accuracy@10** (one-time exact top-10; streams the doc
   embeddings, does not load the HNSW index — thread it via `OMP_NUM_THREADS`):
   ```bash
   OMP_NUM_THREADS=14 python compute_groundtruth.py <model> --dataset msmarco-on-cast
   ```

3. **baseline + QLR sweep in ONE run** — `--mode both` loads the ~157 GiB index
   *once*, then runs the plain-HNSW efSearch sweep **and** the QLR sweep into one CSV
   (the `mode` column separates them; EP/s_max are built once and reused across the
   grid). A bare `--sweep` reproduces the full paper grid; override single axes with
   `--th-list/--kprime-list/--ef-list/--pca-list`:
   ```bash
   python -u qlr.py <model> --dataset msmarco-on-cast \
       --mode both --sweep --log-limit 0 --threads 1 --out <model>_sweep.csv
   ```

| Flag | Meaning | Paper grid / default |
|------|---------|----------------------|
| `--th` | routing threshold (fallback when `s < th`) | snowflake {0.3…0.7}; dragon data-driven |
| `--k-prime` | log queries retrieved per incoming query | {10, 20} |
| `--pca-dim` | PCA on `I_Q` | dim/4 = 256 (snowflake) / 192 (dragon) |
| `--ef-default` | operating point (upper bound of adaptive ef') | 100 |
| `--log-limit` | cap on \|Q_L\| (0 = full ~808k log) | 0 |
| `--mode` | `baseline` / `qlr` / `both` | qlr |

Each CSV row is one config: `mode, th, k_prime, ef, pca, Accuracy@10, NDCG@10,
MRR@10, latency_ms_per_q` (+ `qlr_route/seeded/fallback_ms_per_q` = the routed-latency
breakdown). Compare `mode=baseline` vs `mode=qlr` at matched Accuracy@10.

**On `MMAP`:** leave it unset (default off) so the index loads fully into RAM and the
latency stays comparable to `combine_hnsw.py`; the ~157 GiB snowflake index needs that
much free RAM. `MMAP=1` is safe for accuracy/`avg_visited` (identical) — only a
real-latency comparison would then not match.

### Speedup at matched accuracy (iso-accuracy)

QLR has no single "best" config: like `nprobe`/`efSearch` for IVF/HNSW, `--ef-default`
is an operating-point dial — raise it for higher Accuracy@10 at more latency. At each
Accuracy@10 target you read off the fastest config of each method that reaches it; the
speedup *grows* with the target (the paper's headline property). Example points
(`latency_ms_per_q` = faithful batched, single thread):

| Model     | QLR config at this point       | Acc@10 | QLR lat | plain-HNSW @ same Acc | Speedup |
|-----------|--------------------------------|--------|---------|-----------------------|---------|
| snowflake | `th=0.6 k'=10 pca=256 ef=70`   | 0.935  | 1.15 ms | 1.47 ms (ef=100)      | **1.28×** |
| dragon    | `th=0.7 k'=10 pca=192 ef=110`  | 0.931  | 0.72 ms | 1.09 ms (ef=170)      | **1.51×** |

dragon's plain-HNSW baseline caps at Acc@10 ≈ 0.938, so at Acc 0.95 QLR wins outright
(no baseline config reaches it).

### Notes

- The routed level-0 beam runs natively via FAISS `search_level_0` (`search_type=2`,
  no custom C++); all routed queries are batched into ONE call so the `VisitedTable`
  (~38 MB) is allocated once — `latency_ms_per_q` is therefore real single-thread
  latency. A pure-Python beam is kept only in `test_qlr_pipeline.py` as the correctness
  reference (its top-k is asserted identical to the batched FAISS path).
- snowflake `th` is cosine (default 0.5); dragon is cosine too (L2-normalised for
  HNSW/IVF navigation) but its `th` is calibrated from the data when unset, because
  its query/context encoders are asymmetric.

## Interactive Demo (Streamlit)

> **Note:** the demo is a standalone illustration and is **not** part of the
> result pipeline above. It reflects the code as of the demo presentation and
> is not kept in sync with later changes — use the pipeline sections for the
> reproducible results.

A small Streamlit app that runs conversational retrieval over a compact subset
so it fits on a laptop. Per query it shows the retrieved passages, the number of
vectors compared (an efficiency proxy), the speedup vs exact search, and live
NDCG/MRR for known CAsT turns. Exact and IVF use plain FAISS; TopLoc IVF calls
the real `toploc_search` C++ kernel (Python fallback if uncompiled). It runs in
its own conda environment (`toploc-demo`), separate from `venv`/`venv-qlr`.

```bash
# 1. Get the demo subset -> demo/data/ (generated artifact, gitignored)
cd demo
gh release download demo-data -p demo_data.zip && unzip demo_data.zip   # prebuilt (recommended)
# or rebuild where the full data lives: python demo/build_demo_subset.py

# 2. Build the env + C++ kernel (needs a C++ compiler; run once per machine)
./demo/build_demo.sh

# 3. Run
conda activate toploc-demo
streamlit run demo/demo_app.py
```

Pick a known CAsT turn (precomputed) or type a free-text query, switch between
**Exact / IVF / TopLoc IVF**, and tune `nprobe` and cached centroids `h`. The
first turn seeds the TopLoc cache; follow-up turns reuse it and scan far fewer
vectors.

## Data Structure

Every script is driven by two roots — the input embeddings/topics
(`EMB_BASE` / `DATASET_DIR` / `MSMARCO_BASE`) and the index output (`CACHE_BASE`).
Both default to the cluster layout below:

```
Datasets/conversational/
  CAST2019/
    CAST2019collection.tsv              38.6M passages (MS MARCO v1 + TREC CAR)
    topics/                             DATASET_DIR: topics.tsv, qrels.qrel (CAsT 2019)
    snowflake_embeddings/               EMB_BASE input: document embeddings (per model)
    dragon_embeddings/
    msmarco/                            MSMARCO_BASE: QLR test set
      msmarco_embeddings/dev_query/       dev queries (snowflake)
      msmarco_embeddings/dev_query_dragon/ dev queries (dragon)
      qrels.dev.small.tsv
  CAST2020/topics/                      DATASET_DIR for CAsT 2020 (same index)
  msmarco/<model>/                      QLR historical query log Q_L (train queries)

Datasets/toploc2/                       CACHE_BASE: built indexes + QLR caches (EP/I_Q/GT)
  snowflake/{exact,ivf,hnsw}_index.index  (+ *_ids.npy)
  dragon/{exact,ivf,hnsw}_index.index     dragon/_cosine_old/ is read by HNSW/QLR
                                          (see "Dragon only" under Build Index)
```

## `old scripts/`

Earlier, superseded implementations kept for reference — iterations of the TopLoc
IVF/HNSW work that led to the current `combine_hnsw.py` / `combine_IVF.py`. Not part
of the pipeline above. The one exception is `toploc_search.cpp` (+ `CMakeLists.txt`):
the demo still compiles its C++ kernel from here (see `demo/build_demo.sh`), so the
folder is not removable as a whole.
