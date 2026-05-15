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
python create_index.py <model> <index_type>
```

- `model`: `snowflake` or `dragon`
- `index_type`: `exact`, `ivf`, `hnsw`, a comma-separated list (e.g. `exact,ivf`), or `all`

Examples:
```bash
python create_index.py snowflake ivf           # single index
python create_index.py snowflake exact,ivf     # build two sequentially
python create_index.py snowflake all           # exact + ivf + hnsw
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

## Data Structure

```
dataset/              Raw data (passages, topics, qrels)
data/snowflake/       Snowflake embeddings, id map, and indices
data/dragon/          Dragon embeddings, id map, and indices
```
