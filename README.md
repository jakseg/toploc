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
- `index_type`: `exact`, `ivf`, or `hnsw`

Examples:
```bash
python create_index.py snowflake ivf
python create_index.py dragon hnsw
python create_index.py snowflake exact
```

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
