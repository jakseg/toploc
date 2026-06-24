import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import faiss
import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

model_name = sys.argv[1] if len(sys.argv) > 1 else "snowflake"
data_dir = f"data/{model_name}"

print(f"Testing IVF Pipeline ({model_name})...")

# ================= 1. Load Components =================

ivf_index = faiss.read_index(f"{data_dir}/ivf_index.index")
print(f"Loaded IVF index: {ivf_index.ntotal} vectors, trained={ivf_index.is_trained}")

with open(f"{data_dir}/passage_id_map.json", "r") as f:
    id_map = json.load(f)
print(f"Loaded ID mapping: {len(id_map)} passages")

# Set nprobe (how many clusters to scan). 8 is safe for a small subset.
ivf_index.nprobe = 8
print(f"Set nprobe={ivf_index.nprobe}")

# ================= 2. Load Model & Embed Query =================

test_query = "What was the Chinese Exclusion Acts In the 1850s?"

if model_name == "snowflake":
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
    print("Snowflake model loaded")
    # CRITICAL: normalize_embeddings=True matches how passages were encoded
    query_emb = model.encode(
        [test_query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")
elif model_name == "dragon":
    tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
    model = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
    model.eval()
    print("Dragon query encoder loaded")
    tokens = tokenizer([test_query], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        query_emb = outputs.last_hidden_state[:, 0, :]
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    query_emb = query_emb.cpu().numpy().astype("float32")
else:
    raise ValueError(f"Unknown model: {model_name}")

print(f"Query embedded: shape={query_emb.shape}, dim={query_emb.shape[1]}")

# ================= 3. Run Search =================

k = 5
scores, indices = ivf_index.search(query_emb, k)

# ================= 4. Map & Print Results =================

print(f"\nQuery: '{test_query}'")
print(f"Top {k} retrieved passages:")
for rank in range(k):
    idx = int(indices[0][rank])
    score = float(scores[0][rank])
    passage_id = id_map.get(str(idx), "UNKNOWN_ID")
    print(f"  Rank {rank+1:2d} | ID: {passage_id:<12} | Score: {score:.4f}")

# ================= 5. Sanity Checks =================

assert ivf_index.is_trained, "Index is not trained!"
assert query_emb.shape[1] == ivf_index.d, "Query dimension mismatch!"
assert all(str(i) in id_map for i in indices[0]), "ID mapping incomplete!"

print("\nAll sanity checks passed. Pipeline is working correctly!")
