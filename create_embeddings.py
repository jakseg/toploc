import csv
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

# Read subset with IDs
subset_size = 20000
passage_ids = []
passage_texts = []

with open("dataset/head_2000_rows.tsv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)  # Skip header row (id, text)

    for i, row in enumerate(reader):
        if i >= subset_size:
            break
        passage_ids.append(row[0])  # MARCO_0, MARCO_1, etc.
        passage_texts.append(row[1])  # The actual text

print(f"Loaded {len(passage_texts)} passages")
print(f"Sample ID: {passage_ids[0]}")
print(f"Sample text: {passage_texts[0][:100]}...")

# Generate embeddings
passage_embeddings = model.encode(
    passage_texts,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

# Save embeddings
np.save("data/passage_embeddings_subset.npy", passage_embeddings)

# Save ID mapping (CRITICAL!)
with open("data/passage_id_map.json", "w") as f:
    json.dump({str(i): pid for i, pid in enumerate(passage_ids)}, f)

print(f"Saved:")
print(f"  - data/passage_embeddings_subset.npy: {passage_embeddings.shape}")
print(f"  - data/passage_id_map.json: {len(passage_ids)} IDs mapped")