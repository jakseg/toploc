import os
os.environ["OMP_NUM_THREADS"] = "1" #remove when running on cluster. Needed for dev on mac

import csv
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

os.makedirs("data/dragon", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-context-encoder")
model = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")
model.eval()

subset_size = 20000
passage_ids = []
passage_texts = []

with open("dataset/head_2000_rows.tsv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)

    for i, row in enumerate(reader):
        if i >= subset_size:
            break
        passage_ids.append(row[0])
        passage_texts.append(row[1])

print(f"Loaded {len(passage_texts)} passages")

batch_size = 64
all_embeddings = []

for start in range(0, len(passage_texts), batch_size):
    end = min(start + batch_size, len(passage_texts))
    batch = passage_texts[start:end]

    tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)
        embs = outputs.last_hidden_state[:, 0, :]  # CLS token

    # L2 normalize (paper: mandatory normalization for cosine via inner product)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    all_embeddings.append(embs.cpu().numpy())

    if (start // batch_size) % 5 == 0:
        print(f"  {end}/{len(passage_texts)} passages encoded")

passage_embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")

np.save("data/dragon/passage_embeddings.npy", passage_embeddings)

with open("data/dragon/passage_id_map.json", "w") as f:
    json.dump({str(i): pid for i, pid in enumerate(passage_ids)}, f)

print(f"Saved:")
print(f"  - data/dragon/passage_embeddings.npy: {passage_embeddings.shape}")
print(f"  - data/dragon/passage_id_map.json: {len(passage_ids)} IDs mapped")
