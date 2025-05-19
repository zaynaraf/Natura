import os
import pickle
import faiss
import numpy as np
from rag.embedder import embed_texts  # Should use MiniLM embeddings
import pandas as pd

# Load FAISS index + metadata
base_path = os.path.dirname(__file__)
index_path = os.path.join(base_path, "../db/db.faiss")
meta_path = os.path.join(base_path, "../db/embeddings.pkl")

index = faiss.read_index(index_path)
with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

# Load full remedy CSV (for full access to instructions, etc.)
csv_path = os.path.join(base_path, "remedies.csv")
df = pd.read_csv(csv_path)

def retrieve(query: str, top_k: int = 1) -> list:
    query_vec = embed_texts([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(df):
            row = df.iloc[idx]
            results.append({
                "name": row["name"],
                "instructions": row["instructions"],
                "source_title": row["source_title"],
                "source_url": row["source_url"],
                "cautions": row.get("cautions", "")
            })
    return results
