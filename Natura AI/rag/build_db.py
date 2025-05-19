import pandas as pd
import faiss
import pickle
import os
from embedder import embed_texts


# === Load remedies CSV ===
df = pd.read_csv("rag/remedies.csv")

# === Format each remedy for embedding ===
def format_entry(row):
    return f"""
    Remedy: {row['name']}
    Conditions: {row['conditions']}
    Skin Types: {row['skin_types']}
    Severity: {row['severity_levels']}
    Ingredients: {row['ingredients']}
    Instructions: {row['instructions']}
    Cautions: {row['cautions']}
    """.strip()

# Create list of formatted documents
docs = df.apply(format_entry, axis=1).tolist()
metadata = df.to_dict(orient="records")  # For reference later

# === Generate embeddings ===
print("🔄 Generating embeddings...")
embeddings = embed_texts(docs)

# === Create FAISS index ===
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# === Save index and metadata ===
os.makedirs("db", exist_ok=True)
faiss.write_index(index, "db/db.faiss")
with open("db/embeddings.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS database and metadata saved.")

