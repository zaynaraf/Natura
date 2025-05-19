📌 MiniLM Embedding Model Setup (for FAISS embeddings)

This folder is expected to contain the MiniLM model used to embed remedies for retrieval.

Model Used:
sentence-transformers/all-MiniLM-L6-v2

Instructions to set up locally:

1. Download the model from Hugging Face:
👉 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

2. Extract or clone the contents into this folder:
./rag/embedding_model

It must match the embedding model path used in `embedder.py`.

✅ Example file structure:
rag/
└── embedding_model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── ...