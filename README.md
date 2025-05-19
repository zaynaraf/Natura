Natura — Intelligent Natural Skincare Assistant

Natura is an advanced AI-powered skincare assistant designed for personalized natural remedy recommendations. This project combines cutting-edge large language models (LLMs) with a Retrieval-Augmented Generation (RAG) pipeline to deliver precise, contextually grounded skincare advice tailored to individual skin profiles.

Built with OpenChat 3.5 (4-bit quantized), and enhanced by fine tuning with QLORA on natural remedies and integrated with a structured remedy database, Natura demonstrates a robust implementation of conversational AI in the wellness domain.

---

Overview

The goal of Natura is to simulate a professional yet friendly natural skincare advisor capable of engaging in dynamic conversations, understanding user concerns, and offering relevant, evidence-based natural treatments. Unlike generic chatbots, Natura uses semantic retrieval to ground its responses in an authenticated remedy database, ensuring recommendations are context-aware, accurate, and explainable.

---

Features

- LLM-Driven Dialogue: Utilizes OpenChat 3.5 in 4-bit quantization for efficient, high-quality responses.
- RAG Integration: Embeds and indexes remedies using MiniLM for fast, semantic vector retrieval via FAISS.
- Contextual Awareness: Gathers detailed user input including skin type, primary concern, and ingredient sensitivities before suggesting remedies.
- Strict Source Attribution: Every remedy recommendation is linked to a verified source with a direct citation and URL.
- Allergy & Sensitivity Filtering: Avoids ingredients flagged by the user as allergens or irritants.
- Scalable Architecture: Designed for local GPU inference with memory-efficient execution and reproducible results.

---

Directory Structure

Natura/
├── main.py                  # Core loop and LLM generation
├── rag/
│   ├── build_db.py          # Builds FAISS index from remedies
│   ├── retriever.py         # Vector retrieval logic
│   ├── embedder.py          # Sentence embedding using MiniLM
│   └── remedies.csv         # Curated database of natural treatments
├── models/
│   └── [OpenChat-3.5 model] # (Download instructions provided)
├── embedding_model/
│   └── [MiniLM model files] # (Download instructions provided)
└── README.md

---

Setup Instructions

1. Clone the repository
   git clone https://github.com/your-username/Natura.git
   cd Natura

2. Create a virtual environment
   conda create -n natura_env python=3.10
   conda activate natura_env

3. Install dependencies
   pip install -r requirements.txt

4. Download models
   - Place the OpenChat 3.5 4-bit model in models/openchat-3.5-0106/
   - Place the MiniLM-L6-v2 files in rag/embedding_model/
   - See the provided .txt instructions in those folders for download details

5. Build the retrieval index
   python rag/build_db.py

6. Run the assistant
   python main.py

---

Model Details

- Language Model: OpenChat-3.5-0106 (4-bit quantized using BitsAndBytes)
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Retriever: FAISS (L2 distance, dense vectors)
- Prompting Strategy: Custom role-based instruction, structured context injection, deterministic or temperature sampling
- Inference Engine: Transformers (HF) + Torch CUDA backend

---

Example Use Case

User: I have oily skin with occasional breakouts. What can I use naturally?
Assistant: Based on your skin profile, I found this remedy:
Remedy: Tea Tree Oil Spot Treatment
Instructions: Dilute 3 drops of tea tree oil in 1 tsp jojoba oil. Apply to acne-prone areas.
Source: Natural Acne Treatments — https://www.example.com/remedy
Caution: Avoid undiluted use. Perform a patch test before full application.

---

License

This repository is made available for research and educational purposes. Please consult a dermatologist before applying any remedy in production or consumer-facing tools.

---

Author

Developed and maintained by [Arib Zayn Araf & Mahfooz Anas]
For inquiries or professional collaboration, please contact via GitHub or email.
