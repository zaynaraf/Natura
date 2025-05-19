from sentence_transformers import SentenceTransformer, models
import os

# Get absolute path to the local MiniLM model
model_dir = os.path.join(os.path.dirname(__file__), "embedding_model")

# Load transformer and pooling models
word_embedding_model = models.Transformer(model_dir)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# Combine into full SentenceTransformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def embed_texts(texts):
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)



