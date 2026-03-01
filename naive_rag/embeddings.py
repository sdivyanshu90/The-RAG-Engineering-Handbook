from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_text(text: str):
    return embedding_model.encode(
        [text],
        normalize_embeddings=True
    ).tolist()[0]


def get_embedding_dim():
    return len(embed_text("test"))