"""Sentence-transformers embedder with lazy per-process model load."""

import numpy as np

_MODEL = None
_MODEL_NAME = None


def _get_model(model_name):
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != model_name:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
    return _MODEL


def encode(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return embeddings


def cosine_sim_matrix(a, b):
    """Cosine similarity matrix. a:(n,d), b:(m,d) -> (n,m)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T
