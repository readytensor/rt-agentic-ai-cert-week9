from sklearn.metrics.pairwise import cosine_similarity


def embed_and_score_similarity(text1: str, text2: str, embedder) -> float:
    """
    Compute cosine similarity between two texts using a provided embedder.

    Args:
        text1 (str): First text input.
        text2 (str): Second text input.
        embedder: An embedding model instance (e.g., from HuggingFace).

    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    emb1 = embedder.embed_query(text1)
    emb2 = embedder.embed_query(text2)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity
