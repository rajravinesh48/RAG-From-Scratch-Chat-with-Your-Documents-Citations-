import numpy as np
from embeddings import text_to_vector, tokenize


def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_matched_keywords(question, chunk_text):
    question_words = set(tokenize(question))
    chunk_words = set(tokenize(chunk_text))

    matched = question_words.intersection(chunk_words)

    return sorted(list(matched))


def retrieve_relevant_chunks(
    question,
    embedded_chunks,
    vocabulary,
    idf_values=None,
    top_k=3,
    min_score=0.01
):
    question_vector = text_to_vector(question, vocabulary, idf_values)

    scored_chunks = []

    for chunk in embedded_chunks:
        score = cosine_similarity(question_vector, chunk["embedding"])

        if score >= min_score:
            scored_chunks.append({
                "filename": chunk.get("filename"),
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
                "matched_keywords": get_matched_keywords(question, chunk["text"])
            })

    scored_chunks = sorted(
        scored_chunks,
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_chunks[:top_k]