import re
import math
import numpy as np


STOPWORDS = {
    "the", "is", "are", "a", "an", "and", "or", "to", "of", "in",
    "on", "for", "with", "this", "that", "it", "as", "by", "from"
}


def tokenize(text):
    text = text.lower()
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text)

    clean_words = []

    for word in words:
        if word not in STOPWORDS:
            clean_words.append(word)

    return clean_words


def build_vocabulary(chunks):
    vocabulary = {}

    for chunk in chunks:
        words = tokenize(chunk["text"])

        for word in words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    return vocabulary


def calculate_idf(chunks, vocabulary):
    total_documents = len(chunks)
    idf_values = {}

    for word in vocabulary:
        document_count = 0

        for chunk in chunks:
            words = tokenize(chunk["text"])

            if word in words:
                document_count += 1

        idf_values[word] = math.log((total_documents + 1) / (document_count + 1)) + 1

    return idf_values


def text_to_vector(text, vocabulary, idf_values=None):
    vector = np.zeros(len(vocabulary))

    words = tokenize(text)

    if not words:
        return vector

    word_frequency = {}

    for word in words:
        word_frequency[word] = word_frequency.get(word, 0) + 1

    for word, count in word_frequency.items():
        if word in vocabulary:
            index = vocabulary[word]

            tf = count / len(words)

            if idf_values and word in idf_values:
                vector[index] = tf * idf_values[word]
            else:
                vector[index] = tf

    return vector


def create_chunk_embeddings(chunks):
    vocabulary = build_vocabulary(chunks)
    idf_values = calculate_idf(chunks, vocabulary)

    embedded_chunks = []

    for chunk in chunks:
        vector = text_to_vector(chunk["text"], vocabulary, idf_values)

        embedded_chunks.append({
            "filename": chunk.get("filename"),
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": vector.tolist()
        })

    return embedded_chunks, vocabulary, idf_values