import re

import numpy as np

from embeddings import (
    text_to_vector,
    tokenize,
)


# =========================================================
# QUERY WORDS THAT MUST NOT CREATE A MATCH BY THEMSELVES
# =========================================================

QUERY_STOPWORDS = {
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "which",
    "whose",
    "whom",

    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",

    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",

    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",

    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "and",
    "or",
    "as",
    "by",

    "tell",
    "explain",
    "define",
    "describe",
    "please",
    "about",
    "me",

    "call",
    "called",
    "name",
    "named",
    "information",
    "detail",
    "details",
    "give",
    "show",
    "provide",
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_query_word(word):
    """
    Normalize a word only for relevance checking.

    The actual TF-IDF vector remains unchanged.
    """
    word = str(word or "").lower().strip()

    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"

    elif len(word) > 3 and word.endswith("s"):
        word = word[:-1]

    return word


def get_meaningful_words(text):
    """
    Get query/content words that carry subject meaning.

    Generic question words such as "what" and "how" are
    deliberately excluded so they cannot cause false matches.
    """
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        str(text or "").lower(),
    )

    meaningful = set()

    for word in words:
        normalized = normalize_query_word(
            word
        )

        if (
            normalized
            and normalized not in QUERY_STOPWORDS
            and len(normalized) > 1
        ):
            meaningful.add(
                normalized
            )

    return meaningful



def is_yes_no_question(question):
    first_word_match = re.match(
        r"^\s*([a-zA-Z]+)",
        str(question or ""),
    )

    if not first_word_match:
        return False

    return (
        first_word_match.group(1).lower()
        in {
            "is",
            "are",
            "was",
            "were",
            "can",
            "could",
            "does",
            "do",
            "did",
            "has",
            "have",
            "had",
        }
    )


def get_pool_meaningful_words(embedded_chunks):
    pool_words = set()

    for chunk in embedded_chunks or []:
        pool_words.update(
            get_meaningful_words(
                chunk.get(
                    "text",
                    "",
                )
            )
        )

    return pool_words


def query_supported_by_selected_pool(
    question,
    meaningful_query_words,
    embedded_chunks,
):
    """
    Conservative no-context gate for the CURRENT search pool.

    Normal questions require every meaningful query term to exist
    somewhere in the selected document pool.

    Yes/no questions may have one missing value because the document
    can contradict the user's proposed value.
    """
    pool_words = get_pool_meaningful_words(
        embedded_chunks
    )

    if not pool_words:
        return False

    supported = (
        meaningful_query_words
        .intersection(
            pool_words
        )
    )

    missing = (
        meaningful_query_words
        - pool_words
    )

    if not missing:
        return True

    if (
        is_yes_no_question(
            question
        )
        and len(missing) <= 1
        and len(supported) >= 2
    ):
        return True

    return False


def chunk_query_coverage(
    meaningful_query_words,
    chunk_text,
):
    if not meaningful_query_words:
        return 0.0

    chunk_words = get_meaningful_words(
        chunk_text
    )

    matched = (
        meaningful_query_words
        .intersection(
            chunk_words
        )
    )

    return (
        len(matched)
        / len(meaningful_query_words)
    )


def vocabulary_contains_meaningful_term(
    meaningful_words,
    vocabulary,
):
    """
    Confirm that at least one real subject term from the query
    exists in the indexed vocabulary.

    Example:
        "What is JavaScript?"
        -> "what" is ignored
        -> "javascript" is not indexed
        -> immediately return No Match.
    """
    normalized_vocabulary = {
        normalize_query_word(word)
        for word in vocabulary.keys()
    }

    return bool(
        meaningful_words.intersection(
            normalized_vocabulary
        )
    )


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity from scratch with NumPy.

    cosine(A, B) =
        dot(A, B) / (||A|| * ||B||)
    """
    vector1 = np.array(
        vector1,
        dtype=float,
    )

    vector2 = np.array(
        vector2,
        dtype=float,
    )

    dot_product = np.dot(
        vector1,
        vector2,
    )

    norm1 = np.linalg.norm(
        vector1
    )

    norm2 = np.linalg.norm(
        vector2
    )

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(
        dot_product
        /
        (norm1 * norm2)
    )


# =========================================================
# MATCHED KEYWORDS
# =========================================================

def get_matched_keywords(
    question,
    chunk_text,
):
    """
    Return only meaningful matched terms.

    "what", "how", "is", etc. are no longer shown as
    citation keywords.
    """
    question_words = get_meaningful_words(
        question
    )

    chunk_words = get_meaningful_words(
        chunk_text
    )

    return sorted(
        question_words.intersection(
            chunk_words
        )
    )


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_relevant_chunks(
    question,
    embedded_chunks,
    vocabulary,
    idf_values=None,
    top_k=3,
    min_score=0.01,
):
    """
    Retrieve top relevant chunks while preventing false matches.

    Important guards:
    1. The query must contain a meaningful subject word.
    2. At least one meaningful query word must exist in the
       vector-store vocabulary.
    3. A returned chunk must share at least one meaningful
       query word.
    4. Cosine similarity must still pass min_score.

    This prevents cases such as:
        "What is JavaScript?"
        -> matching an animal chunk only because both contain "what".

        "Explain blockchain mining."
        -> returning a fish/cow answer even though those terms are
           absent from the indexed documents.
    """

    meaningful_query_words = get_meaningful_words(
        question
    )

    # No real subject words.
    if not meaningful_query_words:
        return []

    # The currently selected document pool must support the
    # meaningful terms in the user's question.
    #
    # Example:
    #   "What is the password for Aurora Research Station?"
    #
    # The entity exists, but "password" does not. Returning [] here
    # correctly triggers the application's No Context response.
    if not query_supported_by_selected_pool(
        question,
        meaningful_query_words,
        embedded_chunks,
    ):
        return []

    question_vector = text_to_vector(
        question,
        vocabulary,
        idf_values,
    )

    # If the TF-IDF question vector is empty, retrieval cannot
    # produce a legitimate cosine match.
    if np.linalg.norm(
        np.array(
            question_vector,
            dtype=float,
        )
    ) == 0:
        return []

    scored_chunks = []

    for chunk in embedded_chunks:

        embedding = chunk.get(
            "embedding"
        )

        if embedding is None:
            continue

        chunk_text = chunk.get(
            "text",
            "",
        )

        matched_keywords = (
            get_matched_keywords(
                question,
                chunk_text,
            )
        )

        # This is the critical false-positive guard.
        if not matched_keywords:
            continue

        coverage = chunk_query_coverage(
            meaningful_query_words,
            chunk_text,
        )

        minimum_coverage = (
            0.66
            if is_yes_no_question(
                question
            )
            else 1.0
        )

        if coverage < minimum_coverage:
            continue

        score = cosine_similarity(
            question_vector,
            embedding,
        )

        if score < min_score:
            continue

        scored_chunks.append({

            "filename":
                chunk.get(
                    "filename",
                    "Unknown document",
                ),

            "chunk_id":
                chunk.get(
                    "chunk_id",
                ),

            "page":
                chunk.get(
                    "page",
                    chunk.get(
                        "page_number",
                    ),
                ),

            # Keep both names for compatibility with your UI.
            "page_number":
                chunk.get(
                    "page_number",
                    chunk.get(
                        "page",
                    ),
                ),

            "text":
                chunk_text,

            "score":
                float(score),

            "matched_keywords":
                matched_keywords,

            "query_coverage":
                float(coverage),
        })

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]
