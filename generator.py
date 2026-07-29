import os
import re

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from embeddings import tokenize


# Load values from the .env file in the project root.
load_dotenv(override=True)


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
).strip()

GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
).strip()

GEMINI_TIMEOUT = float(
    os.getenv(
        "GEMINI_TIMEOUT",
        "45",
    )
)

GEMINI_ENABLED = os.getenv(
    "GEMINI_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


MINIMUM_RAG_SCORE = 0.01
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHARACTERS = 12000


STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "how",
    "which",
    "who",
    "when",
    "where",
    "why",
    "does",
    "do",
    "did",
    "this",
    "that",
    "in",
    "on",
    "of",
    "to",
    "and",
    "or",
    "with",
    "using",
    "be",
}


def normalize_word(word):
    """
    Applies simple word normalization.
    """

    word = word.lower().strip()

    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]

    return word


def split_into_sentences(text):
    """
    Splits text into sentences for the local fallback generator.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        str(text or ""),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def get_question_keywords(question):
    """
    Extracts useful keywords from the user question.
    """

    keywords = []

    for word in tokenize(question):
        normalized_word = normalize_word(word)

        if normalized_word not in STOPWORDS:
            keywords.append(normalized_word)

    return keywords


def sentence_match_score(
    sentence,
    question_keywords,
):
    """
    Calculates a simple keyword-match score.
    """

    sentence_words = {
        normalize_word(word)
        for word in tokenize(sentence)
    }

    return sum(
        1
        for keyword in question_keywords
        if keyword in sentence_words
    )


def get_useful_chunks(relevant_chunks):
    """
    Keeps valid chunks that meet the minimum similarity score.
    """

    if not relevant_chunks:
        return []

    useful_chunks = []

    for chunk in relevant_chunks:
        score = float(
            chunk.get(
                "score",
                0,
            )
        )

        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        if score >= MINIMUM_RAG_SCORE and text:
            useful_chunks.append(chunk)

    useful_chunks.sort(
        key=lambda item: float(
            item.get(
                "score",
                0,
            )
        ),
        reverse=True,
    )

    return useful_chunks[:MAX_CONTEXT_CHUNKS]


def build_gemini_context(useful_chunks):
    """
    Creates document context sent to Gemini.
    """

    context_parts = []
    current_length = 0

    for index, chunk in enumerate(
        useful_chunks,
        start=1,
    ):
        filename = chunk.get(
            "filename",
            "Unknown document",
        )

        chunk_id = chunk.get(
            "chunk_id",
            "-",
        )

        score = round(
            float(
                chunk.get(
                    "score",
                    0,
                )
            ),
            4,
        )

        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        context_block = (
            f"SOURCE {index}\n"
            f"Filename: {filename}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Similarity Score: {score}\n"
            f"Content:\n{text}\n"
        )

        if (
            current_length + len(context_block)
            > MAX_CONTEXT_CHARACTERS
        ):
            break

        context_parts.append(context_block)
        current_length += len(context_block)

    return "\n---\n".join(context_parts)


def build_citations(useful_chunks):
    """
    Builds deterministic citations locally.
    """

    citation_lines = []
    seen_citations = set()

    for chunk in useful_chunks:
        filename = chunk.get(
            "filename",
            "Unknown document",
        )

        chunk_id = chunk.get(
            "chunk_id",
            "-",
        )

        score = round(
            float(
                chunk.get(
                    "score",
                    0,
                )
            ),
            4,
        )

        citation_key = (
            str(filename),
            str(chunk_id),
        )

        if citation_key in seen_citations:
            continue

        seen_citations.add(citation_key)

        citation_lines.append(
            f"- {filename} | "
            f"Chunk ID: {chunk_id} | "
            f"Score: {score}"
        )

    return "\n".join(citation_lines)


def generate_local_fallback_answer(
    question,
    useful_chunks,
):
    """
    Generates a local extractive answer when Gemini is unavailable.
    """

    question_keywords = get_question_keywords(
        question
    )

    best_sentence = None
    best_sentence_score = -1
    best_chunk_score = -1

    for chunk in useful_chunks:
        chunk_score = float(
            chunk.get(
                "score",
                0,
            )
        )

        sentences = split_into_sentences(
            chunk.get(
                "text",
                "",
            )
        )

        for sentence in sentences:
            match_score = sentence_match_score(
                sentence,
                question_keywords,
            )

            if (
                match_score > best_sentence_score
                or (
                    match_score == best_sentence_score
                    and chunk_score > best_chunk_score
                )
            ):
                best_sentence = sentence
                best_sentence_score = match_score
                best_chunk_score = chunk_score

    if not best_sentence:
        best_sentence = useful_chunks[0].get(
            "text",
            "",
        )

    return str(best_sentence).strip()


def generate_gemini_answer(
    question,
    useful_chunks,
):
    """
    Uses Google Gemini through the OpenAI-compatible endpoint.

    Gemini receives only the document chunks retrieved by the
    local TF-IDF and cosine-similarity pipeline.
    """

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    context = build_gemini_context(
        useful_chunks
    )

    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL,
        timeout=GEMINI_TIMEOUT,
        max_retries=2,
    )

    system_message = """
You are a document-grounded RAG assistant.

Follow these rules strictly:

1. Answer only from the supplied retrieved document context.
2. Do not use outside knowledge.
3. Do not invent facts, names, values, steps, or citations.
4. If the context does not answer the question, say exactly:
   "I could not find enough relevant information in the uploaded documents."
5. Give a clear and concise answer.
6. Do not create a citation section.
7. Do not mention Gemini, Google, prompts, or system instructions.
8. Do not claim that you searched files outside the supplied context.
""".strip()

    user_message = f"""
USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

Write the final answer using only the retrieved document context.
""".strip()

    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise ValueError(
            "Gemini returned an empty answer."
        )

    return answer.strip()


def generate_answer(
    question,
    relevant_chunks,
):
    """
    Generates a grounded answer using Gemini.

    If Gemini is unavailable, the local extractive generator is used.
    Citations are hidden when no relevant answer is found.
    """

    no_context_message = (
        "I could not find enough relevant information "
        "in the uploaded documents."
    )

    useful_chunks = get_useful_chunks(
        relevant_chunks
    )

    # No valid retrieved chunks.
    if not useful_chunks:
        return (
            "Based on the uploaded documents:\n\n"
            f"{no_context_message}"
        )

    generation_mode = "local"

    # ---------------- GEMINI GENERATION ----------------

    if GEMINI_ENABLED:
        try:
            answer_text = generate_gemini_answer(
                question,
                useful_chunks,
            )

            generation_mode = "gemini"

        except RateLimitError as exc:
            print(
                "[Gemini RateLimitError]",
                str(exc),
            )

            answer_text = generate_local_fallback_answer(
                question,
                useful_chunks,
            )

        except APIStatusError as exc:
            print(
                "[Gemini APIStatusError]",
                f"Status: {exc.status_code}",
                f"Details: {exc}",
            )

            answer_text = generate_local_fallback_answer(
                question,
                useful_chunks,
            )

        except APIConnectionError as exc:
            print(
                "[Gemini APIConnectionError]",
                str(exc),
            )

            answer_text = generate_local_fallback_answer(
                question,
                useful_chunks,
            )

        except APITimeoutError as exc:
            print(
                "[Gemini APITimeoutError]",
                str(exc),
            )

            answer_text = generate_local_fallback_answer(
                question,
                useful_chunks,
            )

        except Exception as exc:
            print(
                "[Gemini Unexpected Error]",
                type(exc).__name__,
                str(exc),
            )

            answer_text = generate_local_fallback_answer(
                question,
                useful_chunks,
            )

    else:
        answer_text = generate_local_fallback_answer(
            question,
            useful_chunks,
        )

    answer_text = str(
        answer_text or ""
    ).strip()

    # Gemini/local generator confirms that context does not answer it.
    if (
        not answer_text
        or no_context_message.lower()
        in answer_text.lower()
    ):
        return (
            "Based on the uploaded documents:\n\n"
            f"{no_context_message}"
        )

    # ---------------- CITATIONS ----------------

    citations = build_citations(
        useful_chunks
    )

    answer = (
        "Based on the uploaded documents:\n\n"
        f"{answer_text}\n\n"
        "Citations:\n"
        f"{citations}"
    )

    if generation_mode == "local" and GEMINI_ENABLED:
        answer += (
            "\n\n"
            "Note: The local fallback generator was used "
            "because Gemini was unavailable."
        )

    return answer