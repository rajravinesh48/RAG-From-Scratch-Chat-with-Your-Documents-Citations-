import json
import os
import re
import urllib.error
import urllib.request

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from embeddings import tokenize


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv(override=True)


# -------------------------
# DEFAULT LLM PROVIDER
# -------------------------

DEFAULT_LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama",
).strip().lower()


# -------------------------
# OLLAMA
# -------------------------

OLLAMA_ENABLED = os.getenv(
    "OLLAMA_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).strip().rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:4b",
).strip()

OLLAMA_TIMEOUT = float(
    os.getenv(
        "OLLAMA_TIMEOUT",
        "120",
    )
)


# -------------------------
# GEMINI
# -------------------------

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


# =========================================================
# RAG SETTINGS
# =========================================================

MINIMUM_RAG_SCORE = 0.01
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHARACTERS = 12000

NO_CONTEXT_MESSAGE = (
    "I could not find enough relevant information "
    "in the uploaded documents."
)


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


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_word(word):
    word = word.lower().strip()

    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]

    return word


def split_into_sentences(text):
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
    sentence_words = {
        normalize_word(word)
        for word in tokenize(sentence)
    }

    return sum(
        1
        for keyword in question_keywords
        if keyword in sentence_words
    )


# =========================================================
# RETRIEVED CHUNK HELPERS
# =========================================================

def get_useful_chunks(relevant_chunks):
    """
    Keeps valid retrieved chunks that meet the minimum
    similarity threshold.
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

        if (
            score >= MINIMUM_RAG_SCORE
            and text
        ):
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


def build_llm_context(useful_chunks):
    """
    Builds the same grounded RAG context for both
    Ollama and Gemini.

    PDF page metadata is included when available.
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

        page_number = chunk.get(
            "page_number"
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

        page_line = (
            f"Page: {page_number}\n"
            if page_number is not None
            else ""
        )

        context_block = (
            f"SOURCE {index}\n"
            f"Filename: {filename}\n"
            f"{page_line}"
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


# Backward compatibility with older imports/code.
def build_gemini_context(useful_chunks):
    return build_llm_context(
        useful_chunks
    )


def build_citations(useful_chunks):
    """
    Builds deterministic citation text.

    The web UI renders source cards separately, but this
    helper is retained for compatibility.
    """
    citation_lines = []
    seen_citations = set()

    for chunk in useful_chunks:
        filename = chunk.get(
            "filename",
            "Unknown document",
        )

        page_number = chunk.get(
            "page_number"
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
            str(page_number),
            str(chunk_id),
        )

        if citation_key in seen_citations:
            continue

        seen_citations.add(citation_key)

        page_text = (
            f"Page: {page_number} | "
            if page_number is not None
            else ""
        )

        citation_lines.append(
            f"- {filename} | "
            f"{page_text}"
            f"Chunk ID: {chunk_id} | "
            f"Score: {score}"
        )

    return "\n".join(citation_lines)


# =========================================================
# PROMPTS
# =========================================================

def get_system_message():
    return """
You are a document-grounded RAG assistant.

Follow these rules strictly:

1. Answer only from the supplied retrieved document context.
2. Do not use outside knowledge.
3. Do not invent facts, names, values, steps, or citations.
4. If the context does not answer the question, say exactly:
   "I could not find enough relevant information in the uploaded documents."
5. Give a clear, concise and natural answer.
6. Do not create a separate citation section.
7. Do not mention Ollama, Gemma, Gemini, Google, prompts,
   model providers, or system instructions.
8. Do not claim that you searched files outside the supplied context.
9. Return only the direct answer.
10. Do not start with phrases such as:
    "Based on the provided document",
    "According to the context",
    or "The supplied context says".
""".strip()


def get_user_message(
    question,
    useful_chunks,
):
    context = build_llm_context(
        useful_chunks
    )

    return f"""
USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

Write the final answer using only the retrieved document context.
""".strip()


# =========================================================
# LOCAL EXTRACTIVE FALLBACK
# =========================================================

def generate_local_fallback_answer(
    question,
    useful_chunks,
):
    """
    Final emergency fallback.

    This is NOT an LLM. It chooses the most relevant
    sentence from the retrieved document chunks.
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


# =========================================================
# OLLAMA / LOCAL LLM
# =========================================================

def normalize_ollama_model(model):
    """
    Accepts:
        gemma3:4b
        ollama:gemma3:4b

    Returns:
        gemma3:4b
    """
    selected = str(
        model or OLLAMA_MODEL
    ).strip()

    if selected.lower().startswith(
        "ollama:"
    ):
        selected = selected[
            len("ollama:"):
        ].strip()

    return selected or OLLAMA_MODEL


def generate_ollama_answer(
    question,
    useful_chunks,
    model=None,
):
    """
    Uses a local Ollama model through POST /api/chat.

    No API key is required for a normal local Ollama server.
    """
    if not OLLAMA_ENABLED:
        raise RuntimeError(
            "Ollama is disabled."
        )

    model_to_use = normalize_ollama_model(
        model
    )

    payload = {
        "model": model_to_use,
        "messages": [
            {
                "role": "system",
                "content": get_system_message(),
            },
            {
                "role": "user",
                "content": get_user_message(
                    question,
                    useful_chunks,
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    request_data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    api_url = (
        f"{OLLAMA_BASE_URL}/api/chat"
    )

    http_request = urllib.request.Request(
        api_url,
        data=request_data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            http_request,
            timeout=OLLAMA_TIMEOUT,
        ) as response:
            response_body = (
                response.read().decode(
                    "utf-8"
                )
            )

    except urllib.error.HTTPError as exc:
        try:
            details = exc.read().decode(
                "utf-8",
                errors="ignore",
            )
        except Exception:
            details = str(exc)

        raise RuntimeError(
            "Ollama returned HTTP "
            f"{exc.code}: {details}"
        ) from exc

    except urllib.error.URLError as exc:
        raise ConnectionError(
            "Could not connect to Ollama at "
            f"{OLLAMA_BASE_URL}. Make sure "
            "Ollama is installed and running."
        ) from exc

    except TimeoutError as exc:
        raise TimeoutError(
            "Ollama request timed out."
        ) from exc

    try:
        response_json = json.loads(
            response_body
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ollama returned invalid JSON."
        ) from exc

    answer = (
        response_json
        .get("message", {})
        .get("content", "")
    )

    if not answer or not str(answer).strip():
        error_message = response_json.get(
            "error"
        )

        if error_message:
            raise RuntimeError(
                f"Ollama error: {error_message}"
            )

        raise ValueError(
            "Ollama returned an empty answer."
        )

    return str(answer).strip()


# =========================================================
# GEMINI / CLOUD LLM
# =========================================================

def normalize_gemini_model(model):
    """
    Accepts:
        gemini-3.5-flash
        gemini:gemini-3.5-flash
    """
    selected = str(
        model or GEMINI_MODEL
    ).strip()

    if selected.lower().startswith(
        "gemini:"
    ):
        selected = selected[
            len("gemini:"):
        ].strip()

    return selected or GEMINI_MODEL


def generate_gemini_answer(
    question,
    useful_chunks,
    model=None,
):
    """
    Uses Gemini through Google's OpenAI-compatible endpoint.
    """
    if not GEMINI_ENABLED:
        raise RuntimeError(
            "Gemini is disabled."
        )

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    model_to_use = normalize_gemini_model(
        model
    )

    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL,
        timeout=GEMINI_TIMEOUT,
        max_retries=1,
    )

    response = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {
                "role": "system",
                "content": get_system_message(),
            },
            {
                "role": "user",
                "content": get_user_message(
                    question,
                    useful_chunks,
                ),
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise ValueError(
            "Gemini returned an empty answer."
        )

    return answer.strip()


# =========================================================
# PROVIDER SELECTION
# =========================================================

def detect_provider_from_model(model):
    """
    Explicit values supported:
        ollama:gemma3:4b
        gemini:gemini-3.5-flash

    Bare gemini-* values are treated as Gemini for backward
    compatibility with the current web_app.py.

    Bare local model values such as gemma3:4b are Ollama.
    """
    model_text = str(
        model or ""
    ).strip()

    lowered = model_text.lower()

    if lowered.startswith("ollama:"):
        return "ollama"

    if lowered.startswith("gemini:"):
        return "gemini"

    if lowered.startswith("gemini-"):
        return "gemini"

    if model_text:
        return "ollama"

    if DEFAULT_LLM_PROVIDER in {
        "ollama",
        "gemini",
    }:
        return DEFAULT_LLM_PROVIDER

    return "ollama"


def format_model_display_name(
    provider,
    model,
):
    """
    Returns a friendly label for the model that actually
    generated the answer.
    """
    provider = str(
        provider or ""
    ).strip().lower()

    model = str(
        model or ""
    ).strip()

    if provider == "ollama":
        if model.lower().startswith(
            "gemma3:"
        ):
            size = model.split(
                ":",
                1,
            )[1].upper()

            return (
                f"Ollama · Gemma 3 {size}"
            )

        return (
            f"Ollama · {model}"
        )

    if provider == "gemini":
        display_model = (
            model
            .replace("-", " ")
            .title()
        )

        return (
            f"Gemini · {display_model}"
        )

    if provider == "fallback":
        return "Local Extractive Fallback"

    if provider == "none":
        return "No LLM · No Context"

    return model or "Unknown"


def build_generation_result(
    answer,
    provider,
    model,
    preferred_provider=None,
):
    """
    Standard response returned to web_app.py.
    """
    provider = str(
        provider or "none"
    ).strip().lower()

    model = str(
        model or ""
    ).strip()

    preferred_provider = str(
        preferred_provider or provider
    ).strip().lower()

    fallback_used = (
        provider == "fallback"
        or (
            provider not in {
                "none",
                preferred_provider,
            }
        )
    )

    return {
        "answer": clean_generated_answer(
            answer
        ),
        "provider": provider,
        "model": model,
        "model_label": format_model_display_name(
            provider,
            model,
        ),
        "fallback_used": fallback_used,
    }


def clean_generated_answer(answer_text):
    answer_text = str(
        answer_text or ""
    ).strip()

    if (
        not answer_text
        or NO_CONTEXT_MESSAGE.lower()
        in answer_text.lower()
    ):
        return NO_CONTEXT_MESSAGE

    return answer_text


# =========================================================
# PUBLIC GENERATOR
# =========================================================

def generate_answer(
    question,
    relevant_chunks,
    model=None,
):
    """
    Generates a grounded answer and returns both the answer
    and details about the generator that ACTUALLY answered.

    Result:
        {
            "answer": "...",
            "provider": "ollama" | "gemini" | "fallback" | "none",
            "model": "...",
            "model_label": "...",
            "fallback_used": True | False,
        }
    """
    useful_chunks = get_useful_chunks(
        relevant_chunks
    )

    preferred_provider = (
        detect_provider_from_model(
            model
        )
    )

    # No useful retrieved context: do not call any LLM.
    if not useful_chunks:
        return build_generation_result(
            answer=NO_CONTEXT_MESSAGE,
            provider="none",
            model="",
            preferred_provider=preferred_provider,
        )

    # =====================================================
    # OLLAMA SELECTED
    # =====================================================

    if preferred_provider == "ollama":

        if OLLAMA_ENABLED:
            try:
                actual_model = (
                    normalize_ollama_model(
                        model
                    )
                )

                answer = (
                    generate_ollama_answer(
                        question,
                        useful_chunks,
                        model=actual_model,
                    )
                )

                return build_generation_result(
                    answer=answer,
                    provider="ollama",
                    model=actual_model,
                    preferred_provider=preferred_provider,
                )

            except Exception as exc:
                print(
                    "[Ollama Error]",
                    type(exc).__name__,
                    str(exc),
                )

        # Ollama unavailable -> Gemini fallback.
        if (
            GEMINI_ENABLED
            and GEMINI_API_KEY
        ):
            try:
                actual_model = GEMINI_MODEL

                answer = (
                    generate_gemini_answer(
                        question,
                        useful_chunks,
                        model=actual_model,
                    )
                )

                return build_generation_result(
                    answer=answer,
                    provider="gemini",
                    model=actual_model,
                    preferred_provider=preferred_provider,
                )

            except Exception as exc:
                print(
                    "[Gemini Fallback Error]",
                    type(exc).__name__,
                    str(exc),
                )

    # =====================================================
    # GEMINI SELECTED
    # =====================================================

    else:

        if (
            GEMINI_ENABLED
            and GEMINI_API_KEY
        ):
            try:
                actual_model = (
                    normalize_gemini_model(
                        model
                    )
                )

                answer = (
                    generate_gemini_answer(
                        question,
                        useful_chunks,
                        model=actual_model,
                    )
                )

                return build_generation_result(
                    answer=answer,
                    provider="gemini",
                    model=actual_model,
                    preferred_provider=preferred_provider,
                )

            except Exception as exc:
                print(
                    "[Gemini Error]",
                    type(exc).__name__,
                    str(exc),
                )

        # Gemini unavailable -> Ollama fallback.
        if OLLAMA_ENABLED:
            try:
                actual_model = OLLAMA_MODEL

                answer = (
                    generate_ollama_answer(
                        question,
                        useful_chunks,
                        model=actual_model,
                    )
                )

                return build_generation_result(
                    answer=answer,
                    provider="ollama",
                    model=actual_model,
                    preferred_provider=preferred_provider,
                )

            except Exception as exc:
                print(
                    "[Ollama Fallback Error]",
                    type(exc).__name__,
                    str(exc),
                )

    # =====================================================
    # FINAL NON-LLM FALLBACK
    # =====================================================

    print(
        "[Generator] Both LLM providers were unavailable. "
        "Using local extractive fallback."
    )

    answer = generate_local_fallback_answer(
        question,
        useful_chunks,
    )

    return build_generation_result(
        answer=answer,
        provider="fallback",
        model="extractive",
        preferred_provider=preferred_provider,
    )

