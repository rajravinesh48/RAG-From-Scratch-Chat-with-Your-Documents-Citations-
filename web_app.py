import inspect
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from generator import generate_answer
from ingest import ingest_documents
from intent_classifier import classify_question, clear_model_cache, get_model_status
from retriever import retrieve_relevant_chunks
from train_intent_model import train_model


app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "rag_demo_secret_key_change_this",
)

DOCUMENTS_DIR = Path("documents")
VECTOR_STORE_FILE = Path("vector_store/store.json")
USERS_FILE = Path("users.json")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
ALLOWED_TOP_K = {1, 2, 3, 5}
MAX_CHAT_ITEMS = 10
MAX_CHAT_SESSIONS = 8
MAX_SOURCE_TEXT_LENGTH = 800

# =========================================================
# LLM MODEL CONFIGURATION
# =========================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama",
).strip().lower()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:4b",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
).strip()


def build_model_options():
    """
    Models exposed in the UI.

    Prefixes tell generator.py which provider to use:
        ollama:<model>
        gemini:<model>
    """
    options = [
        {
            "value": f"ollama:{OLLAMA_MODEL}",
            "label": (
                "Ollama · Gemma 3 4B"
                if OLLAMA_MODEL == "gemma3:4b"
                else f"Ollama · {OLLAMA_MODEL}"
            ),
            "provider": "ollama",
        },
        {
            "value": f"gemini:{GEMINI_MODEL}",
            "label": (
                "Gemini · "
                + GEMINI_MODEL.replace("-", " ").title()
            ),
            "provider": "gemini",
        },
        {
            "value": "gemini:gemini-3.1-flash-lite",
            "label": "Gemini · Gemini 3.1 Flash Lite",
            "provider": "gemini",
        },
    ]

    unique_options = []
    seen_values = set()

    for option in options:
        if option["value"] in seen_values:
            continue

        seen_values.add(option["value"])
        unique_options.append(option)

    return unique_options


MODEL_OPTIONS = build_model_options()

ALLOWED_LLM_MODELS = {
    option["value"]
    for option in MODEL_OPTIONS
}


DEFAULT_LLM_MODEL = (
    f"gemini:{GEMINI_MODEL}"
    if LLM_PROVIDER == "gemini"
    else f"ollama:{OLLAMA_MODEL}"
)


def normalize_selected_model(model):
    """
    Converts older session values such as:
        gemini-3.5-flash

    into:
        gemini:gemini-3.5-flash
    """
    model = str(model or "").strip()

    if model in ALLOWED_LLM_MODELS:
        return model

    if model.startswith("gemini-"):
        prefixed = f"gemini:{model}"

        if prefixed in ALLOWED_LLM_MODELS:
            return prefixed

    if model == OLLAMA_MODEL:
        prefixed = f"ollama:{model}"

        if prefixed in ALLOWED_LLM_MODELS:
            return prefixed

    return DEFAULT_LLM_MODEL


def get_model_label(model):
    normalized_model = normalize_selected_model(
        model
    )

    for option in MODEL_OPTIONS:
        if option["value"] == normalized_model:
            return option["label"]

    return normalized_model


# =========================================================
# USER MANAGEMENT
# =========================================================

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=4)


def load_users():
    if not USERS_FILE.exists():
        save_users({
            "admin": {
                "password": generate_password_hash("admin123"),
                "role": "admin",
            },
            "user": {
                "password": generate_password_hash("user123"),
                "role": "user",
            },
        })

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read {USERS_FILE}: {exc}") from exc


def current_user():
    if "username" not in session:
        return None
    return {
        "username": session.get("username"),
        "role": session.get("role"),
    }


def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return function(*args, **kwargs)
    return wrapper


def admin_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))

        if session.get("role") != "admin":
            return redirect(
                url_for(
                    "home",
                    error="Only admin can access this function.",
                )
            )
        return function(*args, **kwargs)
    return wrapper


# =========================================================
# RAG HELPERS
# =========================================================

def load_vector_store():
    if not VECTOR_STORE_FILE.exists():
        return None, None, None

    try:
        with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None, None, None

    return (
        data.get("vocabulary"),
        data.get("idf_values"),
        data.get("chunks"),
    )


def is_allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_documents():
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    return sorted(
        path.name
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file() and is_allowed_file(path.name)
    )


def get_store_stats():
    vocabulary, _idf_values, chunks = load_vector_store()

    if vocabulary is None or chunks is None:
        return {
            "status": "Not created",
            "total_chunks": 0,
            "vocabulary_size": 0,
            "document_count": len(get_documents()),
        }

    return {
        "status": "Ready",
        "total_chunks": len(chunks),
        "vocabulary_size": len(vocabulary),
        "document_count": len(get_documents()),
    }


def get_confidence(score):
    if score >= 0.30:
        return "High"
    if score >= 0.10:
        return "Medium"
    if score > 0:
        return "Low"
    return "No Match"


def get_ml_top_k(intent, selected_top_k):
    recommended = {
        "definition": 1,
        "file_support": 2,
        "component": 2,
        "process": 3,
        "comparison": 3,
        "troubleshooting": 3,
        "general": selected_top_k,
    }.get(intent, selected_top_k)

    return min(selected_top_k, recommended)


def search_chunks(question, embedded_chunks, vocabulary, idf_values, top_k):
    try:
        return retrieve_relevant_chunks(
            question=question,
            embedded_chunks=embedded_chunks,
            vocabulary=vocabulary,
            idf_values=idf_values,
            top_k=top_k,
            min_score=0.01,
        )
    except TypeError:
        return retrieve_relevant_chunks(
            question=question,
            embedded_chunks=embedded_chunks,
            vocabulary=vocabulary,
            top_k=top_k,
        )


def safe_source_text(text):
    text = str(text or "")
    if len(text) <= MAX_SOURCE_TEXT_LENGTH:
        return text
    return text[:MAX_SOURCE_TEXT_LENGTH] + "..."


def get_user_list():
    users = load_users()
    return [
        {
            "username": username,
            "role": details["role"],
        }
        for username, details in users.items()
    ]


def generate_answer_for_model(
    question,
    relevant_chunks,
    selected_model,
):
    """
    Normalizes the generator result.

    New generator.py returns answer + actual provider/model.
    Older string-returning versions are also supported.
    """
    parameters = inspect.signature(
        generate_answer
    ).parameters

    if "model" in parameters:
        result = generate_answer(
            question,
            relevant_chunks,
            model=selected_model,
        )
    else:
        result = generate_answer(
            question,
            relevant_chunks,
        )

    if isinstance(result, dict):
        return {
            "answer": str(
                result.get(
                    "answer",
                    "",
                )
            ).strip(),
            "provider": str(
                result.get(
                    "provider",
                    "unknown",
                )
            ).strip(),
            "model": str(
                result.get(
                    "model",
                    "",
                )
            ).strip(),
            "model_label": str(
                result.get(
                    "model_label",
                    "",
                )
            ).strip(),
            "fallback_used": bool(
                result.get(
                    "fallback_used",
                    False,
                )
            ),
        }

    return {
        "answer": str(
            result or ""
        ).strip(),
        "provider": (
            "gemini"
            if selected_model.startswith(
                "gemini:"
            )
            else "ollama"
        ),
        "model": selected_model.split(
            ":",
            1,
        )[-1],
        "model_label": get_model_label(
            selected_model
        ),
        "fallback_used": False,
    }


# =========================================================
# CHAT SESSION HELPERS
# =========================================================

def create_chat_id():
    return uuid4().hex[:12]


def create_chat_record(title="New Chat"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_id = create_chat_id()

    return chat_id, {
        "id": chat_id,
        "title": title,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def initialize_chat_session():
    chat_sessions = session.get("chat_sessions")

    if not isinstance(chat_sessions, dict):
        chat_id, chat = create_chat_record()
        session["chat_sessions"] = {chat_id: chat}
        session["active_chat_id"] = chat_id

    else:
        active_chat_id = session.get("active_chat_id")

        if not active_chat_id or active_chat_id not in chat_sessions:
            if chat_sessions:
                session["active_chat_id"] = next(iter(chat_sessions))
            else:
                chat_id, chat = create_chat_record()
                session["chat_sessions"] = {chat_id: chat}
                session["active_chat_id"] = chat_id

    session.setdefault("selected_document", None)
    session.setdefault("current_sources", [])
    session["selected_model"] = normalize_selected_model(
        session.get(
            "selected_model",
            DEFAULT_LLM_MODEL,
        )
    )

    session.modified = True


def get_active_chat():
    initialize_chat_session()
    return session.get("chat_sessions", {}).get(
        session.get("active_chat_id")
    )


def get_chat_sessions_for_view():
    initialize_chat_session()
    chats = list(session.get("chat_sessions", {}).values())

    chats.sort(
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )

    return chats


def get_active_messages():
    active_chat = get_active_chat()
    return active_chat.get("messages", []) if active_chat else []


def make_chat_title(question):
    words = question.strip().split()

    if not words:
        return "New Chat"

    title = " ".join(words[:6])

    if len(words) > 6:
        title += "..."

    return title


def trim_chat_sessions(chat_sessions):
    if len(chat_sessions) <= MAX_CHAT_SESSIONS:
        return chat_sessions

    ordered = sorted(
        chat_sessions.items(),
        key=lambda item: item[1].get("updated_at", ""),
        reverse=True,
    )

    return dict(ordered[:MAX_CHAT_SESSIONS])


# =========================================================
# AUTH ROUTES
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        users = load_users()

        if (
            username in users
            and check_password_hash(users[username]["password"], password)
        ):
            session.clear()
            session["username"] = username
            session["role"] = users[username]["role"]
            initialize_chat_session()
            session.modified = True

            return redirect(url_for("home"))

        error = "Invalid username or password."

    return render_template(
        "login.html",
        error=error,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================================================
# CHAT / DOCUMENT / MODEL SELECTION ROUTES
# =========================================================

@app.route("/new-chat", methods=["POST"])
@login_required
def new_chat():
    initialize_chat_session()

    chat_sessions = session.get("chat_sessions", {})
    chat_id, chat = create_chat_record()

    chat_sessions[chat_id] = chat
    chat_sessions = trim_chat_sessions(chat_sessions)

    session["chat_sessions"] = chat_sessions
    session["active_chat_id"] = chat_id
    session["current_sources"] = []
    session.modified = True

    return redirect(url_for("home"))


@app.route("/select-chat/<chat_id>", methods=["POST"])
@login_required
def select_chat(chat_id):
    initialize_chat_session()

    chat_sessions = session.get("chat_sessions", {})

    if chat_id not in chat_sessions:
        return redirect(
            url_for(
                "home",
                error="Chat session not found.",
            )
        )

    session["active_chat_id"] = chat_id

    messages = chat_sessions[chat_id].get("messages", [])

    session["current_sources"] = (
        messages[-1].get("sources", [])
        if messages
        else []
    )

    session.modified = True
    return redirect(url_for("home"))


@app.route("/select-document/<filename>", methods=["POST"])
@login_required
def select_document(filename):
    filename = secure_filename(filename)

    if filename not in get_documents():
        return redirect(
            url_for(
                "home",
                error="Document not found.",
            )
        )

    session["selected_document"] = filename
    session["current_sources"] = []
    session.modified = True

    return redirect(url_for("home"))


@app.route("/select-all-documents", methods=["POST"])
@login_required
def select_all_documents():
    session["selected_document"] = None
    session["current_sources"] = []
    session.modified = True

    return redirect(url_for("home"))


@app.route("/set-model", methods=["POST"])
@login_required
def set_model():
    selected_model = request.form.get(
        "model",
        DEFAULT_LLM_MODEL,
    ).strip()

    if selected_model not in ALLOWED_LLM_MODELS:
        return redirect(
            url_for(
                "home",
                error="Invalid LLM model selected.",
            )
        )

    session["selected_model"] = selected_model
    session["current_sources"] = []
    session.modified = True

    return redirect(url_for("home"))


# =========================================================
# MAIN CHAT ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    initialize_chat_session()

    question = ""
    selected_top_k = 3

    message = request.args.get("message")
    error = request.args.get("error")

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        try:
            selected_top_k = int(request.form.get("top_k", 3))
        except (TypeError, ValueError):
            selected_top_k = 3

        if selected_top_k not in ALLOWED_TOP_K:
            selected_top_k = 3

        if not question:
            error = "Please enter a question."

        else:
            vocabulary, idf_values, embedded_chunks = load_vector_store()

            if vocabulary is None or embedded_chunks is None:
                error = (
                    "Vector store not found or invalid. "
                    "Please rebuild the vector store, or ask an admin to upload documents."
                )

            else:
                selected_document = session.get("selected_document")

                search_pool = embedded_chunks

                if selected_document:
                    search_pool = [
                        chunk
                        for chunk in embedded_chunks
                        if chunk.get("filename") == selected_document
                    ]

                if not search_pool:
                    error = (
                        "No vector-store chunks were found "
                        "for the selected document."
                    )

                else:
                    intent_result = classify_question(question)

                    predicted_intent = intent_result.get(
                        "intent",
                        "general",
                    )

                    intent_confidence = float(
                        intent_result.get(
                            "confidence",
                            0.0,
                        )
                    )

                    model_available = bool(
                        intent_result.get(
                            "model_available",
                            False,
                        )
                    )

                    intent_low_confidence = bool(
                        intent_result.get(
                            "low_confidence",
                            True,
                        )
                    )

                    effective_top_k = get_ml_top_k(
                        predicted_intent,
                        selected_top_k,
                    )

                    relevant_chunks = search_chunks(
                        question=question,
                        embedded_chunks=search_pool,
                        vocabulary=vocabulary,
                        idf_values=idf_values,
                        top_k=effective_top_k,
                    )

                    selected_model = normalize_selected_model(
                        session.get(
                            "selected_model",
                            DEFAULT_LLM_MODEL,
                        )
                    )

                    generation_result = (
                        generate_answer_for_model(
                            question,
                            relevant_chunks,
                            selected_model,
                        )
                    )

                    answer = generation_result.get(
                        "answer",
                        "",
                    )

                    generation_provider = (
                        generation_result.get(
                            "provider",
                            "unknown",
                        )
                    )

                    generation_model = (
                        generation_result.get(
                            "model",
                            "",
                        )
                    )

                    generation_model_label = (
                        generation_result.get(
                            "model_label",
                            "",
                        )
                    )

                    generation_fallback_used = bool(
                        generation_result.get(
                            "fallback_used",
                            False,
                        )
                    )

                    no_context_found = (
                        "I could not find enough relevant information "
                        "in the uploaded documents."
                    ).lower() in answer.lower()

                    if no_context_found:
                        rag_confidence = "No Match"

                    elif relevant_chunks:
                        top_score = float(
                            relevant_chunks[0].get(
                                "score",
                                0,
                            )
                        )
                        rag_confidence = get_confidence(top_score)

                    else:
                        rag_confidence = "No Match"

                    sources = []

                    if not no_context_found:
                        for source_index, chunk in enumerate(
                            relevant_chunks,
                            start=1,
                        ):
                            chunk_score = float(
                                chunk.get(
                                    "score",
                                    0,
                                )
                            )

                            if chunk_score <= 0:
                                continue

                            matched_keywords = chunk.get(
                                "matched_keywords",
                                [],
                            )

                            if isinstance(matched_keywords, list):
                                matched_keywords_text = ", ".join(
                                    matched_keywords
                                )
                            else:
                                matched_keywords_text = str(
                                    matched_keywords
                                )

                            sources.append({
                                "index": source_index,
                                "filename": chunk.get(
                                    "filename",
                                    "Unknown",
                                ),
                                "page_number": chunk.get(
                                    "page_number"
                                ),
                                "chunk_id": chunk.get(
                                    "chunk_id",
                                    "-",
                                ),
                                "page_chunk_id": chunk.get(
                                    "page_chunk_id"
                                ),
                                "score": round(
                                    chunk_score,
                                    4,
                                ),
                                "matched_keywords": matched_keywords_text,
                                "text": safe_source_text(
                                    chunk.get(
                                        "text",
                                        "",
                                    )
                                ),
                            })

                    chat_sessions = session.get("chat_sessions", {})
                    active_chat_id = session.get("active_chat_id")
                    active_chat = chat_sessions.get(active_chat_id)

                    if active_chat is None:
                        active_chat_id, active_chat = create_chat_record()

                    message_data = {
                        "question": question,
                        "answer": answer,
                        "confidence": rag_confidence,
                        "sources": sources,
                        "time": datetime.now().strftime(
                            "%d-%m-%Y %I:%M:%S %p"
                        ),
                        "intent": predicted_intent.replace(
                            "_",
                            " ",
                        ).title(),
                        "intent_confidence": round(
                            intent_confidence * 100,
                            2,
                        ),
                        "intent_low_confidence": intent_low_confidence,
                        "top_k_used": (
                            0
                            if no_context_found
                            else len(sources)
                        ),
                        "ml_model_available": model_available,
                        "selected_document": selected_document,

                        # Selected in the dropdown.
                        "selected_model": selected_model,
                        "selected_model_label": get_model_label(
                            selected_model
                        ),

                        # Actually used to generate this answer.
                        "generation_provider": generation_provider,
                        "generation_model": generation_model,
                        "generation_model_label": generation_model_label,
                        "generation_fallback_used": generation_fallback_used,
                    }

                    active_chat.setdefault("messages", [])
                    active_chat["messages"].append(message_data)
                    active_chat["messages"] = active_chat["messages"][
                        -MAX_CHAT_ITEMS:
                    ]

                    if active_chat.get("title") == "New Chat":
                        active_chat["title"] = make_chat_title(question)

                    active_chat["updated_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    chat_sessions[active_chat_id] = active_chat

                    session["chat_sessions"] = trim_chat_sessions(
                        chat_sessions
                    )
                    session["active_chat_id"] = active_chat_id
                    session["current_sources"] = sources
                    session.modified = True

                    question = ""

    active_chat = get_active_chat()

    active_messages = (
        active_chat.get("messages", [])
        if active_chat
        else []
    )

    return render_template(
        "dashboard.html",

        # New 3-panel UI variables
        documents=get_documents(),
        current_document=session.get("selected_document"),
        chat_sessions=get_chat_sessions_for_view(),
        active_chat_id=session.get("active_chat_id"),
        active_chat_title=(
            active_chat.get("title", "New Chat")
            if active_chat
            else "New Chat"
        ),
        messages=active_messages,
        current_sources=session.get("current_sources", []),
        selected_model=normalize_selected_model(
            session.get(
                "selected_model",
                DEFAULT_LLM_MODEL,
            )
        ),
        model_options=MODEL_OPTIONS,

        # Compatibility for older templates.
        allowed_models=[
            option["value"]
            for option in MODEL_OPTIONS
        ],

        # Existing template variables kept for compatibility
        question=question,
        selected_top_k=selected_top_k,
        chat_history=list(reversed(active_messages)),
        stats=get_store_stats(),
        ml_status=get_model_status(),
        users=get_user_list(),
        current_user=current_user(),
        message=message,
        error=error,
    )


# =========================================================
# DOCUMENT ROUTES
# =========================================================

@app.route("/upload", methods=["POST"])
@admin_required
def upload_document():
    if "document" not in request.files:
        return redirect(
            url_for(
                "home",
                error="No file selected.",
            )
        )

    file = request.files["document"]

    if not file.filename:
        return redirect(
            url_for(
                "home",
                error="No file selected.",
            )
        )

    if not is_allowed_file(file.filename):
        return redirect(
            url_for(
                "home",
                error="Only PDF, TXT and DOCX files are supported.",
            )
        )

    DOCUMENTS_DIR.mkdir(exist_ok=True)

    filename = secure_filename(file.filename)

    if not filename:
        return redirect(
            url_for(
                "home",
                error="Invalid filename.",
            )
        )

    file.save(DOCUMENTS_DIR / filename)

    try:
        ingest_documents()
    except Exception as exc:
        return redirect(
            url_for(
                "home",
                error=(
                    "File uploaded but vector store rebuild failed: "
                    f"{exc}"
                ),
            )
        )

    session["selected_document"] = filename
    session["current_sources"] = []
    session.modified = True

    return redirect(
        url_for(
            "home",
            message=(
                "Document uploaded and vector store rebuilt successfully."
            ),
        )
    )


@app.route("/delete/<filename>", methods=["POST"])
@admin_required
def delete_document(filename):
    safe_filename = secure_filename(filename)
    file_path = DOCUMENTS_DIR / safe_filename

    if not safe_filename or not file_path.exists():
        return redirect(
            url_for(
                "home",
                error="Document not found.",
            )
        )

    file_path.unlink()

    try:
        ingest_documents()
    except Exception as exc:
        return redirect(
            url_for(
                "home",
                error=(
                    "Document deleted but vector store rebuild failed: "
                    f"{exc}"
                ),
            )
        )

    if session.get("selected_document") == safe_filename:
        session["selected_document"] = None

    session["current_sources"] = []
    session.modified = True

    return redirect(
        url_for(
            "home",
            message=(
                "Document deleted and vector store rebuilt successfully."
            ),
        )
    )


@app.route("/rebuild", methods=["POST"])
@login_required
def rebuild_store():
    try:
        ingest_documents()
    except Exception as exc:
        return redirect(
            url_for(
                "home",
                error=f"Vector store rebuild failed: {exc}",
            )
        )

    session["current_sources"] = []
    session.modified = True

    return redirect(
        url_for(
            "home",
            message="Vector store rebuilt successfully.",
        )
    )


# =========================================================
# MACHINE LEARNING ROUTE
# =========================================================

@app.route("/train-ml-model", methods=["POST"])
@admin_required
def train_ml_model():
    try:
        result = train_model()
        clear_model_cache()

        validation_accuracy = float(
            result.get(
                "accuracy",
                0,
            )
        )

        training_samples = int(
            result.get(
                "samples",
                0,
            )
        )

        message = (
            "Machine learning model trained successfully. "
            f"Validation accuracy: {validation_accuracy:.2%}. "
            f"Training samples: {training_samples}."
        )

        cross_validation_accuracy = result.get(
            "cross_validation_accuracy"
        )

        if cross_validation_accuracy is not None:
            message += (
                " Cross-validation accuracy: "
                f"{float(cross_validation_accuracy):.2%}."
            )

        return redirect(
            url_for(
                "home",
                message=message,
            )
        )

    except Exception as exc:
        return redirect(
            url_for(
                "home",
                error=f"ML model training failed: {exc}",
            )
        )


# =========================================================
# USER ADMIN ROUTES
# =========================================================

@app.route("/create-user", methods=["POST"])
@admin_required
def create_user():
    username = secure_filename(
        request.form.get(
            "new_username",
            "",
        ).strip()
    )

    password = request.form.get(
        "new_password",
        "",
    ).strip()

    role = request.form.get(
        "new_role",
        "user",
    ).strip()

    if not username or not password:
        return redirect(
            url_for(
                "home",
                error="Username and password are required.",
            )
        )

    if len(password) < 6:
        return redirect(
            url_for(
                "home",
                error=(
                    "Password must contain at least 6 characters."
                ),
            )
        )

    if role not in {"admin", "user"}:
        role = "user"

    users = load_users()

    if username in users:
        return redirect(
            url_for(
                "home",
                error="User already exists.",
            )
        )

    users[username] = {
        "password": generate_password_hash(password),
        "role": role,
    }

    save_users(users)

    return redirect(
        url_for(
            "home",
            message="User created successfully.",
        )
    )


@app.route("/delete-user/<username>", methods=["POST"])
@admin_required
def delete_user(username):
    username = secure_filename(username)

    if username == session.get("username"):
        return redirect(
            url_for(
                "home",
                error=(
                    "You cannot delete your own account while logged in."
                ),
            )
        )

    users = load_users()

    if username not in users:
        return redirect(
            url_for(
                "home",
                error="User not found.",
            )
        )

    del users[username]
    save_users(users)

    return redirect(
        url_for(
            "home",
            message="User deleted successfully.",
        )
    )


# =========================================================
# CHAT HISTORY ROUTES
# =========================================================

@app.route("/clear", methods=["POST"])
@login_required
def clear_chat():
    initialize_chat_session()

    chat_sessions = session.get("chat_sessions", {})
    active_chat_id = session.get("active_chat_id")

    if active_chat_id in chat_sessions:
        chat_sessions[active_chat_id]["messages"] = []
        chat_sessions[active_chat_id]["title"] = "New Chat"
        chat_sessions[active_chat_id]["updated_at"] = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    session["chat_sessions"] = chat_sessions
    session["current_sources"] = []
    session.modified = True

    return redirect(
        url_for(
            "home",
            message="Chat cleared.",
        )
    )


@app.route("/download-chat", methods=["POST"])
@login_required
def download_chat():
    active_chat = get_active_chat()

    messages = (
        active_chat.get("messages", [])
        if active_chat
        else []
    )

    title = (
        active_chat.get("title", "RAG Chat")
        if active_chat
        else "RAG Chat"
    )

    content = f"RAG Chat History - {title}\n"
    content += f"User: {session.get('username')}\n"
    content += f"Role: {session.get('role')}\n"

    selected_document = session.get("selected_document")

    content += (
        "Document Scope: "
        + (
            selected_document
            if selected_document
            else "All Documents"
        )
        + "\n"
    )

    selected_model = normalize_selected_model(
        session.get(
            "selected_model",
            DEFAULT_LLM_MODEL,
        )
    )

    content += (
        "LLM Model: "
        f"{get_model_label(selected_model)}\n"
    )

    content += "=" * 60 + "\n\n"

    if not messages:
        content += "No chat history available.\n"

    else:
        for index, chat in enumerate(messages, start=1):
            content += f"Question {index}\n"
            content += (
                "Time: "
                f"{chat.get('time', 'Not available')}\n"
            )
            content += (
                "RAG Confidence: "
                f"{chat.get('confidence', 'Unknown')}\n"
            )
            content += (
                "ML Intent: "
                f"{chat.get('intent', 'General')}\n"
            )
            content += (
                "ML Confidence: "
                f"{chat.get('intent_confidence', 0)}%\n"
            )

            if chat.get("intent_low_confidence", False):
                content += (
                    "ML Warning: Low-confidence prediction\n"
                )

            content += (
                "Chunks Used: "
                f"{chat.get('top_k_used', 0)}\n"
            )

            content += (
                "Generated By: "
                f"{chat.get(
                    'generation_model_label',
                    chat.get(
                        'selected_model_label',
                        'Unknown',
                    ),
                )}\n"
            )

            if chat.get(
                "generation_fallback_used",
                False,
            ):
                content += (
                    "Generation Fallback: Yes\n"
                )

            content += "\n"
            content += (
                f"User:\n{chat.get('question', '')}\n\n"
            )
            content += (
                f"Answer:\n{chat.get('answer', '')}\n"
            )

            sources = chat.get("sources", [])

            if sources:
                content += "\nSources:\n"

                for source in sources:
                    page_number = source.get(
                        "page_number"
                    )

                    page_text = (
                        f"Page {page_number} | "
                        if page_number is not None
                        else ""
                    )

                    content += (
                        f"[{source.get('index', '-')}] "
                        f"{source.get('filename', 'Unknown')} | "
                        f"{page_text}"
                        f"Chunk {source.get('chunk_id', '-')} | "
                        f"Score {source.get('score', 0)}\n"
                    )

            content += "\n" + "-" * 60 + "\n\n"

    response = make_response(content)

    response.headers["Content-Disposition"] = (
        "attachment; filename=rag_chat_history.txt"
    )

    response.headers["Content-Type"] = (
        "text/plain; charset=utf-8"
    )

    return response


if __name__ == "__main__":
    load_users()
    app.run(debug=True)
