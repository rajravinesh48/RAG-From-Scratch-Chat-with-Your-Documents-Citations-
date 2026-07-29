import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from generator import generate_answer

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
from intent_classifier import (
    classify_question,
    clear_model_cache,
    get_model_status,
)
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
MAX_SOURCE_TEXT_LENGTH = 1200


# ---------------- USER MANAGEMENT ----------------

def save_users(users):
    """
    Saves user records in users.json.
    """
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=4)


def load_users():
    """
    Loads users from users.json.

    Creates one demo admin and one demo user when the file
    does not exist.
    """
    if not USERS_FILE.exists():
        default_users = {
            "admin": {
                "password": generate_password_hash("admin123"),
                "role": "admin",
            },
            "user": {
                "password": generate_password_hash("user123"),
                "role": "user",
            },
        }

        save_users(default_users)

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read {USERS_FILE}: {exc}"
        ) from exc


def current_user():
    """
    Returns currently logged-in user information.
    """
    if "username" not in session:
        return None

    return {
        "username": session.get("username"),
        "role": session.get("role"),
    }


def login_required(function):
    """
    Allows route access only to logged-in users.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


def admin_required(function):
    """
    Allows route access only to admin users.
    """
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


# ---------------- RAG HELPERS ----------------

def load_vector_store():
    """
    Loads vocabulary, IDF values and embedded chunks from JSON.
    """
    if not VECTOR_STORE_FILE.exists():
        return None, None, None

    try:
        with open(
            VECTOR_STORE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (OSError, json.JSONDecodeError):
        return None, None, None

    vocabulary = data.get("vocabulary")
    idf_values = data.get("idf_values")
    chunks = data.get("chunks")

    return vocabulary, idf_values, chunks


def is_allowed_file(filename):
    """
    Checks whether an uploaded filename has a supported extension.
    """
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_documents():
    """
    Returns supported documents from the documents folder.
    """
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    files = [
        file_path.name
        for file_path in DOCUMENTS_DIR.iterdir()
        if file_path.is_file()
        and is_allowed_file(file_path.name)
    ]

    return sorted(files)


def get_store_stats():
    """
    Returns vector-store statistics for the dashboard.
    """
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
    """
    Converts a cosine-similarity score into a readable label.
    """
    if score >= 0.30:
        return "High"

    if score >= 0.10:
        return "Medium"

    if score > 0:
        return "Low"

    return "No Match"


def get_ml_top_k(intent, selected_top_k):
    """
    Uses the predicted ML intent to choose an appropriate number
    of chunks while respecting the user's selected maximum.
    """
    recommended_top_k = {
        "definition": 1,
        "file_support": 2,
        "component": 2,
        "process": 3,
        "comparison": 3,
        "troubleshooting": 3,
        "general": selected_top_k,
    }.get(intent, selected_top_k)

    return min(selected_top_k, recommended_top_k)


def search_chunks(
    question,
    embedded_chunks,
    vocabulary,
    idf_values,
    top_k,
):
    """
    Supports both the old retriever and the TF-IDF retriever.
    """
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


def get_user_list():
    """
    Returns user records without password hashes.
    """
    users = load_users()

    return [
        {
            "username": username,
            "role": details["role"],
        }
        for username, details in users.items()
    ]


def get_display_chat_history():
    """
    Returns newest chats first without modifying session order.
    """
    return list(
        reversed(
            session.get("chat_history", [])
        )
    )


def safe_source_text(text):
    """
    Limits source text stored in Flask's client-side session.
    """
    text = str(text or "")

    if len(text) <= MAX_SOURCE_TEXT_LENGTH:
        return text

    return text[:MAX_SOURCE_TEXT_LENGTH] + "..."


# ---------------- AUTH ROUTES ----------------

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Displays login form and authenticates a user.
    """
    if "username" in session:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = request.form.get(
            "username",
            "",
        ).strip()

        password = request.form.get(
            "password",
            "",
        ).strip()

        users = load_users()

        if (
            username in users
            and check_password_hash(
                users[username]["password"],
                password,
            )
        ):
            session["username"] = username
            session["role"] = users[username]["role"]
            session["chat_history"] = []
            session.modified = True

            return redirect(url_for("home"))

        error = "Invalid username or password."

    return render_template(
        "login.html",
        error=error,
    )


@app.route("/logout")
def logout():
    """
    Logs out the current user.
    """
    session.clear()
    return redirect(url_for("login"))


# ---------------- MAIN CHAT ROUTE ----------------

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    """
    Main ML + RAG chat dashboard.
    """
    question = ""
    selected_top_k = 3

    message = request.args.get("message")
    error = request.args.get("error")

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        question = request.form.get(
            "question",
            "",
        ).strip()

        try:
            selected_top_k = int(
                request.form.get(
                    "top_k",
                    3,
                )
            )
        except (TypeError, ValueError):
            selected_top_k = 3

        if selected_top_k not in ALLOWED_TOP_K:
            selected_top_k = 3

        if not question:
            error = "Please enter a question."

        else:
            vocabulary, idf_values, embedded_chunks = (
                load_vector_store()
            )

            if vocabulary is None or embedded_chunks is None:
                error = (
                    "Vector store not found or invalid. "
                    "Admin should upload documents or "
                    "rebuild the vector store."
                )

            else:
                # -------- MACHINE LEARNING INTENT --------

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
                    intent=predicted_intent,
                    selected_top_k=selected_top_k,
                )

                # -------- RAG RETRIEVAL --------

                relevant_chunks = search_chunks(
                    question=question,
                    embedded_chunks=embedded_chunks,
                    vocabulary=vocabulary,
                    idf_values=idf_values,
                    top_k=effective_top_k,
                )

                answer = generate_answer(
                    question,
                    relevant_chunks,
                )

                # Check whether Gemini/local generator found no valid answer.
                no_context_found = (
                    "I could not find enough relevant information "
                    "in the uploaded documents."
                ).lower() in answer.lower()


                # ---------------- RAG CONFIDENCE ----------------

                if no_context_found:
                    top_score = 0.0
                    rag_confidence = "No Match"

                elif relevant_chunks:
                    top_score = float(
                        relevant_chunks[0].get(
                            "score",
                            0,
                        )
                    )

                    rag_confidence = get_confidence(
                        top_score
                    )

                else:
                    top_score = 0.0
                    rag_confidence = "No Match"


                # ---------------- SOURCE CHUNKS ----------------

                sources = []

                # Do not show unrelated citations when no answer was found.
                if not no_context_found:

                    for chunk in relevant_chunks:
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

                        sources.append(
                            {
                                "filename": chunk.get(
                                    "filename",
                                    "Unknown",
                                ),
                                "chunk_id": chunk.get(
                                    "chunk_id",
                                    "-",
                                ),
                                "score": round(
                                    chunk_score,
                                    4,
                                ),
                                "matched_keywords": (
                                    matched_keywords_text
                                ),
                                "text": safe_source_text(
                                    chunk.get(
                                        "text",
                                        "",
                                    )
                                ),
                            }
                        )
                
                # -------- CHAT HISTORY --------

                chat_history = session.get(
                    "chat_history",
                    [],
                )

                chat_history.append(
                    {
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
                        "intent_low_confidence": (
                            intent_low_confidence
                        ),
                        "top_k_used": (
                            0
                            if no_context_found
                            else len(sources)
                        ),
                        "ml_model_available": model_available,
                    }
                )

                session["chat_history"] = chat_history[
                    -MAX_CHAT_ITEMS:
                ]

                session.modified = True

                # Clear text area after successful processing.
                question = ""

    return render_template(
        "dashboard.html",
        question=question,
        selected_top_k=selected_top_k,
        chat_history=get_display_chat_history(),
        documents=get_documents(),
        stats=get_store_stats(),
        ml_status=get_model_status(),
        users=get_user_list(),
        current_user=current_user(),
        message=message,
        error=error,
    )


# ---------------- DOCUMENT ROUTES ----------------

@app.route("/upload", methods=["POST"])
@admin_required
def upload_document():
    """
    Uploads a supported document and rebuilds the vector store.
    """
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
                error=(
                    "Only PDF, TXT and DOCX files "
                    "are supported."
                ),
            )
        )

    DOCUMENTS_DIR.mkdir(exist_ok=True)

    filename = secure_filename(
        file.filename
    )

    if not filename:
        return redirect(
            url_for(
                "home",
                error="Invalid filename.",
            )
        )

    save_path = DOCUMENTS_DIR / filename
    file.save(save_path)

    try:
        ingest_documents()

    except Exception as exc:
        return redirect(
            url_for(
                "home",
                error=(
                    "File uploaded but vector store "
                    f"rebuild failed: {exc}"
                ),
            )
        )

    return redirect(
        url_for(
            "home",
            message=(
                "Document uploaded and vector store "
                "rebuilt successfully."
            ),
        )
    )


@app.route("/delete/<filename>", methods=["POST"])
@admin_required
def delete_document(filename):
    """
    Deletes a document and rebuilds the vector store.
    """
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
                    "Document deleted but vector store "
                    f"rebuild failed: {exc}"
                ),
            )
        )

    return redirect(
        url_for(
            "home",
            message=(
                "Document deleted and vector store "
                "rebuilt successfully."
            ),
        )
    )


@app.route("/rebuild", methods=["POST"])
@admin_required
def rebuild_store():
    """
    Manually rebuilds the vector store.
    """
    try:
        ingest_documents()

    except Exception as exc:
        return redirect(
            url_for(
                "home",
                error=(
                    "Vector store rebuild failed: "
                    f"{exc}"
                ),
            )
        )

    return redirect(
        url_for(
            "home",
            message="Vector store rebuilt successfully.",
        )
    )


# ---------------- MACHINE LEARNING ROUTE ----------------

@app.route("/train-ml-model", methods=["POST"])
@admin_required
def train_ml_model():
    """
    Trains or retrains the ML intent-classification model.
    """
    try:
        result = train_model()

        # Reload newly trained model on next prediction.
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


# ---------------- USER ADMIN ROUTES ----------------

@app.route("/create-user", methods=["POST"])
@admin_required
def create_user():
    """
    Creates a new admin or standard user.
    """
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
                    "Password must contain at least "
                    "6 characters."
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
    """
    Deletes a user except the currently logged-in account.
    """
    username = secure_filename(username)

    if username == session.get("username"):
        return redirect(
            url_for(
                "home",
                error=(
                    "You cannot delete your own account "
                    "while logged in."
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


# ---------------- CHAT HISTORY ROUTES ----------------

@app.route("/clear", methods=["POST"])
@login_required
def clear_chat():
    """
    Clears current user's session chat history.
    """
    session["chat_history"] = []
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
    """
    Downloads current user's chat history as a TXT file.
    """
    chat_history = session.get(
        "chat_history",
        [],
    )

    content = "RAG Chat History\n"
    content += f"User: {session.get('username')}\n"
    content += f"Role: {session.get('role')}\n"
    content += "=" * 60 + "\n\n"

    if not chat_history:
        content += "No chat history available.\n"

    else:
        for index, chat in enumerate(
            reversed(chat_history),
            start=1,
        ):
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

            if chat.get(
                "intent_low_confidence",
                False,
            ):
                content += (
                    "ML Warning: "
                    "Low-confidence prediction\n"
                )

            content += (
                "Chunks Used: "
                f"{chat.get('top_k_used', 3)}\n\n"
            )

            content += (
                f"User: {chat.get('question', '')}\n\n"
            )

            content += "Answer:\n"
            content += chat.get(
                "answer",
                "",
            ) + "\n"

            sources = chat.get(
                "sources",
                [],
            )

            if sources:
                content += "\nSources:\n"

                for source in sources:
                    content += (
                        f"- {source.get('filename', 'Unknown')} | "
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