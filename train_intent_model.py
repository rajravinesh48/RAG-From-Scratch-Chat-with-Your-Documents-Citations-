import json
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline


MODEL_DIR = Path("ml_model")
MODEL_FILE = MODEL_DIR / "intent_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"


# 10 examples per class = 70 balanced samples
TRAINING_EXAMPLES = [
    # ---------------- DEFINITION ----------------
    ("What does RAG stand for?", "definition"),
    ("Explain RAG", "definition"),
    ("Define RAG in simple words", "definition"),
    ("What is retrieval augmented generation?", "definition"),
    ("What does retrieval augmented generation mean?", "definition"),
    ("Give the full form of RAG", "definition"),
    ("What is RAG?", "definition"),
    ("Define retrieval augmented generation", "definition"),
    ("What is chunking?", "definition"),
    ("What is an embedding vector?", "definition"),
    ("Explain cosine similarity", "definition"),
    ("What is a vector store?", "definition"),
    ("What does document ingestion mean?", "definition"),
    ("Define a document chunk", "definition"),
    ("What is TF-IDF?", "definition"),
    ("What is similarity search?", "definition"),

    # ---------------- PROCESS ----------------
    ("How does RAG work?", "process"),
    ("How are documents processed?", "process"),
    ("How are chunks retrieved?", "process"),
    ("How does document ingestion work?", "process"),
    ("How is an answer generated?", "process"),
    ("Explain the complete RAG pipeline", "process"),
    ("How are embeddings created?", "process"),
    ("How does similarity search work?", "process"),
    ("How is the vector store created?", "process"),
    ("How are citations generated?", "process"),
    ("Describe how the RAG workflow operates", "process"),
    ("What happens after a document is uploaded?", "process"),
    ("How is a question converted into a vector?", "process"),
    ("How does the application select source chunks?", "process"),
    ("How does the system create citations?", "process"),
    ("What steps occur before generating an answer?", "process"),
    ("How are relevant chunks retrieved?", "process"),

    # ---------------- FILE SUPPORT ----------------
    ("Which file types are supported?", "file_support"),
    ("Can I upload a PDF?", "file_support"),
    ("Does this project support DOCX?", "file_support"),
    ("Can TXT files be processed?", "file_support"),
    ("What document formats can be uploaded?", "file_support"),
    ("Are PDF TXT and DOCX supported?", "file_support"),
    ("Which extensions are allowed?", "file_support"),
    ("Can this system read Word documents?", "file_support"),
    ("Does the chatbot accept text files?", "file_support"),
    ("What files can I add to the documents folder?", "file_support"),
    ("Can the application read a DOCX document?", "file_support"),
    ("Is PDF upload available?", "file_support"),
    ("Does the system support plain text documents?", "file_support"),
    ("Which document extensions can be processed?", "file_support"),
    ("Can I add Microsoft Word files?", "file_support"),
    ("What type of documents can the chatbot read?", "file_support"),

    # ---------------- COMPONENT ----------------
    ("Which file creates embeddings?", "component"),
    ("What does ingest.py do?", "component"),
    ("What is the purpose of retriever.py?", "component"),
    ("Which file contains the chatbot?", "component"),
    ("What does generator.py do?", "component"),
    ("Which library extracts PDF text?", "component"),
    ("What is chunker.py used for?", "component"),
    ("What does web_app.py contain?", "component"),
    ("Which file trains the intent model?", "component"),
    ("What does intent_classifier.py do?", "component"),
    ("Which file extracts text from documents?", "component"),
    ("Which module calculates cosine similarity?", "component"),
    ("Where is the vector store generated?", "component"),
    ("Which file manages user login?", "component"),
    ("Which module predicts question intent?", "component"),
    ("Which file contains the Flask routes?", "component"),

    # ---------------- COMPARISON ----------------
    (
        "What is the difference between TF-IDF and word count?",
        "comparison",
    ),
    ("Compare PDF and DOCX processing", "comparison"),
    (
        "What is the difference between chunks and embeddings?",
        "comparison",
    ),
    (
        "Compare cosine similarity and Euclidean distance",
        "comparison",
    ),
    (
        "How is a vector store different from a database?",
        "comparison",
    ),
    (
        "What is the difference between retrieval and generation?",
        "comparison",
    ),
    ("Compare admin and user access", "comparison"),
    (
        "What is the difference between RAG and normal chat?",
        "comparison",
    ),
    (
        "Compare machine learning classification and RAG retrieval",
        "comparison",
    ),
    (
        "What is the difference between training and prediction?",
        "comparison",
    ),
    ("Compare TF-IDF embeddings and semantic embeddings", "comparison"),
    ("How are PDF and TXT extraction different?", "comparison"),
    ("Compare cosine similarity with dot product", "comparison"),
    ("What differs between user and admin permissions?", "comparison"),
    ("Compare document ingestion and retrieval", "comparison"),
    ("What is different between an intent model and a retriever?", "comparison"),

    # ---------------- TROUBLESHOOTING ----------------
    ("Why is similarity score zero?", "troubleshooting"),
    (
        "Why does cosine similarity return zero?",
        "troubleshooting",
    ),
    ("Why are no relevant chunks found?", "troubleshooting"),
    (
        "Why is retrieval confidence low?",
        "troubleshooting",
    ),
    (
        "Why is the answer unrelated to my question?",
        "troubleshooting",
    ),
    (
        "How can I improve document matching?",
        "troubleshooting",
    ),
    (
        "Why is the vector store missing?",
        "troubleshooting",
    ),
    (
        "Why is document upload failing?",
        "troubleshooting",
    ),
    (
        "How do I fix an unsupported file error?",
        "troubleshooting",
    ),
    (
        "Why are citations not displayed?",
        "troubleshooting",
    ),
    ("Why is the ML prediction incorrect?", "troubleshooting"),
    ("Why is the intent confidence low?", "troubleshooting"),
    ("Why does the application return no citation?", "troubleshooting"),
    ("Why is the uploaded document not listed?", "troubleshooting"),
    ("Why is the answer taken from the wrong document?", "troubleshooting"),
    ("How can I improve the retrieved source chunks?", "troubleshooting"),

    # ---------------- GENERAL ----------------
    ("Tell me about this project", "general"),
    ("Give me an overview", "general"),
    ("What can this application do?", "general"),
    ("Show project features", "general"),
    ("Describe the system", "general"),
    ("Tell me more about this application", "general"),
    ("What information is available?", "general"),
    ("Explain this application", "general"),
    ("What are the main project features?", "general"),
    ("Give me a summary of the project", "general"),
    ("Provide details about the project", "general"),
    ("What are the capabilities of this system?", "general"),
    ("Give an introduction to this application", "general"),
    ("What functions does the chatbot provide?", "general"),
    ("Summarize the application features", "general"),
    ("Describe the purpose of this project", "general"),
]


def build_pipeline():
    """
    Creates the machine-learning pipeline.

    TF-IDF converts questions into vectors.
    Logistic Regression predicts the intent class.
    """

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def validate_training_data():
    """
    Checks for duplicate questions and prints class counts.
    """

    questions = [question.lower().strip() for question, _ in TRAINING_EXAMPLES]

    if len(questions) != len(set(questions)):
        raise ValueError(
            "Duplicate training questions found. "
            "Each training question must be unique."
        )

    class_counts = {}

    for _question, label in TRAINING_EXAMPLES:
        class_counts[label] = class_counts.get(label, 0) + 1

    print("Training samples per class:")

    for label, count in sorted(class_counts.items()):
        print(f"- {label}: {count}")

    return class_counts


def train_model():
    validate_training_data()

    texts = [
        question
        for question, _label in TRAINING_EXAMPLES
    ]

    labels = [
        label
        for _question, label in TRAINING_EXAMPLES
    ]

    # Holdout validation
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    evaluation_model = build_pipeline()
    evaluation_model.fit(x_train, y_train)

    predictions = evaluation_model.predict(x_test)

    holdout_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    # Cross-validation gives a more reliable score for a small dataset.
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_model = build_pipeline()

    cv_scores = cross_val_score(
        cv_model,
        texts,
        labels,
        cv=cross_validation,
        scoring="accuracy",
    )

    cv_mean_accuracy = float(cv_scores.mean())
    cv_std_accuracy = float(cv_scores.std())

    # Train the final model on all labelled questions.
    final_model = build_pipeline()
    final_model.fit(texts, labels)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_FILE,
    )

    metrics = {
        "accuracy": float(holdout_accuracy),
        "holdout_accuracy": float(holdout_accuracy),
        "cross_validation_accuracy": cv_mean_accuracy,
        "cross_validation_std": cv_std_accuracy,
        "cross_validation_scores": [
            float(score)
            for score in cv_scores
        ],
        "samples": len(texts),
        "trained_at": datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        ),
        "classes": sorted(set(labels)),
        "classification_report": report,
        "model_file": str(MODEL_FILE),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nML intent model trained successfully.")
    print(f"Training samples: {metrics['samples']}")
    print(
        "Holdout validation accuracy: "
        f"{holdout_accuracy:.2%}"
    )
    print(
        "5-fold cross-validation accuracy: "
        f"{cv_mean_accuracy:.2%} "
        f"(+/- {cv_std_accuracy:.2%})"
    )
    print(f"Model saved at: {MODEL_FILE}")
    print(f"Metrics saved at: {METRICS_FILE}")

    return metrics


if __name__ == "__main__":
    train_model()