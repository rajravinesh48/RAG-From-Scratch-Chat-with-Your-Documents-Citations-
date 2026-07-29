from intent_classifier import classify_question, get_model_status


TEST_QUESTIONS = [
    "What is chunking?",
    "How are relevant chunks retrieved?",
    "Which file types are supported?",
    "What does ingest.py do?",
    "Compare chunks and embeddings",
    "Why is similarity score zero?",
    "Tell me about this project",
]


def main():
    status = get_model_status()

    print("Model Status:")
    print(status)
    print("-" * 60)

    if not status["available"]:
        print(
            "Model is not trained. Run: "
            "python train_intent_model.py"
        )
        return

    for question in TEST_QUESTIONS:
        result = classify_question(question)

        print(f"Question: {question}")
        print(f"Intent: {result['intent']}")
        print(
            "Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
