import json
from pathlib import Path

from retriever import retrieve_relevant_chunks
from generator import generate_answer


VECTOR_STORE_FILE = Path("vector_store/store.json")


def load_vector_store():
    if not VECTOR_STORE_FILE.exists():
        raise FileNotFoundError(
            "Vector store not found. Please run python ingest.py first."
        )

    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    vocabulary = data["vocabulary"]
    idf_values = data.get("idf_values")
    chunks = data["chunks"]

    return vocabulary, idf_values, chunks

    if not VECTOR_STORE_FILE.exists():
        raise FileNotFoundError(
            "Vector store not found. Please run python ingest.py first."
        )

    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["vocabulary"], data["chunks"]


def main():
    print("=" * 60)
    print("RAG From Scratch - Chat with Your Documents + Citations")
    print("=" * 60)

    vocabulary, idf_values, embedded_chunks = load_vector_store()

    print("Vector store loaded successfully.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        relevant_chunks = retrieve_relevant_chunks(
            question=question,
            embedded_chunks=embedded_chunks,
            vocabulary=vocabulary,
            idf_values=idf_values,
            top_k=3,
            min_score=0.01
        )

        answer = generate_answer(question, relevant_chunks)

        print("\nAnswer:")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()