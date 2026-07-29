import json
from pathlib import Path

from extract_text import extract_text
from chunker import chunk_text
from embeddings import create_chunk_embeddings


DOCUMENTS_DIR = Path("documents")
VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_FILE = VECTOR_STORE_DIR / "store.json"


def ingest_documents():
    all_chunks = []

    supported_extensions = [".pdf", ".txt", ".docx"]

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)

    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.suffix.lower() not in supported_extensions:
            continue

        print(f"Reading file: {file_path.name}")

        text = extract_text(file_path)

        chunks = chunk_text(text, chunk_size=60, overlap=10)
        
        for chunk in chunks:
            all_chunks.append({
                "filename": file_path.name,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            })

    if not all_chunks:
        if VECTOR_STORE_FILE.exists():
            VECTOR_STORE_FILE.unlink()

        print("No documents found. Vector store cleared.")
        return

    embedded_chunks, vocabulary, idf_values = create_chunk_embeddings(all_chunks)

    store_data = {
    "vocabulary": vocabulary,
    "idf_values": idf_values,
    "chunks": embedded_chunks
    }

    with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as file:
        json.dump(store_data, file, indent=4)

    print("Vector store created successfully.")
    print(f"Total chunks saved: {len(embedded_chunks)}")
    print(f"Saved at: {VECTOR_STORE_FILE}")


if __name__ == "__main__":
    ingest_documents()