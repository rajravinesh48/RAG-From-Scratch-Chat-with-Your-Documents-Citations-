from extract_text import extract_text
from chunker import chunk_text
from embeddings import create_chunk_embeddings
from retriever import retrieve_relevant_chunks

file_path = "documents/sample.txt"

text = extract_text(file_path)

chunks = chunk_text(text, chunk_size=100, overlap=20)

embedded_chunks, vocabulary = create_chunk_embeddings(chunks)

question = input("Ask a question: ")

results = retrieve_relevant_chunks(
    question=question,
    embedded_chunks=embedded_chunks,
    vocabulary=vocabulary,
    top_k=3
)

print("\nTop Relevant Chunks:\n")

for chunk in results:
    print("Chunk ID:", chunk["chunk_id"])
    print("Score:", round(chunk["score"], 4))
    print("Text:", chunk["text"])
    print("-" * 50)