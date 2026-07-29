from extract_text import extract_text
from chunker import chunk_text
from embeddings import create_chunk_embeddings

file_path = "documents/sample.txt"

text = extract_text(file_path)

chunks = chunk_text(text, chunk_size=100, overlap=20)

embedded_chunks, vocabulary = create_chunk_embeddings(chunks)

print("Vocabulary:")
print(vocabulary)

print("\nEmbedded Chunks:")

for chunk in embedded_chunks:
    print("Chunk ID:", chunk["chunk_id"])
    print("Text:", chunk["text"])
    print("Embedding:", chunk["embedding"])
    print("-" * 50)