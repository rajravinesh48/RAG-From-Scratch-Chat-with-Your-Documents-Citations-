from extract_text import extract_text
from chunker import chunk_text
from embeddings import create_chunk_embeddings
from retriever import retrieve_relevant_chunks


file_path = "documents/sample.txt"

text = extract_text(file_path)

chunks = chunk_text(
    text,
    chunk_size=100,
    overlap=20,
)

(
    embedded_chunks,
    vocabulary,
    idf_values,
) = create_chunk_embeddings(chunks)

question = input("Ask a question: ").strip()

results = retrieve_relevant_chunks(
    question=question,
    embedded_chunks=embedded_chunks,
    vocabulary=vocabulary,
    idf_values=idf_values,
    top_k=3,
)


print("\nTop Relevant Chunks:\n")

if not results:
    print("No relevant chunks found.")

for chunk in results:
    print(
        "Filename:",
        chunk.get("filename", "-"),
    )

    if chunk.get("page") is not None:
        print(
            "Page:",
            chunk["page"],
        )

    print(
        "Chunk ID:",
        chunk.get("chunk_id", "-"),
    )

    print(
        "Score:",
        round(
            float(chunk.get("score", 0)),
            4,
        ),
    )

    print(
        "Matched Keywords:",
        ", ".join(
            chunk.get(
                "matched_keywords",
                [],
            )
        ),
    )

    print(
        "Text:",
        chunk.get("text", ""),
    )

    print("-" * 50)
