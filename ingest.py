import json
from pathlib import Path

from chunker import chunk_text
from embeddings import create_chunk_embeddings
from extract_text import extract_document_sections


DOCUMENTS_DIR = Path("documents")
VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_FILE = (
    VECTOR_STORE_DIR
    / "store.json"
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}

CHUNK_SIZE = 60
CHUNK_OVERLAP = 10


def build_document_chunks(
    file_path,
):
    """
    Extracts a document and creates chunks while preserving
    PDF page numbers.

    PDF chunks:
        filename
        page_number
        chunk_id
        page_chunk_id
        text

    TXT/DOCX chunks:
        page_number=None
    """
    sections = extract_document_sections(
        file_path
    )

    document_chunks = []
    document_chunk_id = 1

    for section in sections:
        page_number = section.get(
            "page_number"
        )

        text = str(
            section.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        page_chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for page_chunk in page_chunks:
            document_chunks.append(
                {
                    "filename": (
                        file_path.name
                    ),

                    # Physical PDF page number.
                    # None for TXT and DOCX.
                    "page_number": (
                        page_number
                    ),

                    # Unique chunk number across
                    # the complete document.
                    "chunk_id": (
                        document_chunk_id
                    ),

                    # Chunk number inside the
                    # individual PDF page/section.
                    "page_chunk_id": (
                        page_chunk.get(
                            "chunk_id"
                        )
                    ),

                    "text": page_chunk.get(
                        "text",
                        "",
                    ),
                }
            )

            document_chunk_id += 1

    return document_chunks


def ingest_documents():
    """
    Reads all supported documents, chunks them, creates
    embeddings, and writes the JSON vector store.

    PDF page_number metadata is preserved all the way
    into vector_store/store.json.
    """
    all_chunks = []

    DOCUMENTS_DIR.mkdir(
        exist_ok=True
    )

    VECTOR_STORE_DIR.mkdir(
        exist_ok=True
    )

    document_paths = sorted(
        (
            file_path
            for file_path
            in DOCUMENTS_DIR.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ),
        key=lambda path: path.name.lower(),
    )

    for file_path in document_paths:
        print(
            f"Reading file: "
            f"{file_path.name}"
        )

        document_chunks = (
            build_document_chunks(
                file_path
            )
        )

        all_chunks.extend(
            document_chunks
        )

        pdf_pages = {
            chunk["page_number"]
            for chunk in document_chunks
            if chunk.get(
                "page_number"
            ) is not None
        }

        print(
            f"  Chunks created: "
            f"{len(document_chunks)}"
        )

        if pdf_pages:
            print(
                f"  PDF pages indexed: "
                f"{len(pdf_pages)}"
            )

    if not all_chunks:
        if VECTOR_STORE_FILE.exists():
            VECTOR_STORE_FILE.unlink()

        print(
            "No documents found. "
            "Vector store cleared."
        )

        return

    (
        embedded_chunks,
        vocabulary,
        idf_values,
    ) = create_chunk_embeddings(
        all_chunks
    )

    store_data = {
        "vocabulary": vocabulary,
        "idf_values": idf_values,
        "chunks": embedded_chunks,
    }

    with open(
        VECTOR_STORE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            store_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "Vector store created successfully."
    )

    print(
        f"Total chunks saved: "
        f"{len(embedded_chunks)}"
    )

    print(
        f"Saved at: "
        f"{VECTOR_STORE_FILE}"
    )


if __name__ == "__main__":
    ingest_documents()
