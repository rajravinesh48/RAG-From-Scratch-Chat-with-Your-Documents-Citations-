from pathlib import Path

from docx import Document
from pypdf import PdfReader


def extract_pages_from_pdf(file_path):
    """
    Extracts PDF text page by page.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]

    Empty pages are skipped.
    """
    file_path = Path(file_path)
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_text = page.extract_text()

        if not page_text:
            continue

        page_text = page_text.strip()

        if not page_text:
            continue

        pages.append(
            {
                "page_number": page_number,
                "text": page_text,
            }
        )

    return pages


def extract_text_from_pdf(file_path):
    """
    Backward-compatible PDF extractor.

    Existing parts of the project that expect one string
    can continue using this function.
    """
    pages = extract_pages_from_pdf(file_path)

    return "\n".join(
        page["text"]
        for page in pages
    )


def extract_text_from_docx(file_path):
    """
    Extracts text from a DOCX file.
    """
    file_path = Path(file_path)
    doc = Document(file_path)

    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def extract_text_from_txt(file_path):
    """
    Extracts text from a TXT file.
    """
    file_path = Path(file_path)

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        return file.read()


def extract_document_sections(file_path):
    """
    Returns document content together with page metadata.

    PDF:
        One section per physical PDF page.

    TXT / DOCX:
        One section with page_number=None because those
        formats do not have reliable physical page numbers
        in this simple from-scratch implementation.

    Example:
        [
            {
                "page_number": 3,
                "text": "..."
            }
        ]
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pages_from_pdf(
            file_path
        )

    if extension == ".docx":
        text = extract_text_from_docx(
            file_path
        )

        return (
            [
                {
                    "page_number": None,
                    "text": text,
                }
            ]
            if text.strip()
            else []
        )

    if extension == ".txt":
        text = extract_text_from_txt(
            file_path
        )

        return (
            [
                {
                    "page_number": None,
                    "text": text,
                }
            ]
            if text.strip()
            else []
        )

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def extract_text(file_path):
    """
    Original public helper kept for backward compatibility.

    Returns the complete document as one string.
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(
            file_path
        )

    if extension == ".docx":
        return extract_text_from_docx(
            file_path
        )

    if extension == ".txt":
        return extract_text_from_txt(
            file_path
        )

    raise ValueError(
        f"Unsupported file type: {extension}"
    )
