from pathlib import Path
from pypdf import PdfReader
from docx import Document


def extract_text_from_pdf(file_path):
    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def extract_text(file_path):
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    elif extension == ".docx":
        return extract_text_from_docx(file_path)

    elif extension == ".txt":
        return extract_text_from_txt(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")