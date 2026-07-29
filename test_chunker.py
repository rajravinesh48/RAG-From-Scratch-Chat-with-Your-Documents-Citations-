from extract_text import extract_text
from chunker import chunk_text

# File path from documents folder
file_path = "documents/sample.txt"

# Extract text from PDF/TXT/DOCX
text = extract_text(file_path)

# Convert extracted text into chunks
# Here chunk_size means number of words if you updated chunker.py to word-based chunking
chunks = chunk_text(text, chunk_size=20, overlap=5)

# Print all chunks
for chunk in chunks:
    print("Chunk ID:", chunk["chunk_id"])
    print(chunk["text"])
    print("-" * 50)