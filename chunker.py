def chunk_text(text, chunk_size=80, overlap=20):
    """
    Splits text into word-based chunks with overlap.

    chunk_size = number of words in one chunk
    overlap = repeated words between chunks
    """

    chunks = []

    if not text:
        return chunks

    text = text.replace("\n", " ").strip()
    words = text.split()

    start = 0
    chunk_id = 1

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        # Stop if remaining chunk is too small
        if len(chunk_words) < 5 and chunk_id > 1:
            break

        chunk_text_value = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_value
        })

        # If this is the last chunk, stop
        if end >= len(words):
            break

        chunk_id += 1
        start = end - overlap

    return chunks