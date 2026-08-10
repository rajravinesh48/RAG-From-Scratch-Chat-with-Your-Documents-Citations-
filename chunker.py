import re


SEPARATOR_RE = re.compile(r"^[=\-_*#~]{3,}$")
HEADING_RE = re.compile(r"^(?:SECTION|CHAPTER)\b", flags=re.IGNORECASE)


def _clean_line(line):
    return re.sub(r"\s+", " ", str(line or "")).strip()


def _split_long_unit(text, chunk_size, overlap):
    words = str(text or "").split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [str(text).strip()]

    parts = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        part = " ".join(words[start:end]).strip()
        if part:
            parts.append(part)
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return parts


def _text_to_logical_units(text):
    raw_lines = (
        str(text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    )
    lines = [_clean_line(line) for line in raw_lines]

    units = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line:
            index += 1
            continue

        if SEPARATOR_RE.fullmatch(line):
            index += 1
            continue

        if HEADING_RE.match(line):
            units.append(line)
            index += 1
            continue

        if re.match(r"^Q\s*:", line, flags=re.IGNORECASE):
            qa_parts = [line]
            index += 1

            while index < len(lines):
                next_line = lines[index]

                if not next_line:
                    index += 1
                    continue

                if (
                    SEPARATOR_RE.fullmatch(next_line)
                    or HEADING_RE.match(next_line)
                    or re.match(r"^Q\s*:", next_line, flags=re.IGNORECASE)
                ):
                    break

                qa_parts.append(next_line)
                index += 1

                if re.match(r"^A\s*:", next_line, flags=re.IGNORECASE):
                    break

            units.append(" ".join(qa_parts).strip())
            continue

        sentences = re.split(r"(?<=[.!?])\s+", line)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)

        index += 1

    return units


def chunk_text(text, chunk_size=80, overlap=20):
    """
    Sentence/Q&A-aware chunking with overlap.

    Improvements:
    - keeps Q: + A: together
    - avoids gluing SECTION headings to answers
    - prefers complete sentence boundaries
    - preserves overlap between adjacent chunks
    """
    if not text:
        return []

    chunk_size = max(int(chunk_size or 80), 20)
    overlap = max(min(int(overlap or 0), chunk_size - 1), 0)

    units = []
    for unit in _text_to_logical_units(text):
        units.extend(_split_long_unit(unit, chunk_size, overlap))

    if not units:
        return []

    chunks = []
    current_units = []
    current_word_count = 0

    def flush_current():
        nonlocal current_units, current_word_count

        if not current_units:
            return

        value = " ".join(current_units).strip()
        if value:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "text": value,
            })

        overlap_units = []
        overlap_words = 0

        if overlap > 0:
            for previous_unit in reversed(current_units):
                word_count = len(previous_unit.split())

                if overlap_units and overlap_words + word_count > overlap:
                    break

                overlap_units.insert(0, previous_unit)
                overlap_words += word_count

                if overlap_words >= overlap:
                    break

        current_units = overlap_units
        current_word_count = sum(len(unit.split()) for unit in current_units)

    for unit in units:
        unit_word_count = len(unit.split())

        if current_units and current_word_count + unit_word_count > chunk_size:
            flush_current()

            while (
                current_units
                and current_word_count + unit_word_count > chunk_size
            ):
                removed = current_units.pop(0)
                current_word_count -= len(removed.split())

        current_units.append(unit)
        current_word_count += unit_word_count

    flush_current()
    return chunks
