# RAG From Scratch — Final Submission Checklist

## Core files to submit

- web_app.py
- extract_text.py
- chunker.py
- embeddings.py
- ingest.py
- retriever.py
- generator.py
- requirements.txt
- README.md
- .env.example
- .gitignore
- templates/
- static/
- documents/ with safe sample documents
- vector_store/store.json (optional but useful for demonstration)

## Optional extra-feature files

- intent_classifier.py
- train_intent_model.py
- ml_model/
- github_storage_service.py (only if the live app actually uses it)
- tests/

## Do not submit

- .env
- venv/
- __pycache__/
- users.json
- storage_service.py if Supabase is no longer used
- test_storage.txt
- API keys/tokens
- backup files

## Gemini

Gemini is NOT required by the supplied project specification.

The required RAG pipeline is:
1. Extract PDF/TXT/DOCX text
2. Chunk text
3. Create vectors/embeddings
4. Store vectors + metadata
5. Retrieve with cosine similarity
6. Generate an answer from retrieved context only
7. Show citations

This project can satisfy step 6 with:
- Ollama/Gemma (optional local LLM), or
- the local extractive fallback.

Gemini should be described as an optional cloud enhancement.

## Important fixes included in this pack

1. embeddings.py now preserves page_number and page_chunk_id.
2. test_embeddings.py now handles the 3 values returned by create_chunk_embeddings().
3. test_retriever.py now handles idf_values correctly.
4. test_gemini.py skips safely when no Gemini key exists.
5. requirements.txt removes the unused Supabase dependency.
