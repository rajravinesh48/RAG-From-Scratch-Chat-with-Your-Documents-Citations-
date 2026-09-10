# RAG From Scratch - Project README

## Overview

This project implements a complete Retrieval Augmented Generation (RAG)
pipeline built from scratch.

Pipeline:

    Documents
        |
        v
    extract_text.py
        |
        v
    chunker.py
        |
        v
    embeddings.py
        |
        v
    vector_store/store.json
        |
        v
    retriever.py
        |
        v
    generator.py
        |
        v
    web_app.py

The system supports PDF, DOCX and TXT documents.

## Document Processing

### extract_text.py

Responsibilities:

-   PDF text extraction
-   DOCX paragraph extraction
-   TXT file reading
-   PDF page metadata preservation

PDF documents maintain:

-   filename
-   page_number
-   text content

------------------------------------------------------------------------

## Chunking

### chunker.py

Features:

-   Sentence-aware chunking
-   Q&A-aware chunking
-   Section heading handling
-   Chunk overlap support

Default ingestion settings:

    chunk_size = 60
    overlap = 10

------------------------------------------------------------------------

## Embeddings

### embeddings.py

The project uses custom TF-IDF based embeddings.

Implemented features:

-   Vocabulary creation
-   TF calculation
-   IDF calculation
-   Vector generation using NumPy
-   Metadata preservation

Each stored chunk contains:

    filename
    page_number
    chunk_id
    page_chunk_id
    text
    embedding

Vector store structure:

    store.json

    {
        vocabulary,
        idf_values,
        chunks
    }

------------------------------------------------------------------------

## Ingestion

### ingest.py

Creates the vector database.

Process:

1.  Reads files from documents folder
2.  Extracts document sections
3.  Creates chunks
4.  Generates TF-IDF vectors
5.  Saves vector_store/store.json

Run:

    python ingest.py

------------------------------------------------------------------------

# Retriever

## retriever.py

The retriever performs document relevance search.

Features:

## TF-IDF Similarity

Uses cosine similarity between:

-   question vector
-   document chunk vector

## Keyword Matching

Includes:

-   meaningful query words
-   stop word filtering
-   query coverage calculation
-   keyword matching

## Query Understanding

Handles:

-   question word removal
-   singular/plural normalization
-   yes/no question handling
-   unsupported query detection

## Retrieval Output

Returns:

-   filename
-   page number
-   chunk id
-   similarity score
-   matched keywords
-   relevant text

------------------------------------------------------------------------

# Generator

## generator.py

Generates answers using retrieved context only.

Supported modes:

### Local Extractive RAG

Default mode.

No external LLM required.

### Optional Ollama

Example:

    gemma3:4b

### Optional Gemini

Cloud enhancement.

Gemini is optional and not required for the core RAG pipeline.

------------------------------------------------------------------------

# Flask Application

## web_app.py

Provides:

-   Chat interface
-   Document upload
-   Retrieval execution
-   Answer generation
-   Session handling
-   Source display

Supported uploads:

    .pdf
    .docx
    .txt

------------------------------------------------------------------------

# Intent Classification (Optional)

Files:

    intent_classifier.py
    train_intent_model.py

Uses:

-   TF-IDF features
-   Logistic Regression model

Supported categories:

-   definition
-   process
-   file support
-   component
-   comparison
-   troubleshooting

Training:

    python train_intent_model.py

------------------------------------------------------------------------

# Installation

Create environment:

    python -m venv venv

Activate:

Windows:

    venv\Scripts\activate

Install:

    pip install -r requirements.txt

------------------------------------------------------------------------

# Running Project

## Add Documents

Place files:

    documents/

Supported:

    PDF
    DOCX
    TXT

## Create Vector Store

    python ingest.py

## Start Application

    python web_app.py

------------------------------------------------------------------------

# Testing

Available test scripts:

    test_extract.py
    test_chunker.py
    test_embeddings.py
    test_retriever.py
    test_intent_classifier.py
    test_gemini.py

------------------------------------------------------------------------

# Environment

Example:

    LLM_PROVIDER=local

    OLLAMA_ENABLED=true
    OLLAMA_MODEL=gemma3:4b

    GEMINI_ENABLED=false
    GEMINI_API_KEY=

------------------------------------------------------------------------

# Main Dependencies

-   Flask
-   NumPy
-   Scikit-learn
-   SciPy
-   PyPDF
-   python-docx
-   Joblib
-   OpenAI client for optional Gemini compatibility

------------------------------------------------------------------------

# Submission Structure

Core files:

    web_app.py
    extract_text.py
    chunker.py
    embeddings.py
    ingest.py
    retriever.py
    generator.py
    requirements.txt
    README.md
    .env.example
    .gitignore
    templates/
    static/
    documents/

Optional:

    intent_classifier.py
    train_intent_model.py
    tests/
    github_storage_service.py

------------------------------------------------------------------------

# Security

Do not commit:

    .env
    users.json
    API keys
    tokens
    venv/
    __pycache__/
    runtime files
    backup files

------------------------------------------------------------------------

# Architecture Principle

This is a RAG-from-scratch implementation.

Retrieval is implemented locally using:

-   custom document extraction
-   custom chunking
-   TF-IDF vector generation
-   NumPy similarity calculations
-   relevance filtering

External LLMs are optional and used only for answer generation.
