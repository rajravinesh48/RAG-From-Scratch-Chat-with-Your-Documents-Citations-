# RAG From Scratch - Chat with Your Documents + Citations

This project is a simple Retrieval Augmented Generation system built from scratch using Python.

The system reads documents, splits them into chunks, converts chunks into vectors, stores them in a vector store, retrieves relevant chunks based on a user question, and generates an answer with citations.

## Features

* Read PDF, TXT, and DOCX files
* Extract text from documents
* Split text into word-based chunks
* Create embeddings using Bag of Words
* Store vectors and metadata in JSON format
* Retrieve relevant chunks using cosine similarity
* Generate answer using only retrieved document chunks
* Show citations with filename and chunk ID

## Tech Stack

* Python 3.10+
* NumPy
* pypdf
* python-docx

## Project Structure

```text
rag-from-scratch-citations/
│
├── documents/
│   └── sample.txt
│
├── vector_store/
│
├── app.py
├── ingest.py
├── extract_text.py
├── chunker.py
├── embeddings.py
├── retriever.py
├── generator.py
│
├── test_extract.py
├── test_chunker.py
├── test_embeddings.py
├── test_retriever.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add documents

Place your PDF, TXT, or DOCX files inside the `documents/` folder.

### 5. Create vector store

```bash
python ingest.py
```

### 6. Start chat application

```bash
python app.py
```

### 7. Ask questions

Example:

```text
What is RAG?
```

The system will return an answer based on the uploaded documents and show citations.

## Example Output

```text
Answer:
Based on the uploaded documents:

RAG means Retrieval Augmented Generation.

Citations:
- sample.txt | Chunk ID: 1 | Score: 0.343
```

## Explanation

This project follows the complete RAG pipeline:

1. Ingest documents from the documents folder
2. Extract text from PDF, TXT, and DOCX files
3. Split text into smaller chunks
4. Convert chunks into numeric vectors
5. Store vectors with metadata
6. Convert user question into a vector
7. Compare question vector with chunk vectors using cosine similarity
8. Retrieve top matching chunks
9. Generate answer using retrieved chunks only
10. Display citations with filename and chunk ID

## Conclusion

This is a basic RAG implementation from scratch. It does not depend on external vector databases or AI frameworks. The purpose is to understand how document-based question answering works internally.
