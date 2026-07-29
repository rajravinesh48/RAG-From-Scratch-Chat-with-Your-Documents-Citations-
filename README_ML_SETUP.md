# Machine Learning Integration Setup

This update adds a supervised machine-learning question intent classifier
to the existing RAG project.

## Added ML Features

- TF-IDF feature extraction
- Logistic Regression intent classifier
- Seven question intent classes:
  - definition
  - process
  - file_support
  - component
  - comparison
  - troubleshooting
  - general
- ML confidence percentage
- Intent-based Top-K retrieval
- Admin-only Train/Retrain ML Model button
- Model status and validation accuracy on the dashboard
- ML information included in downloaded chat history

## Files to Replace/Add

```text
web_app.py
intent_classifier.py
train_intent_model.py
test_intent_classifier.py
requirements.txt
templates/login.html
templates/dashboard.html
ml_model/.gitkeep
```

Keep all your existing RAG files:

```text
generator.py
ingest.py
retriever.py
embeddings.py
chunker.py
extract_text.py
documents/
vector_store/
```

## Installation

Activate your virtual environment and run:

```powershell
pip install -r requirements.txt
```

## Train the Model

You can train from the command line:

```powershell
python train_intent_model.py
```

Or:

1. Run `python web_app.py`
2. Login as admin
3. Click **Train ML Intent Model**

The following files will be generated:

```text
ml_model/intent_model.joblib
ml_model/metrics.json
```

## Test the ML Model

```powershell
python test_intent_classifier.py
```

## Run the Website

```powershell
python web_app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Demo Accounts

```text
Admin: admin / admin123
User:  user / user123
```

Change the default passwords and Flask secret before real deployment.
