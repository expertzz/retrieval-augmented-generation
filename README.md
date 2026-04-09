# Retrieval-Augmented Generation

A RAG system that lets you ask questions about a PDF using semantic search and Claude.

## Setup 

1. Install dependencies:
   ```pip install -r requirements.txt```

2. Download NLTK data (once):
   ```python3 -c "import nltk; nltk.download('punkt_tab')"```

3. Add your API keys to a .env file:
   OPENAI_API_KEY, ANTHROPIC_API_KEY

## Usage

First time, index a new PDF:
```python3 main.py [pdf_name].pdf```

Next time, index already exist:
```python3 main.py```
