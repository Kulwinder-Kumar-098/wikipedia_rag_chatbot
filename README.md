# Wikipedia RAG Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot that fetches a Wikipedia article, cleans and chunks the text, creates embeddings, stores them in a FAISS index, and answers user questions using retrieved context with an LLM.

## Features

* Fetches Wikipedia content using the Wikipedia API
* Cleans and preprocesses raw article text
* Splits the article into meaningful chunks
* Generates embeddings with `sentence-transformers`
* Stores and searches vectors using FAISS
* Answers questions using retrieved context through Groq

## Project Structure

```bash
data/
  raw/                 # Raw fetched Wikipedia article
  processed/           # Cleaned text and chunked data
  embeddings/          # Embedding outputs and metadata
index/
  wiki_index.faiss     # FAISS vector index
src/
  fetch.py             # Download Wikipedia article
  clean.py             # Clean raw text
  chunk.py             # Split text into chunks
  embeddings.py        # Generate embeddings
  faiss_store.py       # Build FAISS index
  retrieval.py         # Retrieve relevant chunks
  llm_integration.py   # Generate answers with Groq
  wikipedia_chatbot.py # Main chatbot UI
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Fetch the Wikipedia article:

```bash
python src/fetch.py
```

2. Clean the article:

```bash
python src/clean.py
```

3. Create chunks:

```bash
python src/chunk.py
```

4. Generate embeddings:

```bash
python src/embeddings.py
```

5. Build the FAISS index:

```bash
python src/faiss_store.py
```

6. Start the chatbot:

```bash
python src/wikipedia_chatbot.py
```

## Notes

* The chatbot expects the FAISS index and chunk file to exist before startup.
* If the embedding model or LLM is unavailable, the pipeline will fail until the required dependencies and API credentials are configured.
* This project is a simple demonstration of a local RAG pipeline built on Wikipedia content.

## Tech Stack

* Python
* Wikipedia API
* Sentence Transformers
* FAISS
* Groq
* LangChain-style retrieval pipeline
