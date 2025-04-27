# Local RAG Demo
This is a simple local project to demonstrate Retrieval-Augmented Generation (RAG).

## How it works
1. Index documents into a FAISS vector store.
2. Retrieve top-k documents relevant to a query.
3. Feed retrieved docs + query into a small text generator (GPT-2 locally).

## Setup
```bash
pip install -r requirements.txt
