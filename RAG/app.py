import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be.*")

import os
from indexing import build_faiss_index
from retrieval import load_faiss_index, retrieve_docs
from generation import generate_answer

# Paths
DOC_FOLDER = "knowledge_base"
INDEX_PATH = "embeddings/doc_index.faiss"

# Step 1: Build index (only once, then comment this out)
if not os.path.exists(INDEX_PATH):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    model, documents = build_faiss_index(DOC_FOLDER, INDEX_PATH)
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    documents = []
    for fname in os.listdir(DOC_FOLDER):
        with open(os.path.join(DOC_FOLDER, fname), 'r') as f:
            documents.append(f.read())

# Step 2: Load index
index = load_faiss_index(INDEX_PATH)

# Step 3: Query Loop
while True:
    query = input("\nEnter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    retrieved_docs = retrieve_docs(query, model, index, documents)
    answer = generate_answer(query, retrieved_docs)

    print("\n--- Retrieved Documents ---")
    for doc in retrieved_docs:
        print("-", doc[:200], "...\n")  # first 200 chars

    print("\n--- Generated Answer ---")
    print(answer)
    print("\n--- End of Response ---")