#for indexing documents for RAG
# This code is used to build a FAISS index for a set of documents.

# It uses the SentenceTransformer model to encode the documents into embeddings,
# which are then added to a FAISS index for efficient similarity search.
# The index is saved to a specified path for later use.
# It also returns the model and the list of documents for further processing.
# Import necessary libraries
import os
import  faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(doc_folder, index_save_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    documents = []
    filenames = []

    for fname in os.listdir(doc_folder):
        with open(os.path.join(doc_folder, fname), 'r') as f:
            documents.append(f.read())
            filenames.append(fname)

    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_save_path)
    return model, documents
