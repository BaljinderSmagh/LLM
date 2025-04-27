#for retrieving documents for RAG
# This code is used to retrieve documents from a FAISS index based on a query.
import faiss

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def retrieve_docs(query, model, index, documents, top_k=2):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in I[0]]
    return retrieved_docs
