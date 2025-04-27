import streamlit as st
from indexing import build_faiss_index
from retrieval import load_faiss_index, retrieve_docs
from generation import generate_answer
import os

# ---- CONFIG ----
DOC_FOLDER = "knowledge_base"
INDEX_PATH = "embeddings/doc_index.faiss"

# ---- Load / Build Index ----
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

index = load_faiss_index(INDEX_PATH)

# ---- Streamlit UI ----
st.title("🧠 Local RAG Demo App")
st.caption("Retrieval-Augmented Generation built from scratch, running locally!")

query = st.text_input("Enter your query here:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a valid query!")
    else:
        with st.spinner("Retrieving and generating answer..."):
            retrieved_docs = retrieve_docs(query, model, index, documents)
            answer = generate_answer(query, retrieved_docs)

            st.subheader("Generated Answer:")
            st.success(answer)

            st.subheader("Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                with st.expander(f"Document {i}"):
                    st.write(doc)
