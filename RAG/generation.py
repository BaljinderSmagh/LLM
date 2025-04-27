#for generating answers

from transformers import pipeline

def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the following based on the context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    generator = pipeline("text-generation", model="gpt2", max_length=300)
    output = generator(prompt)[0]['generated_text']
    return output
