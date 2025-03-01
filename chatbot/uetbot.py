import numpy as np

def retrieve_documents(query, model_embed, vector_store, chunk_data, k=3):
    query_embedding = model_embed.encode([query])[0]
    distances, indices = vector_store.search(np.array([query_embedding]), k)
    return [chunk_data[idx] for idx in indices[0]]

def generate_answer(query, retrieved_docs, tokenizer, model):
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    prompt = f"Dựa trên dữ liệu sau, trả lời câu hỏi:\nCâu hỏi: {query}\nDữ liệu:\n{context}\nTrả lời:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

def chatbot(query, model_embed, tokenizer, llm_model, vector_store, chunk_data, k=3):
    retrieved_docs = retrieve_documents(query, model_embed, vector_store, chunk_data, k)
    answer = generate_answer(query, retrieved_docs, tokenizer, llm_model)
    return answer, retrieved_docs