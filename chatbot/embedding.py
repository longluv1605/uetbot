# Nhúng các đoạn
def embed_chunks(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
