import faiss
import numpy as np
import pickle

# Tạo và lưu chỉ mục FAISS
def create_vector_store(embeddings, chunks, save_path, index_path="vector_store.index"):
    dimension = embeddings.shape[1]  # Chiều của vector
    index = faiss.IndexFlatL2(dimension)  # Chỉ mục L2
    index.add(embeddings)
    faiss.write_index(index, index_path)  # Lưu chỉ mục
    with open(save_path, "wb") as f:
        pickle.dump(chunks, f)  # Lưu metadata

def load_vector_store(save_path, index_path):
    vector_store = faiss.read_index(index_path)  # Load chỉ mục
    with open(save_path, "rb") as f:
        data = pickle.load(f)  # Load metadata
    
    return vector_store, data
        
