import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from chatbot.uetbot import chatbot
from chatbot.embedding import embed_chunks
from chatbot.preprocessing import load_documents, split_documents
from chatbot.database import create_vector_store, load_vector_store

import warnings
warnings.filterwarnings("ignore")

def main(preprocessed=False):
    save_path = 'chunk_data.pkl'
    # Khởi tạo mô hình embedding
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    if not preprocessed:
        data_path = "data/doc*.txt"
        
        # Chunking
        documents = load_documents(data_path)
        chunks = split_documents(documents)
        print(f"Tổng số đoạn: {len(chunks)}")
        
        # Embedding
        embeddings = embed_chunks(chunks, model_embed)
        print(f"Kích thước embedding: {embeddings.shape}")
        
        # Chạy và lưu
        create_vector_store(embeddings, chunks, save_path)
        print("Đã tạo thành công!")
    
    # Load database
    vector_store, chunk_data = load_vector_store(save_path, index_path='vector_store.index')
    print("Đã tải thành công!")
    
    # Khởi tạo TinyLlama
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    llm = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto"
    )
    
    # Thử nghiệm
    query = "Chương trình đào tạo Khoa học và Kỹ thuật dữ liệu – Ngành Khoa học dữ liệu là gì?"
    answer, docs = chatbot(query, model_embed, tokenizer, llm_model, vector_store, chunk_data)
    print(f"Trả lời: {answer}")
    # print("Nguồn:")
    # for doc in docs:
    #     print(f"- {doc['metadata']['url']} ({doc['metadata']['title']})")
        
if __name__ == '__main__':
    main(preprocessed=True)