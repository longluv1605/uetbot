from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# from sentence_transformers import SentenceTransformer



data_path = 'data'
vector_db_path = 'vectorstores/my_db'
embedding_model_file = 'models/all-MiniLM-L6-v2-f16.gguf'

def create_vector_stores():
    
    # Load documents
    loader = DirectoryLoader(data_path, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Embedding
    embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)
    
    # FAISS vector store
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    
    return db

if __name__ == '__main__':
    create_vector_stores()
    