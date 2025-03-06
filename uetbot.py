from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

import random
import streamlit as st
import streamlit.components.v1 as components

model_file = 'models/vinallama-7b-chat_q5_0.gguf'
vector_db_path = 'vectorstores/my_db'
embedding_model_file = 'models/all-MiniLM-L6-v2-f16.gguf'

@st.cache_resource
def load_llm(model_file, model_type='llama', max_new_tokens=1024, temperature=0.01):
    llm = CTransformers(
        model=model_file,
        model_type=model_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt

@st.cache_resource
def create_qa_chain(_prompt, _llm, _db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type='stuff',
        retriever=_db.as_retriever(
            search_kwargs={
                "k": 3
            }
        ),
        return_source_documents=False,
        chain_type_kwargs={
            'prompt': _prompt
        }
    )
    
    return llm_chain

@st.cache_resource
def read_vector_db():
    embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def main():
    st.title("UET Chatbot")
    st.write("Hỏi bất kỳ điều gì về UET, tôi sẽ trả lời dựa trên dữ liệu crawl từ website!")
    st.divider()
    
    db = read_vector_db()
    llm = load_llm(model_file)
    
    template = """<|im_start|>system
                Bạn là một trợ lý AI thân thiện, chuyên trả lời câu hỏi về UET. Nếu không có thông tin phù hợp, hãy lịch sự nói rằng bạn không biết. Tránh lặp lại câu hỏi của người dùng.
                {context}<|im_end|>
                <|im_start|>user
                {question}<|im_end|>
                <|im_start|>assistant"""

    prompt = create_prompt(template)
    
    qa_chain = create_qa_chain(prompt, llm, db)

    # Lưu trữ lịch sử hội thoại
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Hiển thị tin nhắn cũ
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
  
    # Test
    query = st.chat_input("Nhập câu hỏi của bạn:")
    if query:
        # Hiển thị tin nhắn của user
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Xử lý truy vấn
        answer = "OK"
        with st.spinner("Đang xử lý..."):
            response = qa_chain.invoke({'query': query})
            answer = response['result']
        
        # Hiển thị phản hồi của bot
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        
if __name__ == "__main__":
    main()