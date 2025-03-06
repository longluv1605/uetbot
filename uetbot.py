from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st


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
    
    db = read_vector_db()
    llm = load_llm(model_file)

    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = create_prompt(template)
    
    qa_chain = create_qa_chain(prompt, llm, db)

    # Test
    query = st.text_input("Nhập câu hỏi của bạn:", "")
    if st.button("Gửi"):
        if query:
            with st.spinner("Đang xử lý..."):
                response = qa_chain.invoke({'query': query})
                st.subheader("Trả lời: ")
                st.write(response['result'])
        else:
            st.warning("Vui lòng nhập câu hỏi!")

if __name__ == "__main__":
    main()