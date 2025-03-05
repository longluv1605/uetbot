from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

model_file = 'models/vinallama-7b-chat_q5_0.gguf'
vector_db_path = 'vectorstores/my_db'
embedding_model_file = 'models/all-MiniLM-L6-v2-f16.gguf'

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

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(
            search_kwargs={
                "k": 3
            }
        ),
        return_source_documents=False,
        chain_type_kwargs={
            'prompt': prompt
        }
    )
    
    return llm_chain

def read_vector_db():
    embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def main():
    db = read_vector_db()
    llm = load_llm(model_file)

    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = create_prompt(template)
    
    qa_chain = create_qa_chain(prompt, llm, db)

    # Test
    while True:
        question = input("Hãy nhập câu hỏi, nếu muốn dừng, hãy nhập 'stop':\n\t")
        if question.lower() == 'stop':
            return
        response = qa_chain.invoke({'query': question})
        print(f"Trả lời\n\t: {response}")
        print()

if __name__ == "__main__":
    main()