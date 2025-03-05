# UET Chatbot

Đây là dự án cá nhân với mục đích thực hành xây dựng ứng dụng RAG với LLMs.

## Các bước sử dụng

1. Tải models:
    - [Embedding model](https://huggingface.co/caliex/all-MiniLM-L6-v2-f16.gguf/blob/main/all-MiniLM-L6-v2-f16.gguf)
    - [LLM model](https://huggingface.co/vilm/vinallama-7b-chat-GGUF/blob/main/vinallama-7b-chat_q5_0.gguf)
2. Xây dựng database
    - Chạy lệnh `python prepare_vector_db.py`
3. Chạy bot
    - Sử dụng lệnh `python uetbot.py`
