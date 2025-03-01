from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob

# Đọc dữ liệu từ các file doc<ID>.txt
def load_documents(data_path):
    documents = []
    for file in glob(data_path):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n", 3)  # Tách URL, Title, trống, nội dung
            url = lines[0].replace("URL: ", "")
            title = lines[1].replace("Title: ", "")
            text = lines[3] if len(lines) > 3 else ""
            documents.append({"text": text, "metadata": {"url": url, "title": title}})
    return documents

# Chia nhỏ dữ liệu
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Kích thước mỗi đoạn (ký tự)
        chunk_overlap=50  # Chồng lấp để giữ ngữ cảnh
    )
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for text in split_texts:
            chunks.append({
                "text": text,
                "metadata": doc["metadata"]
            })
    return chunks

