import json
import os
import shutil
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Xác định thư mục gốc của project (hospital_booking_chatbot_v1)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(find_dotenv(str(BASE_DIR / ".env")))


def load_pdf_documents(data_dir: Path) -> list[Document]:
    """Đọc và cắt nhỏ các file PDF trong thư mục data (ví dụ: quy trình bệnh viện)."""
    loader = DirectoryLoader(str(data_dir), glob="*.pdf", loader_cls=PyPDFLoader)
    pages = loader.load()
    if not pages:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = text_splitter.split_documents(pages)

    for doc in splits:
        raw_source = doc.metadata.get("source", "")
        filename = os.path.basename(raw_source)
        if "quy_trinh" in filename:
            doc.metadata["category"] = "hospital_procedures"
        elif "trieu_chung" in filename:
            doc.metadata["category"] = "symptoms"
        else:
            doc.metadata["category"] = "general"
        doc.metadata["source_file"] = filename

    return splits


def load_jsonl_documents(data_dir: Path) -> list[Document]:
    """Đọc và chia nhỏ các tài liệu JSONL y khoa (ví dụ: YouMed clean markdown)."""
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )

    all_chunks: list[Document] = []
    for filepath in jsonl_files:
        filename = filepath.name
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"Bỏ qua dòng {line_idx+1} do lỗi JSON: {e}")
                    continue

                disease = data.get("disease", "").strip()
                title = data.get("title", "").strip()
                content = data.get("markdown_content", "").strip()
                url = data.get("url", "").strip()
                author = data.get("author", "").strip()

                if not content:
                    continue

                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    page_content = f"Bệnh: {disease}\n{chunk}" if disease else chunk
                    doc = Document(
                        page_content=page_content,
                        metadata={
                            "category": "symptoms",
                            "disease": disease,
                            "title": title,
                            "source_url": url,
                            "author": author,
                            "source_file": filename,
                        },
                    )
                    all_chunks.append(doc)

    return all_chunks


def run_ingestion():
    data_dir = BASE_DIR / "data"
    persist_directory = str(BASE_DIR / "chroma_db")

    # 1. Dọn sạch database cũ để tránh bị trùng lặp chunk
    if os.path.exists(persist_directory):
        print(f"Đang làm mới thư mục '{persist_directory}'...")
        shutil.rmtree(persist_directory)

    # 2. Tải và xử lý tài liệu PDF
    pdf_splits = load_pdf_documents(data_dir)
    print(f"Đã xử lý {len(pdf_splits)} chunks từ tài liệu PDF.")

    # 3. Tải và xử lý tài liệu JSONL (YouMed)
    jsonl_splits = load_jsonl_documents(data_dir)
    print(f"Đã xử lý {len(jsonl_splits)} chunks từ tài liệu JSONL.")

    all_splits = pdf_splits + jsonl_splits
    if not all_splits:
        print(f"Cảnh báo: Không có dữ liệu nào trong '{data_dir}' để nạp vào database.")
        return

    print(f"Tổng số chunks cần nhúng: {len(all_splits)}")

    # 4. Nhúng và lưu trữ vào ChromaDB theo từng batch để tối ưu tốc độ và an toàn bộ nhớ
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    batch_size = 250
    vector_store = None
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i : i + batch_size]
        print(f"Đang lưu batch {i // batch_size + 1}/{(len(all_splits) + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory=persist_directory,
            )
        else:
            vector_store.add_documents(documents=batch)

    print("Hoàn tất nạp dữ liệu vào ChromaDB thành công!")


if __name__ == "__main__":
    run_ingestion()
