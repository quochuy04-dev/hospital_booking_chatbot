import os
import shutil
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Xác định thư mục gốc của project (hospital_booking_chatbot_v1)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(find_dotenv(str(BASE_DIR / ".env")))


def run_ingestion():
  data_dir = str(BASE_DIR / "data")
  persist_directory = str(BASE_DIR / "chroma_db")

  # 2. Dọn sạch database cũ để tránh bị trùng lặp chunk
  if os.path.exists(persist_directory):
    print(f"Đang làm mới thư mục '{persist_directory}'...")
    shutil.rmtree(persist_directory)

  # 3. Đọc toàn bộ PDF trong thư mục data
  loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
  pages = loader.load()

  if not pages:
    print(f"Cảnh báo: Không tìm thấy file PDF nào trong '{data_dir}'.")
    return

  # 4. Cắt nhỏ văn bản (Chunking)
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=600,
      chunk_overlap=100,
      separators=["\n\n", "\n", ". ", " ", ""],
  )
  splits = text_splitter.split_documents(pages)

  # 5. Gắn nhãn Metadata phục vụ truy vấn theo category
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

  # 6. Nhúng và lưu trữ vào ChromaDB
  embedding = OpenAIEmbeddings(model="text-embedding-3-small")

  Chroma.from_documents(
      documents=splits,
      embedding=embedding,
      persist_directory=persist_directory,
  )


if __name__ == "__main__":
  run_ingestion()