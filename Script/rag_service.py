import os
from pathlib import Path
from typing import List
from dotenv import find_dotenv, load_dotenv

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Xác định thư mục gốc của project (hospital_booking_chatbot_v1)
BASE_DIR = Path(__file__).resolve().parent.parent
_ = load_dotenv(find_dotenv(str(BASE_DIR / ".env")))

# Khởi tạo VectorStore dùng chung
persist_directory = str(BASE_DIR / "chroma_db")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)

# -------------------------------------------------------------
# 1. RAG FAQ QUY TRÌNH
# -------------------------------------------------------------
faq_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3, "filter": {"category": "hospital_procedures"}},
)

faq_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là tư vấn viên hỗ trợ thủ tục bệnh viện. Hãy dùng thông tin sau"
        " đây để trả lời câu hỏi:\n\n{context}\n\nNếu thông tin không có trong"
        " tài liệu, hãy thành thật báo không có và hướng dẫn liên hệ trực tiếp"
        " hotline bệnh viện.",
    ),
    ("human", "{question}"),
])

faq_chain = (
    {
        "context": faq_retriever
        | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | faq_prompt
    | llm
    | StrOutputParser()
)


def answer_hospital_faq(question: str) -> str:
  return faq_chain.invoke(question)


# -------------------------------------------------------------
# 2. RAG TRA CỨU TRIỆU CHỨNG & GỢI Ý CHUYÊN KHOA
# -------------------------------------------------------------
symptom_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3, "filter": {"category": "symptoms"}},
)


symptom_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Bạn là trợ lý AI hỗ trợ cung cấp thông tin sức khỏe
dựa trên tài liệu tham khảo của bệnh viện.

TÀI LIỆU TRUY XUẤT TỪ VECTORDB:
{context}

TRIỆU CHỨNG ĐÃ TRÍCH XUẤT:
{symptoms}

YÊU CẦU:
1. Tóm tắt các triệu chứng người dùng đã cung cấp.
2. Dựa trên tài liệu truy xuất để nêu những nhóm bệnh lý hoặc
   nguyên nhân có thể liên quan, nếu tài liệu có căn cứ.
3. Giải thích ngắn gọn căn cứ của nhận định.
4. Đề xuất chuyên khoa tham khảo nếu tài liệu và triệu chứng
   có đủ thông tin hỗ trợ.
5. Nếu tài liệu không đủ căn cứ, hãy nói rõ thay vì tự suy đoán.
6. Không khẳng định người dùng mắc một bệnh cụ thể.
7. Nếu có dấu hiệu nguy hiểm, ưu tiên hướng dẫn tìm trợ giúp
   y tế khẩn cấp.

CÁCH TRẢ LỜI:
Diễn đạt tự nhiên, ngắn gọn, dễ hiểu bằng tiếng Việt.
Không trình bày như báo cáo nội bộ.

Có thể sử dụng cấu trúc:
- Triệu chứng bạn mô tả
- Nhận định lâm sàng tham khảo
- Chuyên khoa nên cân nhắc

Chỉ hỏi thêm thông tin nếu thực sự cần thiết."""
    ),
    ("human", "{query}"),
])



def triage_symptom_rag(query: str, symptoms: List[str]) -> str:
  # Lấy tài liệu căn cứ triệu chứng
  docs = symptom_retriever.invoke(query)
  context_text = (
      "\n\n".join(d.page_content for d in docs)
      if docs
      else "Không có tài liệu phù hợp."
  )

  # Tạo chain xử lý
  chain = symptom_prompt | llm | StrOutputParser()
  return chain.invoke({
      "context": context_text,
      "symptoms": ", ".join(symptoms) if symptoms else "Chưa ghi nhận rõ",
      "query": query,
  })