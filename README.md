# 🏥 Hospital Booking & Triage Chatbot (Trợ lý Y tế & Phân luồng Khám bệnh)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?style=flat&logo=openai)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-red?style=flat)

Hệ thống Chatbot Y tế thông minh được xây dựng trên kiến trúc **Router + RAG (Retrieval-Augmented Generation)** và **LCEL (LangChain Expression Language)**. Chatbot đóng vai trò tiếp nhận ban đầu, giải đáp thủ tục khám bệnh và phân tích triệu chứng để gợi ý chuyên khoa phù hợp trước khi đặt lịch khám.

---

## 📌 Các Tính Năng Nổi Bật

1. **Ý định phân luồng thông minh (Intent Triage & Router):**
   - Sử dụng **Pydantic Structured Output** (`gpt-4o-mini`) để phân loại chính xác 4 nhóm ý định:
     - `CHAO_HOI_CAM_ON`: Phản hồi xã giao thân thiện, tự nhiên.
     - `QUY_TRINH_KHAM`: Giải đáp thủ tục, viện phí, BHYT, giờ khám.
     - `TRIEU_CHUNG_LAM_SANG`: Tiếp nhận triệu chứng, gợi ý chuyên khoa tham khảo.
     - `NGOAI_PHAM_VI`: Lịch sự từ chối các câu hỏi nằm ngoài phạm vi bệnh viện.
2. **Metadata-Filtered RAG:**
   - Phân đoạn (Chunking) tài liệu có gắn metadata category: `hospital_procedures` (quy trình) và `symptoms` (triệu chứng lâm sàng).
   - Truy vấn mục tiêu theo metadata filter, loại bỏ nhiễu ngữ cảnh giữa thủ tục hành chính và tài liệu y khoa.
3. **Tái tạo ngữ cảnh hội thoại đa lượt (Query Contextualization):**
   - Viết lại câu hỏi độc lập (Standalone Query) dựa vào lịch sử chat (Multi-turn Memory) giúp bot hiểu được các câu hỏi nối tiếp của bệnh nhân mà không gây ảo giác (hallucination).
4. **Nguyên tắc an toàn y tế (Medical Guardrails):**
   - Không tự ý kết luận bệnh cụ thể, bắt buộc đưa disclaimer y khoa và ưu tiên cảnh báo khẩn cấp (Hotline 115) nếu phát hiện triệu chứng nguy kịch.

---

## 📁 Cấu Trúc Dự Án

```text
hospital_booking_chatbot/
├── .env.example                     # File mẫu biến môi trường (API Key)
├── .gitignore                       # Cấu hình bỏ qua .env, chroma_db, venv
├── requirements.txt                 # Danh sách thư viện phụ thuộc
├── data/                            # Tài liệu PDF nguồn của bệnh viện
│   ├── rag_data_quy_trinh_benh_vien.pdf
│   └── rag_data_trieu_chung_suc_khoe.pdf
└── Script/
    ├── ingest_data.py               # Xử lý PDF, Chunking & Nhúng vào ChromaDB
    ├── rag_service.py               # Khởi tạo VectorStore, Retriever & RAG Chains
    ├── triage.py                    # Router phân loại ý định & trích xuất triệu chứng
    ├── evaluate.py                  # Module đánh giá hiệu năng
    └── main.py                      # Bộ điều phối luồng trò chuyện (CLI Entrypoint)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Yêu cầu môi trường
- Python 3.10 trở lên.
- Đã cài đặt `git`.

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Key
Tạo file `.env` từ file mẫu `.env.example`:
```bash
# Tạo file .env và dán OpenAI API Key của bạn vào:
OPENAI_API_KEY="sk-proj-your-api-key-here"
```

### 4. Nạp dữ liệu vào Vector Database (ChromaDB)
Chạy script đọc PDF và khởi tạo vector database:
```bash
python Script/ingest_data.py
```
*Lưu ý: Dữ liệu vector sẽ được lưu vào thư mục `chroma_db/` ở thư mục gốc.*

### 5. Khởi chạy Chatbot
```bash
python Script/main.py
```

---

## 💬 Minh Họa Luồng Trò Chuyện

```text
============================================================
HỆ THỐNG TRỢ LÝ Y TẾ THEO KIẾN TRÚC ROUTER - RAG
============================================================

👤 Bệnh nhân: Chào bạn
[GREETING]
🤖 Bot:
Xin chào! Tôi là Trợ lý ảo y tế của bệnh viện.
Tôi có thể hỗ trợ bạn:
1. Giải đáp quy trình, thủ tục, thời gian khám, chi phí và BHYT.
2. Tiếp nhận triệu chứng sức khỏe ban đầu để tư vấn chuyên khoa phù hợp.

👤 Bệnh nhân: Bệnh viện có khám vào thứ 7 và chủ nhật không?
[FAQ_ANSWER]
🤖 Bot:
Dạ theo quy định của bệnh viện:
- Thứ Bảy: Khám dịch vụ từ 07:00 đến 16:30.
- Chủ Nhật: Chỉ tiếp nhận cấp cứu 24/7...

👤 Bệnh nhân: Tôi bị ho có đờm và đau rát họng 3 ngày nay
[BOOKING_REDIRECT]
🤖 Bot:
- Triệu chứng bạn mô tả: Ho có đờm, đau rát họng kéo dài 3 ngày.
- Nhận định tham khảo: Có thể liên quan đến các vấn đề viêm đường hô hấp trên.
- Chuyên khoa nên cân nhắc: Khoa Tai Mũi Họng hoặc Khoa Hô Hấp.
```


