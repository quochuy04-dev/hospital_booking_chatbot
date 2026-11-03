import os
import random
import re
import string
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv
import numpy as np
import pymysql

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import math

    class BM25Okapi:
        """Fallback BM25 implementation nếu môi trường chưa cài đặt rank-bm25."""
        def __init__(self, corpus, k1=1.5, b=0.75):
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / max(1, self.corpus_size)
            self.doc_len = [len(doc) for doc in corpus]
            nd = {}
            for doc in corpus:
                for word in set(doc):
                    nd[word] = nd.get(word, 0) + 1
            self.idf = {}
            for word, freq in nd.items():
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        def get_scores(self, query):
            score = np.zeros(self.corpus_size)
            for q in query:
                q_idf = self.idf.get(q, 0.0)
                if q_idf <= 0:
                    continue
                for i, doc in enumerate(self.corpus):
                    q_freq = doc.count(q)
                    if q_freq > 0:
                        denom = q_freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(1e-5, self.avgdl))
                        score[i] += q_idf * (q_freq * (self.k1 + 1)) / denom
            return score

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

# Xác định thư mục gốc dự án và nạp biến môi trường
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(find_dotenv(str(BASE_DIR / ".env")))

# -------------------------------------------------------------
# CẤU HÌNH KẾT NỐI DATABASE MYSQL
# -------------------------------------------------------------
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "HUY0406")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "hospital")


def get_mysql_connection():
    """Tạo kết nối tới cơ sở dữ liệu MySQL."""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


# -------------------------------------------------------------
# KHỞI TẠO VECTOR STORE (CHROMADB)
# -------------------------------------------------------------
persist_directory = str(BASE_DIR / "chroma_db")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)


# =============================================================
# NHÓM 1: CÁC CÔNG CỤ RAG TRA CỨU TRI THỨC (CHROMADB)
# =============================================================

@tool
def search_hospital_faq(question: str) -> str:
    """Tra cứu thông tin chính thức về quy trình khám bệnh, thủ tục giấy tờ, viện phí, 
    chế độ Bảo hiểm Y tế (BHYT), giờ làm việc và quy định của bệnh viện từ tài liệu PDF."""
    try:
        docs = vector_store.similarity_search(
            query=question,
            k=2,
            filter={"category": "hospital_procedures"},
        )
        if not docs:
            return "Không tìm thấy thông tin phù hợp trong tài liệu quy trình bệnh viện."

        content_list = []
        for i, doc in enumerate(docs, 1):
            text = doc.page_content.strip()
            if len(text) > 350:
                text = text[:350].rsplit(" ", 1)[0] + "..."
            content_list.append(f"【Trích dẫn {i}】\n{text}")

        return "\n\n".join(content_list)
    except Exception as e:
        return f"Lỗi khi tra cứu tài liệu quy trình: {str(e)}"


# -------------------------------------------------------------
# BẢNG ÁNH XẠ TOÀN DIỆN 100 BỆNH LÝ YOUMED VÀO 6 KHOA CỦA BỆNH VIỆN
# -------------------------------------------------------------
DISEASE_TO_DEPT = {
    # 1. Khoa Khám bệnh tổng quát (Nội tổng quát, Sốt nhiễm trùng, Suy nhược, Đa cơ quan)
    'Sốt xuất huyết': 'Khoa Khám bệnh tổng quát',
    'Bệnh bạch hầu': 'Khoa Khám bệnh tổng quát',
    'Tay chân miệng': 'Khoa Khám bệnh tổng quát',
    'Suy nhược cơ thể': 'Khoa Khám bệnh tổng quát',
    'Thiếu máu thiếu sắt': 'Khoa Khám bệnh tổng quát',
    'Béo phì': 'Khoa Khám bệnh tổng quát',
    'Suy dinh dưỡng': 'Khoa Khám bệnh tổng quát',
    'Dậy thì sớm': 'Khoa Khám bệnh tổng quát',

    # 2. Khoa Tim mạch
    'Bệnh mạch vành': 'Khoa Tim mạch',
    'Đau thắt ngực': 'Khoa Tim mạch',
    'Hạ huyết áp': 'Khoa Tim mạch',
    'Hở van tim': 'Khoa Tim mạch',
    'Nhồi máu cơ tim': 'Khoa Tim mạch',
    'Rối loạn lipid máu': 'Khoa Tim mạch',
    'Suy tim': 'Khoa Tim mạch',
    'Hạ canxi máu': 'Khoa Tim mạch',
    'Hạ đường huyết': 'Khoa Tim mạch',
    'Đái tháo đường': 'Khoa Tim mạch',
    'Đái tháo đường type 2': 'Khoa Tim mạch',
    'Basedow': 'Khoa Tim mạch',
    'Cường giáp': 'Khoa Tim mạch',
    'Suy giáp': 'Khoa Tim mạch',
    'Bướu cổ': 'Khoa Tim mạch',

    # 3. Khoa Thần kinh
    'Rối loạn tiền đình': 'Khoa Thần kinh',
    'Chóng mặt': 'Khoa Thần kinh',
    'Đau đầu dạng căng thẳng': 'Khoa Thần kinh',
    'Bệnh Parkinson': 'Khoa Thần kinh',
    'Alzheimer': 'Khoa Thần kinh',
    'Đột quỵ': 'Khoa Thần kinh',
    'Chấn thương sọ não': 'Khoa Thần kinh',
    'Đau dây thần kinh liên sườn': 'Khoa Thần kinh',
    'Đau dây thần kinh tọa': 'Khoa Thần kinh',
    'Động kinh': 'Khoa Thần kinh',
    'Khó ngủ': 'Khoa Thần kinh',
    'Mất ngủ': 'Khoa Thần kinh',
    'Rối loạn giấc ngủ': 'Khoa Thần kinh',
    'Gai cột sống': 'Khoa Thần kinh',
    'Thoái hóa cột sống cổ': 'Khoa Thần kinh',
    'Thoát vị đĩa đệm': 'Khoa Thần kinh',
    'Đau khớp vai': 'Khoa Thần kinh',
    'Thoái hóa khớp gối': 'Khoa Thần kinh',
    'Loãng xương': 'Khoa Thần kinh',
    'Gout': 'Khoa Thần kinh',
    'Đứt dây chằng chéo trước': 'Khoa Thần kinh',

    # 4. Khoa Tai Mũi Họng
    'Chảy máu cam (chảy máu mũi)': 'Khoa Tai Mũi Họng',
    'Đau họng': 'Khoa Tai Mũi Họng',
    'Polyp mũi': 'Khoa Tai Mũi Họng',
    'Ngủ ngáy': 'Khoa Tai Mũi Họng',
    'Dị ứng thời tiết': 'Khoa Tai Mũi Họng',
    'Quai bị': 'Khoa Tai Mũi Họng',
    'Đau răng khôn': 'Khoa Tai Mũi Họng',
    'Sâu răng': 'Khoa Tai Mũi Họng',
    'Cận thị': 'Khoa Tai Mũi Họng',
    'Khô mắt': 'Khoa Tai Mũi Họng',
    'Lẹo mắt': 'Khoa Tai Mũi Họng',
    'Mỏi mắt': 'Khoa Tai Mũi Họng',

    # 5. Khoa Tiêu hóa & Gan mật
    'Trào ngược dạ dày thực quản': 'Khoa Tiêu hóa & Gan mật',
    'Gan nhiễm mỡ không do rượu': 'Khoa Tiêu hóa & Gan mật',
    'Hội chứng ruột kích thích': 'Khoa Tiêu hóa & Gan mật',
    'Ngộ độc thực phẩm': 'Khoa Tiêu hóa & Gan mật',
    'Nhiễm trùng đường ruột': 'Khoa Tiêu hóa & Gan mật',
    'Nhiệt miệng': 'Khoa Tiêu hóa & Gan mật',
    'Suy gan': 'Khoa Tiêu hóa & Gan mật',
    'Sỏi túi mật': 'Khoa Tiêu hóa & Gan mật',
    'Táo bón': 'Khoa Tiêu hóa & Gan mật',
    'Tiêu chảy': 'Khoa Tiêu hóa & Gan mật',
    'Bệnh thận mạn': 'Khoa Tiêu hóa & Gan mật',
    'Suy thận': 'Khoa Tiêu hóa & Gan mật',
    'Hội chứng thận hư': 'Khoa Tiêu hóa & Gan mật',
    'Nhiễm trùng tiểu': 'Khoa Tiêu hóa & Gan mật',
    'Đái rắt': 'Khoa Tiêu hóa & Gan mật',
    'Buồng trứng đa nang': 'Khoa Tiêu hóa & Gan mật',
    'Đau bụng kinh': 'Khoa Tiêu hóa & Gan mật',
    'Lạc nội mạc tử cung': 'Khoa Tiêu hóa & Gan mật',
    'Hẹp bao quy đầu': 'Khoa Tiêu hóa & Gan mật',
    'Phì đại tuyến tiền liệt': 'Khoa Tiêu hóa & Gan mật',
    'Rối loạn cương dương': 'Khoa Tiêu hóa & Gan mật',
    'Giang mai': 'Khoa Tiêu hóa & Gan mật',
    'Lậu': 'Khoa Tiêu hóa & Gan mật',
    'Nấm âm đạo': 'Khoa Tiêu hóa & Gan mật',
    'Viêm da dị ứng': 'Khoa Tiêu hóa & Gan mật',
    'Chàm da (Eczema)': 'Khoa Tiêu hóa & Gan mật',
    'Da nhiễm corticoid': 'Khoa Tiêu hóa & Gan mật',
    'Ghẻ': 'Khoa Tiêu hóa & Gan mật',
    'Hắc lào': 'Khoa Tiêu hóa & Gan mật',
    'Lang ben': 'Khoa Tiêu hóa & Gan mật',
    'Lupus ban đỏ': 'Khoa Tiêu hóa & Gan mật',
    'Mụn trứng cá': 'Khoa Tiêu hóa & Gan mật',
    'Nấm da đầu': 'Khoa Tiêu hóa & Gan mật',
    'Nấm móng': 'Khoa Tiêu hóa & Gan mật',
    'Nổi mề đay': 'Khoa Tiêu hóa & Gan mật',

    # 6. Khoa Hô hấp & Phổi
    'Áp-xe phổi': 'Khoa Hô hấp & Phổi',
    'Cảm cúm': 'Khoa Hô hấp & Phổi',
    'Cảm lạnh': 'Khoa Hô hấp & Phổi',
    'Cúm mùa': 'Khoa Hô hấp & Phổi',
    'Hen suyễn': 'Khoa Hô hấp & Phổi',
    'Ho gà': 'Khoa Hô hấp & Phổi',
    'Ho kéo dài': 'Khoa Hô hấp & Phổi',
    'Nhiễm khuẩn đường hô hấp trên': 'Khoa Hô hấp & Phổi',
    'Phổi tắc nghẽn mạn tính': 'Khoa Hô hấp & Phổi'
}

# Tập stop words hội thoại tiếng Việt nhằm trích xuất triệu chứng tinh khiết
CONVERSATIONAL_STOPWORDS = {
    'tôi', 'em', 'mình', 'bác', 'sĩ', 'cho', 'hỏi', 'dạo', 'này', 'mấy', 'hôm', 'nay',
    'bị', 'cứ', 'hay', 'rất', 'quá', 'nhiều', 'không', 'biết', 'có', 'phải', 'là',
    'gì', 'khám', 'khoa', 'nào', 'với', 'ạ', 'dạ', 'xin', 'chào', 'tư', 'vấn', 'giúp'
}

_symptoms_bm25 = None
_symptoms_docs = None
_symptoms_metas = None
_symptoms_doc_map = None
_symptoms_tokenized = None


def _init_symptoms_bm25():
    """Khởi tạo một lần (lazy-load) kho từ vựng và chỉ mục BM25 cho 100 bài viết YouMed."""
    global _symptoms_bm25, _symptoms_docs, _symptoms_metas, _symptoms_doc_map, _symptoms_tokenized
    if _symptoms_bm25 is None:
        all_data = vector_store.get(where={"category": "symptoms"}, include=["documents", "metadatas"])
        _symptoms_docs = all_data["documents"]
        _symptoms_metas = all_data["metadatas"]
        _symptoms_doc_map = {doc: i for i, doc in enumerate(_symptoms_docs)}
        _symptoms_tokenized = [re.findall(r"\w+", d.lower()) for d in _symptoms_docs]
        _symptoms_bm25 = BM25Okapi(_symptoms_tokenized)


def clean_symptom_query(query: str) -> str:
    """Loại bỏ từ nối xã giao để giữ lại từ khóa triệu chứng lâm sàng quan trọng."""
    words = re.findall(r"\w+", query.lower())
    clean_words = [w for w in words if w not in CONVERSATIONAL_STOPWORDS]
    clean_str = " ".join(clean_words)
    return clean_str if clean_str.strip() else query


@tool
def search_symptoms_and_triage(symptoms: str) -> str:
    """Tra cứu triệu chứng sức khỏe sử dụng cơ chế Hybrid Retrieval (kết hợp BM25 chính xác từ khóa 
    và Chroma Dense Embedding ngữ nghĩa) từ 100 bài viết y khoa thực tế của YouMed do Bác sĩ biên soạn.
    Cung cấp nhận định lâm sàng phân biệt (Differential Triage), trích dẫn tác giả Bác sĩ và đường dẫn bài viết, 
    đồng thời gợi ý chính xác Chuyên khoa tương ứng tại bệnh viện."""
    try:
        _init_symptoms_bm25()
        cleaned_query = clean_symptom_query(symptoms)
        q_tokens = re.findall(r"\w+", cleaned_query.lower())
        unique_q_tokens = set(q_tokens)

        # 1. Điểm BM25 (từ khóa chính xác về mặt lâm sàng)
        bm25_raw = np.array(_symptoms_bm25.get_scores(q_tokens))
        max_bm25 = bm25_raw.max() if len(bm25_raw) > 0 and bm25_raw.max() > 0 else 1.0
        bm25_norm = bm25_raw / max_bm25

        # 2. Điểm Dense Semantic Search (ngữ nghĩa)
        dense_results = vector_store.similarity_search_with_score(
            cleaned_query,
            k=40,
            filter={"category": "symptoms"},
        )

        scores = np.zeros(len(_symptoms_docs))
        scores += bm25_norm * 0.60

        for doc, dist in dense_results:
            idx = _symptoms_doc_map.get(doc.page_content)
            if idx is not None:
                sim = max(0.0, 1.0 - (dist / 2.0))
                scores[idx] += sim * 0.40

        # 3. Tỷ lệ khớp từ khóa & Ưu tiên trích đoạn có triệu chứng/dấu hiệu
        for i, doc_tokens in enumerate(_symptoms_tokenized):
            doc_token_set = set(doc_tokens)
            matched_tokens = unique_q_tokens.intersection(doc_token_set)
            coverage = len(matched_tokens) / max(1, len(unique_q_tokens))

            # Phạt nếu chỉ khớp 1 từ chung chung mà bỏ qua các từ khóa triệu chứng còn lại
            if len(unique_q_tokens) >= 3 and len(matched_tokens) < 2:
                scores[i] *= 0.35
            else:
                scores[i] *= (0.7 + 0.5 * coverage)

            doc_text = _symptoms_docs[i].lower()
            if any(h in doc_text for h in ["triệu chứng", "dấu hiệu", "biểu hiện lâm sàng", "cảnh báo"]):
                scores[i] *= 1.25

        ranked_indices = np.argsort(-scores)

        # 4. Gom nhóm theo từng bệnh lý riêng biệt (Diversity / Differential Triage)
        disease_groups = {}
        for idx in ranked_indices:
            d_name = _symptoms_metas[idx].get("disease", "Bệnh lý chung")
            if d_name not in disease_groups:
                disease_groups[d_name] = []
            disease_groups[d_name].append((scores[idx], _symptoms_metas[idx], _symptoms_docs[idx]))

        entries = []
        disease_candidates = []
        count = 0
        for d_name, chunk_list in disease_groups.items():
            top_score = chunk_list[0][0]
            if top_score < 0.15 and len(entries) > 0:
                continue

            count += 1
            # Chọn chunk mô tả triệu chứng rõ nhất của bệnh lý đó
            best_chunk = None
            for s, m, t in chunk_list:
                t_lower = t.lower()
                if any(k in t_lower for k in ["triệu chứng", "dấu hiệu", "biểu hiện", "nhận biết"]):
                    best_chunk = (s, m, t)
                    break
            if not best_chunk:
                best_chunk = chunk_list[0]

            score, meta, content = best_chunk
            dept = DISEASE_TO_DEPT.get(d_name, "Khoa Khám bệnh tổng quát")
            title = meta.get("title", "")
            author = meta.get("author", "Bác sĩ chuyên khoa")
            url = meta.get("source_url", "https://youmed.vn")

            disease_candidates.append({
                "rank": count,
                "disease": d_name,
                "dept": dept,
                "score": score
            })

            # Rút gọn trích đoạn lâm sàng tối đa 220 ký tự để tiết kiệm token
            clean_excerpt = content.strip()
            if len(clean_excerpt) > 220:
                clean_excerpt = clean_excerpt[:220].rsplit(" ", 1)[0] + "..."

            entry = (
                f"【{count}. {d_name}】(Khoa: {dept})\n"
                f"- Tác giả: {author} | Bài: {title}\n"
                f"- Dấu hiệu: {clean_excerpt}\n"
                f"- Link: {url}"
            )
            entries.append(entry)
            if count >= 3:
                break

        if not entries:
            return "Không tìm thấy bài viết y khoa phù hợp."

        primary_dept = disease_candidates[0]["dept"]
        primary_disease = disease_candidates[0]["disease"]

        conclusion = (
            f"\n\n🎯 PHÂN LUỒNG: ƯU TIÊN SỐ 1 là '{primary_dept}' (khớp nhất với '{primary_disease}'). "
            f"Hãy chốt rõ chuyên khoa này để người bệnh không phân vân và mời đặt lịch."
        )
        return "\n\n".join(entries) + conclusion
    except Exception as e:
        return f"Lỗi khi tra cứu triệu chứng YouMed: {str(e)}"


# =============================================================
# NHÓM 2: CÁC CÔNG CỤ THAO TÁC CƠ SỞ DỮ LIỆU MYSQL (TRANSACTIONAL)
# =============================================================

@tool
def list_departments_and_doctors(department_name: Optional[str] = None) -> str:
    """Tra cứu danh sách các chuyên khoa hiện có tại bệnh viện, hoặc danh sách bác sĩ thuộc một chuyên khoa nhất định.
    
    Tham số:
    - department_name: Tên hoặc từ khóa chuyên khoa (ví dụ: 'Tim mạch', 'Thần kinh', 'Tai mũi họng', 'Hô hấp', 'Tiêu hóa'...). 
      Nếu không truyền hoặc để trống, công cụ sẽ trả về danh sách tất cả các chuyên khoa.
    """
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            if not department_name or department_name.strip() == "":
                cursor.execute("SELECT id, code, name, description FROM departments;")
                departments = cursor.fetchall()
                if not departments:
                    return "Hiện tại chưa có chuyên khoa nào trong cơ sở dữ liệu."

                lines = ["Danh sách các chuyên khoa tại bệnh viện:"]
                for d in departments:
                    lines.append(f"- [{d['id']}] {d['name']} ({d['code']}): {d['description']}")
                return "\n".join(lines)
            else:
                clean_kw = department_name.strip()
                if "tổng quát" in clean_kw.lower():
                    keyword = "%tổng quát%"
                else:
                    keyword = f"%{clean_kw}%"

                sql = """
                    SELECT d.id AS doc_id, d.full_name, d.degree, d.room_number, d.bio, dept.name AS dept_name
                    FROM doctors d
                    JOIN departments dept ON d.department_id = dept.id
                    WHERE dept.name LIKE %s OR dept.code LIKE %s;
                """
                cursor.execute(sql, (keyword, keyword))
                doctors = cursor.fetchall()
                if not doctors:
                    return f"Không tìm thấy bác sĩ nào thuộc chuyên khoa có từ khóa '{department_name}'."

                lines = [f"Danh sách bác sĩ thuộc {doctors[0]['dept_name']}:"]
                for doc in doctors:
                    lines.append(
                        f"• ID {doc['doc_id']}: {doc['degree']} {doc['full_name']}\n"
                        f"  Phòng khám: {doc['room_number']}\n"
                        f"  Chuyên môn: {doc['bio']}"
                    )
                return "\n\n".join(lines)
    except Exception as e:
        return f"Lỗi khi truy vấn thông tin chuyên khoa/bác sĩ: {str(e)}"
    finally:
        conn.close()


@tool
def check_doctor_available_slots(doctor_name_or_id: str, booking_date: Optional[str] = None) -> str:
    """Kiểm tra các khung giờ khám bệnh còn trống của một bác sĩ.
    
    Tham số:
    - doctor_name_or_id: Tên bác sĩ (ví dụ: 'Đỗ Quốc Cường', 'Trần Văn An', 'bác sĩ Cường', 'bác sĩ Mai'...) hoặc ID bác sĩ (ví dụ: '1').
    - booking_date: Ngày muốn khám theo định dạng 'YYYY-MM-DD' (ví dụ: '2026-11-04'). 
      Nếu không truyền hoặc để trống (None), công cụ sẽ TỰ ĐỘNG liệt kê tất cả các khung giờ còn trống trong 7 ngày tới của bác sĩ đó!
    """
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            # 1. Chuẩn hóa tên bác sĩ để tìm kiếm thông minh
            raw_input = doctor_name_or_id.strip()
            clean_name = raw_input
            for prefix in ["pgs.ts.bs", "bs.ckii", "bs.cki", "ts.bs", "ths.bs", "bác sĩ", "bac si", "bs", "ts", "ths"]:
                if clean_name.lower().startswith(prefix):
                    clean_name = clean_name[len(prefix):].strip()
                    break

            if raw_input.isdigit():
                sql_doc = """
                    SELECT d.id, d.full_name, d.degree, dept.name AS dept_name
                    FROM doctors d
                    JOIN departments dept ON d.department_id = dept.id
                    WHERE d.id = %s;
                """
                cursor.execute(sql_doc, (int(raw_input),))
            else:
                sql_doc = """
                    SELECT d.id, d.full_name, d.degree, dept.name AS dept_name
                    FROM doctors d
                    JOIN departments dept ON d.department_id = dept.id
                    WHERE d.full_name LIKE %s OR d.full_name LIKE %s
                    LIMIT 1;
                """
                cursor.execute(sql_doc, (f"%{clean_name}%", f"%{raw_input}%"))

            doctor = cursor.fetchone()
            if not doctor:
                return f"Không tìm thấy bác sĩ '{doctor_name_or_id}' trong hệ thống. Bạn có thể tra cứu danh sách bác sĩ bằng tool list_departments_and_doctors."

            doc_id = doctor["id"]
            doc_name = f"{doctor['degree']} {doctor['full_name']}"
            dept_name = doctor["dept_name"]

            # 2. Lấy các khung giờ khám còn trống
            if booking_date and booking_date.strip():
                sql_sched = """
                    SELECT id AS schedule_id, work_date, time_slot, max_patients, current_patients
                    FROM schedules
                    WHERE doctor_id = %s AND work_date = %s AND is_available = 1 AND current_patients < max_patients
                    ORDER BY time_slot ASC;
                """
                cursor.execute(sql_sched, (doc_id, booking_date.strip()))
            else:
                # Nếu không truyền ngày, lấy các khung giờ trống của 2-3 ngày gần nhất (tối ưu token)
                sql_sched = """
                    SELECT id AS schedule_id, work_date, time_slot, max_patients, current_patients
                    FROM schedules
                    WHERE doctor_id = %s AND work_date >= CURDATE() AND is_available = 1 AND current_patients < max_patients
                    ORDER BY work_date ASC, time_slot ASC
                    LIMIT 8;
                """
                cursor.execute(sql_sched, (doc_id,))

            slots = cursor.fetchall()
            if not slots:
                if booking_date:
                    return f"Rất tiếc, {doc_name} ({dept_name}) không còn khung giờ trống vào ngày {booking_date}. Bạn có thể chọn ngày khác hoặc bác sĩ khác cùng khoa."
                else:
                    return f"Hiện tại {doc_name} ({dept_name}) chưa có lịch trống trong các ngày tới."

            # Gom nhóm theo ngày
            slots_by_date = {}
            for s in slots:
                d_str = str(s["work_date"])
                slots_by_date.setdefault(d_str, []).append(s)

            weekday_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
            lines = [f"Lịch khám gần nhất của {doc_name} ({dept_name}):"]

            for d_str, s_list in slots_by_date.items():
                d_obj = datetime.strptime(d_str, "%Y-%m-%d")
                weekday_vn = weekday_names[d_obj.weekday()]
                slot_strs = [f"[Mã {s['schedule_id']}] {s['time_slot']}" for s in s_list]
                lines.append(f"• Ngày {d_str} ({weekday_vn}): {', '.join(slot_strs)}")

            return "\n".join(lines)
    except Exception as e:
        return f"Lỗi khi tra cứu lịch khám: {str(e)}"
    finally:
        conn.close()


@tool
def book_appointment(
    patient_name: str,
    phone_number: str,
    schedule_id: int,
    symptoms: str = "",
    date_of_birth: str = "",
    gender: str = "Khác",
) -> str:
    """Thực hiện lưu phiếu đặt lịch khám bệnh chính thức vào cơ sở dữ liệu MySQL và cấp mã phiếu hẹn.
    
    Tham số:
    - patient_name: Họ tên đầy đủ của bệnh nhân (ví dụ: 'Nguyễn Văn Nam').
    - phone_number: Số điện thoại liên hệ (ví dụ: '0987654321').
    - schedule_id: Mã lịch khám (schedule_id) do người dùng chọn từ kết quả của check_doctor_available_slots.
    - symptoms: Tóm tắt triệu chứng hoặc lý do khám bệnh (nếu có).
    - date_of_birth: Năm sinh hoặc ngày sinh (ví dụ: '1995').
    - gender: Giới tính ('Nam', 'Nữ' hoặc 'Khác').
    """
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            # 1. Kiểm tra khung giờ khám có hợp lệ và còn chỗ không
            sql_check = """
                SELECT s.id, s.work_date, s.time_slot, s.max_patients, s.current_patients, s.is_available,
                       d.id AS doc_id, d.full_name AS doc_name, d.degree AS doc_degree, d.room_number,
                       dept.name AS dept_name
                FROM schedules s
                JOIN doctors d ON s.doctor_id = d.id
                JOIN departments dept ON d.department_id = dept.id
                WHERE s.id = %s;
            """
            cursor.execute(sql_check, (schedule_id,))
            sched = cursor.fetchone()

            if not sched:
                return f"Lỗi: Không tìm thấy khung giờ khám có mã schedule_id = {schedule_id}."

            if not sched["is_available"] or sched["current_patients"] >= sched["max_patients"]:
                return f"Rất tiếc: Khung giờ {sched['time_slot']} ngày {sched['work_date']} vừa hết chỗ. Vui lòng chọn khung giờ khác."

            # 2. Tìm hoặc tạo mới hồ sơ bệnh nhân
            sql_patient_find = "SELECT id FROM patients WHERE phone_number = %s;"
            cursor.execute(sql_patient_find, (phone_number.strip(),))
            p_record = cursor.fetchone()

            if p_record:
                patient_id = p_record["id"]
            else:
                sql_patient_create = """
                    INSERT INTO patients (full_name, phone_number, date_of_birth, gender)
                    VALUES (%s, %s, %s, %s);
                """
                cursor.execute(sql_patient_create, (patient_name.strip(), phone_number.strip(), date_of_birth.strip(), gender.strip()))
                patient_id = cursor.lastrowid

            # 3. Tạo mã phiếu khám ngẫu nhiên duy nhất (ví dụ: KB-20261104-A1B2)
            date_prefix = datetime.now().strftime("%Y%m%d")
            rand_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
            appointment_code = f"KB-{date_prefix}-{rand_suffix}"

            # 4. Lưu phiếu đặt lịch vào bảng appointments
            sql_appoint = """
                INSERT INTO appointments (appointment_code, patient_id, doctor_id, schedule_id, symptoms_summary, status)
                VALUES (%s, %s, %s, %s, %s, 'CONFIRMED');
            """
            cursor.execute(sql_appoint, (appointment_code, patient_id, sched["doc_id"], sched["id"], symptoms.strip()))

            # 5. Cập nhật số bệnh nhân đã đặt vào schedules
            new_curr = sched["current_patients"] + 1
            is_avail = 1 if new_curr < sched["max_patients"] else 0
            sql_update_sched = "UPDATE schedules SET current_patients = %s, is_available = %s WHERE id = %s;"
            cursor.execute(sql_update_sched, (new_curr, is_avail, schedule_id))

            # 6. Trả về thông tin xác nhận đặt lịch thành công
            confirmation = (
                f"🎉 ĐẶT LỊCH KHÁM BỆNH THÀNH CÔNG!\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• Mã phiếu hẹn: {appointment_code}\n"
                f"• Bệnh nhân: {patient_name.strip()} (SĐT: {phone_number.strip()})\n"
                f"• Bác sĩ: {sched['doc_degree']} {sched['doc_name']}\n"
                f"• Chuyên khoa: {sched['dept_name']}\n"
                f"• Địa điểm: {sched['room_number']}\n"
                f"• Ngày khám: {sched['work_date']}\n"
                f"• Khung giờ: {sched['time_slot']}\n"
                f"• Triệu chứng ghi nhận: {symptoms.strip() if symptoms else 'Không có'}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Quý khách vui lòng đến trước 15 phút tại quầy tiếp đón và đọc Mã phiếu hẹn để được hướng dẫn vào khám."
            )
            return confirmation
    except Exception as e:
        return f"Lỗi trong quá trình đặt lịch: {str(e)}"
    finally:
        conn.close()


@tool
def lookup_appointment(phone_number: str) -> str:
    """Tra cứu danh sách các lịch khám bệnh đã đặt của bệnh nhân bằng số điện thoại.
    
    Tham số:
    - phone_number: Số điện thoại của bệnh nhân cần tra cứu (ví dụ: '0987654321').
    """
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            sql = """
                SELECT a.appointment_code, a.status, a.symptoms_summary, a.created_at,
                       p.full_name AS patient_name,
                       d.full_name AS doc_name, d.degree AS doc_degree, d.room_number,
                       dept.name AS dept_name,
                       s.work_date, s.time_slot
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                JOIN doctors d ON a.doctor_id = d.id
                JOIN departments dept ON d.department_id = dept.id
                JOIN schedules s ON a.schedule_id = s.id
                WHERE p.phone_number = %s
                ORDER BY a.created_at DESC;
            """
            cursor.execute(sql, (phone_number.strip(),))
            appointments = cursor.fetchall()

            if not appointments:
                return f"Không tìm thấy phiếu hẹn nào được đăng ký bằng số điện thoại '{phone_number}'."

            lines = [f"Danh sách lịch hẹn của số điện thoại {phone_number}:"]
            for i, ap in enumerate(appointments, 1):
                status_vn = "Đã xác nhận" if ap["status"] == "CONFIRMED" else "Đã hủy"
                lines.append(
                    f"【Phiếu {i}: {ap['appointment_code']} - Trạng thái: {status_vn}】\n"
                    f"• Bệnh nhân: {ap['patient_name']}\n"
                    f"• Bác sĩ: {ap['doc_degree']} {ap['doc_name']} ({ap['dept_name']})\n"
                    f"• Phòng khám: {ap['room_number']}\n"
                    f"• Ngày khám: {ap['work_date']} | Khung giờ: {ap['time_slot']}\n"
                    f"• Triệu chứng: {ap['symptoms_summary'] if ap['symptoms_summary'] else 'Không ghi chú'}"
                )

            return "\n\n".join(lines)
    except Exception as e:
        return f"Lỗi khi tra cứu phiếu hẹn: {str(e)}"
    finally:
        conn.close()


# Danh sách toàn bộ 6 tools dùng để truyền vào LangGraph ReAct Agent
ALL_TOOLS = [
    search_hospital_faq,
    search_symptoms_and_triage,
    list_departments_and_doctors,
    check_doctor_available_slots,
    book_appointment,
    lookup_appointment,
]
