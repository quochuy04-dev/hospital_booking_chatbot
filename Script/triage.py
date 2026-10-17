
from pathlib import Path
from typing import List, Literal

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
_ = load_dotenv(find_dotenv(str(BASE_DIR / ".env")))


# 1. Cấu trúc dữ liệu đầu ra
class TriageClassification(BaseModel):
    loai_yeu_cau: Literal[
        "CHAO_HOI_CAM_ON",
        "QUY_TRINH_KHAM",
        "TRIEU_CHUNG_LAM_SANG",
        "NGOAI_PHAM_VI",
    ] = Field(
        description=(
            "Phân loại ý định của người dùng:\n"
            "- CHAO_HOI_CAM_ON: Lời chào hỏi (xin chào, hi, hello), cảm ơn, "
            "tạm biệt hoặc giao tiếp xã giao thông thường.\n"
            "- QUY_TRINH_KHAM: Hỏi thủ tục khám, giờ khám, giấy tờ, "
            "BHYT hoặc chi phí.\n"
            "- TRIEU_CHUNG_LAM_SANG: Mô tả triệu chứng sức khỏe, "
            "hỏi về nguyên nhân có thể liên quan hoặc cần hướng dẫn khám.\n"
            "- NGOAI_PHAM_VI: Câu hỏi lạc đề hoàn toàn không liên quan đến "
            "y tế hay bệnh viện (thời tiết, toán học, lập trình, thơ ca,...)."
        )
    )

    trieu_chung_ghi_nhan: List[str] = Field(
        default_factory=list,
        description=(
            "Danh sách triệu chứng người dùng trực tiếp mô tả. "
            "Không tự suy diễn triệu chứng chưa được cung cấp. "
            "Nếu không có triệu chứng thì trả về danh sách rỗng."
        ),
    )


# 2. Khởi tạo mô hình
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

structured_llm = llm.with_structured_output(
    TriageClassification
)


# 3. Prompt phân loại và trích xuất triệu chứng
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Bạn là bộ phân loại yêu cầu cho chatbot hỗ trợ bệnh nhân.

Nhiệm vụ:
1. Phân loại yêu cầu vào đúng một trong bốn nhóm:
   - CHAO_HOI_CAM_ON: Lời chào hỏi, cảm ơn, tạm biệt hoặc xã giao.
   - QUY_TRINH_KHAM: Hỏi về thủ tục, giờ khám, giấy tờ, BHYT, chi phí hoặc quy trình bệnh viện.
   - TRIEU_CHUNG_LAM_SANG: Kể triệu chứng, mô tả tình trạng sức khỏe hoặc hỏi nguyên nhân liên quan đến triệu chứng.
   - NGOAI_PHAM_VI: Câu hỏi không thuộc phạm vi bệnh viện/y tế (thời tiết, code, toán học, viết văn,...).
2. Trích xuất các triệu chứng được người dùng trực tiếp đề cập.
3. Không chẩn đoán bệnh và không tự thêm triệu chứng.

Ví dụ:
Người dùng: "Xin chào bạn!"
Phân loại: CHAO_HOI_CAM_ON
Triệu chứng: []

Người dùng: "Cảm ơn bot nhiều nhé!"
Phân loại: CHAO_HOI_CAM_ON
Triệu chứng: []

Người dùng: "Bệnh viện có khám bảo hiểm y tế không?"
Phân loại: QUY_TRINH_KHAM
Triệu chứng: []

Người dùng: "Tôi bị ho và sốt hai ngày nay."
Phân loại: TRIEU_CHUNG_LAM_SANG
Triệu chứng: ["ho", "sốt", "kéo dài hai ngày"]

Người dùng: "Tôi đau đầu và chóng mặt, có thể do gì?"
Phân loại: TRIEU_CHUNG_LAM_SANG
Triệu chứng: ["đau đầu", "chóng mặt"]

Người dùng: "Thủ đô của nước Pháp là gì?"
Phân loại: NGOAI_PHAM_VI
Triệu chứng: []

Lưu ý:
- Chỉ trích xuất thông tin có trong câu nói.
- Không coi câu hỏi về một thủ tục bệnh viện là triệu chứng.
- Nếu người dùng kể triệu chứng nhưng chưa nêu thời gian, không tự tạo thời gian mắc triệu chứng.
- Nếu câu nói vừa hỏi thủ tục vừa mô tả triệu chứng, ưu tiên TRIEU_CHUNG_LAM_SANG khi mục đích chính là hỏi về sức khỏe.
        """,
    ),
    ("human", "{user_input}"),
])


# 4. Tạo chuỗi xử lý
triage_chain = prompt | structured_llm


# 5. Hàm phân loại được gọi từ main.py
def classify_intent(user_message: str) -> TriageClassification:
    """Phân loại ý định và trích xuất triệu chứng ban đầu."""
    return triage_chain.invoke({
        "user_input": user_message
    })


