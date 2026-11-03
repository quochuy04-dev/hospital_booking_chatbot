import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from tools import ALL_TOOLS

# Nạp API Key từ .env
load_dotenv(find_dotenv(str(BASE_DIR / ".env")))


def build_system_prompt() -> str:
    """Tạo System Prompt động có chứa ngày giờ thực tế để LLM tính toán chính xác lịch khám."""
    now = datetime.now()
    weekday_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    weekday_vn = weekday_names[now.weekday()]
    today_str = now.strftime("%Y-%m-%d")

    return f"""Bạn là Trợ lý Y tế & Đặt lịch khám thông minh của Bệnh viện Đa khoa.
Hôm nay là: {weekday_vn}, ngày {today_str}.

QUY TẮC SỬ DỤNG CÔNG CỤ (TOOLS):
1. GIẢI ĐÁP QUY TRÌNH & THỦ TỤC:
   - Khi bệnh nhân hỏi về thủ tục, BHYT, viện phí, giờ khám, giấy tờ ➔ Sử dụng công cụ `search_hospital_faq`.

2. TƯ VẤN TRIỆU CHỨNG & PHÂN LUỒNG KHÁM BỆNH (QUY TẮC BẮT BUỘC):
   - Khi bệnh nhân kể triệu chứng sức khỏe ➔ Gọi ngay công cụ `search_symptoms_and_triage`.
   - TUYỆT ĐỐI KHÔNG để bệnh nhân phải tự chọn giữa nhiều chuyên khoa khác nhau! Phải trả lời theo đúng cấu trúc 4 phần rõ ràng:
     1. Khả năng bệnh lý tham khảo: Liệt kê 2-3 bệnh lý từ YouMed kèm triệu chứng phù hợp, tên Bác sĩ và link bài viết.
     2. Dấu hiệu phân biệt (Differentiating Clues):
        Chỉ ra các triệu chứng đi kèm then chốt để bệnh nhân tự nhận biết:
        (Ví dụ: "• Nếu có nôn ói nhiều, đau quặn bụng, rối loạn đại tiện ➔ Thuộc Khoa Tiêu hóa & Gan mật.
                 • Nếu sốt cao kèm ho, đau rát họng, nghẹt mũi ➔ Thuộc Khoa Hô hấp & Phổi.")
     3. CHUYÊN KHOA KHUYẾN NGHỊ ƯU TIÊN SỐ 1:
        Luôn chốt DUY NHẤT 1 chuyên khoa ưu tiên cao nhất thuộc 6 khoa có bác sĩ tại viện (Khoa Khám bệnh tổng quát, Khoa Tim mạch, Khoa Thần kinh, Khoa Tai Mũi Họng, Khoa Tiêu hóa & Gan mật, Khoa Hô hấp & Phổi) dựa trên `CHUYÊN KHOA KHUYẾN NGHỊ ƯU TIÊN SỐ 1` mà tool trả về.
        Mẫu câu bắt buộc: "👉 **Khuyến nghị chuyên khoa ưu tiên số 1:** Anh/chị nên đăng ký khám tại **[Tên Chuyên Khoa Ưu Tiên]** để được bác sĩ kiểm tra và chỉ định xét nghiệm ban đầu."
     4. Mời đặt lịch dứt khoát:
        Chủ động hỏi: "Anh/chị có muốn em hỗ trợ xem danh sách bác sĩ hoặc kiểm tra các khung giờ khám còn trống của **[Tên Chuyên Khoa Ưu Tiên]** để đặt lịch ngay không ạ?".

3. TRA CỨU CHUYÊN KHOA & BÁC SĨ:
   - Khi bệnh nhân hỏi về chuyên khoa hoặc hỏi bác sĩ thuộc một khoa ➔ Sử dụng công cụ `list_departments_and_doctors`.

4. TRA CỨU LỊCH KHÁM & GỢI Ý KHUNG GIỜ (CỰC KỲ QUAN TRỌNG):
   - Khi bệnh nhân quan tâm đến một bác sĩ, hỏi "khi nào khám được", hoặc hỏi "lịch khám của bác sĩ X":
     ➔ HÃY CHỦ ĐỘNG GỌI NGAY CÔNG CỤ `check_doctor_available_slots(doctor_name_or_id=...)` (không cần truyền booking_date nếu bệnh nhân chưa nói ngày cụ thể). Công cụ sẽ tự động liệt kê các khung giờ còn trống trong những ngày tới!
   - Nếu bệnh nhân nói "ngày mai", bạn tự hiểu là ngày tiếp theo dựa trên ngày hôm nay ({today_str}).
   - Sau khi có danh sách khung giờ trống, hãy trình bày rõ ràng (Ngày, Thứ, Khung giờ, Mã lịch) và mời bệnh nhân chọn 1 khung giờ phù hợp.

5. ĐẶT LỊCH KHÁM BỆNH VÀO MYSQL:
   - Khi bệnh nhân đã chọn được Bác sĩ/Khung giờ khám (hoặc mã lịch schedule_id):
     ➔ Hãy thu thập Họ tên và Số điện thoại liên hệ của bệnh nhân.
     ➔ Khi đã có đủ thông tin, gọi ngay công cụ `book_appointment` để lưu vào cơ sở dữ liệu.
     ➔ Đọc lại toàn bộ thông tin xác nhận và Mã phiếu hẹn (KB-...) cho bệnh nhân.

6. TRA CỨU LỊCH HẸN ĐÃ ĐẶT:
   - Hỏi số điện thoại và gọi công cụ `lookup_appointment`.

NGUYÊN TẮC GIAO TIẾP:
- Giọng điệu ân cần, lịch sự, chuyên nghiệp.
- Trả lời bằng tiếng Việt tự nhiên, ngắt dòng rõ ràng, dùng bullet points dễ đọc.
- Đưa ra lời khuyên mang tính chất tham khảo ban đầu, luôn nhắc nhở bệnh nhân đến khám trực tiếp để được chẩn đoán chính xác.
"""


# -------------------------------------------------------------
# KHỞI TẠO LLM VÀ STATE GRAPH
# -------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


def optimize_messages_for_llm(messages: list) -> list:
    """Tối ưu hóa token bằng cách rút gọn nội dung các ToolMessage cũ ở các lượt trước."""
    if not messages:
        return []

    # Tìm các ToolMessage thuộc lượt chạy hiện tại (nằm ở cuối danh sách)
    last_turn_tool_ids = set()
    for m in reversed(messages):
        if getattr(m, "type", None) == "tool":
            last_turn_tool_ids.add(getattr(m, "tool_call_id", None))
        else:
            break

    optimized = []
    for m in messages:
        if getattr(m, "type", None) == "tool":
            tool_id = getattr(m, "tool_call_id", None)
            if tool_id in last_turn_tool_ids:
                optimized.append(m)
            else:
                # Lượt cũ trong quá khứ: rút gọn nội dung xuống câu tóm tắt để tiết kiệm token
                # nhưng VẪN GIỮ NGUYÊN tool_call_id để OpenAI API không báo lỗi cú pháp
                from langchain_core.messages import ToolMessage
                compact_m = ToolMessage(
                    content="[Dữ liệu tra cứu đã được ghi nhận ở lượt trước]",
                    tool_call_id=tool_id,
                    name=getattr(m, "name", "")
                )
                optimized.append(compact_m)
        else:
            optimized.append(m)

    # Giới hạn cửa sổ trượt: nếu lịch sử quá dài (> 12 tin nhắn), chỉ giữ SystemMessage + 10 tin gần nhất
    if len(optimized) > 12:
        system_msg = optimized[0] if getattr(optimized[0], "type", None) == "system" else None
        recent_msgs = optimized[-10:]
        if getattr(recent_msgs[0], "type", None) == "tool":
            recent_msgs = optimized[-11:]
        optimized = [system_msg] + recent_msgs if system_msg else recent_msgs

    return optimized


def call_agent_node(state: MessagesState):
    """Node xử lý của LLM: Tiếp nhận ngữ cảnh hội thoại và quyết định gọi tool hay trả lời."""
    messages = state["messages"]

    # Đảm bảo System Message cập nhật ngày thực tế luôn nằm ở đầu hội thoại
    system_prompt = build_system_prompt()
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages
    else:
        # Cập nhật lại System Message
        messages = [SystemMessage(content=system_prompt)] + messages[1:]

    # Tối ưu hóa token trước khi gửi lên OpenAI
    trimmed_messages = optimize_messages_for_llm(messages)

    response = llm_with_tools.invoke(trimmed_messages)
    return {"messages": [response]}


# -------------------------------------------------------------
# BIÊN DỊCH LANGGRAPH REACT STATEGRAPH
# -------------------------------------------------------------
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_agent_node)
workflow.add_node("tools", ToolNode(ALL_TOOLS))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
hospital_agent = workflow.compile(checkpointer=checkpointer)


def chat_with_hospital_agent(user_message: str, thread_id: str = "default_user") -> str:
    """Gửi tin nhắn của bệnh nhân đến Agent và nhận câu trả lời cuối cùng."""
    config = {"configurable": {"thread_id": thread_id}}
    input_message = HumanMessage(content=user_message)

    events = hospital_agent.invoke({"messages": [input_message]}, config)
    last_message = events["messages"][-1]
    return last_message.content


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM THỬ GỢI Ý LỊCH KHÁM CHỦ ĐỘNG")
    print("=" * 60)

    # Test trường hợp bệnh nhân hỏi lịch khám của bác sĩ mà KHÔNG chỉ định ngày
    test_thread = "test_schedule_proactive"
    q = "Bác sĩ Đỗ Quốc Cường có những lịch khám nào vậy?"
    print(f"👤 Bệnh nhân: {q}")
    reply = chat_with_hospital_agent(q, thread_id=test_thread)
    print(f"🤖 Bot:\n{reply}\n")
