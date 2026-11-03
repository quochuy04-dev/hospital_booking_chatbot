import sys
import uuid
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage

# Cấu hình đường dẫn thư mục gốc và Script để VS Code và Python nhận diện module chuẩn xác
BASE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = BASE_DIR / "Script"
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from Script.agent import hospital_agent
from Script.tools import list_departments_and_doctors, lookup_appointment

# -------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="Trợ lý Y tế & Đặt lịch Bệnh viện",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS giao diện y tế hiện đại
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 1.2rem;
    }
    .ticket-box {
        background-color: #E8F5E9;
        border-left: 5px solid #2E7D32;
        padding: 12px;
        border-radius: 4px;
        margin-top: 8px;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# QUẢN LÝ SESSION STATE
# -------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web_session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Xin chào! Tôi là **Trợ lý Y tế Thông minh** của Bệnh viện Đa khoa.\n\n"
                "Tôi có thể hỗ trợ bạn:\n"
                "1. 💬 **Tư vấn triệu chứng sức khỏe ban đầu** (theo tri thức y khoa YouMed).\n"
                "2. 📋 **Giải đáp thủ tục**, chi phí khám và bảo hiểm y tế (BHYT).\n"
                "3. 👨‍⚕️ **Tra cứu bác sĩ & Xem các khung giờ khám còn trống**.\n"
                "4. 🎟️ **Đặt lịch khám bệnh trực tiếp** và cấp mã phiếu hẹn.\n\n"
                "Bạn đang cần hỗ trợ thông tin nào, hoặc có triệu chứng gì cần tư vấn không?"
            ),
        }
    ]

# -------------------------------------------------------------
# THANH SIDEBAR (TIỆN ÍCH TRA CỨU NHANH)
# -------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.freepik.com/free-vector/hospital-building-modern-parking-lot-50s-style-architecture-exterior-design-isometric-composition_1284-25752.jpg",
        use_container_width=True,
    )
    st.markdown("### 🏥 Bệnh Viện Đa Khoa")
    st.markdown("📍 **Địa chỉ:** 123 Đường Sức Khỏe, Quận 1, TP.HCM  \n📞 **Hotline Cấp cứu:** 115 | **Tổng đài:** 1900 1234")
    st.markdown("---")

    # Nút tạo phiên mới
    if st.button("🔄 Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.thread_id = f"web_session_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Cuộc trò chuyện đã được làm mới! Bạn cần hỗ trợ về triệu chứng hay muốn đặt lịch khám nào không?",
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🔍 Tiện ích Tra cứu Nhanh")

    # Tab tra cứu nhanh Bác sĩ và Phiếu khám
    sidebar_tab = st.radio("Chọn chức năng tra cứu:", ["👨‍⚕️ Danh sách Bác sĩ", "🎫 Tra cứu Lịch đã đặt"], label_visibility="collapsed")

    if sidebar_tab == "👨‍⚕️ Danh sách Bác sĩ":
        selected_dept = st.selectbox(
            "Chọn chuyên khoa:",
            ["Tất cả", "Khám tổng quát", "Tim mạch", "Thần kinh", "Tai Mũi Họng", "Tiêu hóa & Gan mật", "Hô hấp & Phổi"],
        )
        if st.button("Xem bác sĩ", use_container_width=True):
            dept_kw = None if selected_dept == "Tất cả" else selected_dept
            result = list_departments_and_doctors.invoke({"department_name": dept_kw})
            st.info(result)

    elif sidebar_tab == "🎫 Tra cứu Lịch đã đặt":
        phone_input = st.text_input("Nhập số điện thoại:", placeholder="Ví dụ: 0912345678")
        if st.button("Tra cứu phiếu hẹn", use_container_width=True):
            if phone_input.strip():
                lookup_res = lookup_appointment.invoke({"phone_number": phone_input.strip()})
                st.success(lookup_res)
            else:
                st.warning("Vui lòng nhập số điện thoại.")

    st.markdown("---")
    st.caption("⚡ Hệ thống vận hành bằng **LangGraph ReAct Agent** kết hợp **MySQL** & **RAG YouMed**.")

# -------------------------------------------------------------
# GIAO DIỆN CHAT CHÍNH
# -------------------------------------------------------------
st.markdown('<div class="main-header">🏥 Trợ Lý Y Tế & Đặt Lịch Khám Bệnh Thông Minh</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Tư vấn triệu chứng y khoa YouMed • Hướng dẫn thủ tục BHYT • Đặt lịch khám MySQL thời gian thực</div>', unsafe_allow_html=True)

# Gợi ý câu hỏi nhanh (Quick Prompts)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💡 Bị đau đầu, chóng mặt khám khoa nào?"):
        st.session_state.prompt_input = "Tôi bị đau đầu và chóng mặt mất thăng bằng mấy hôm nay, nên khám khoa nào?"
with col2:
    if st.button("📅 Xem lịch khám của bác sĩ Tim mạch"):
        st.session_state.prompt_input = "Bác sĩ khoa Tim mạch có những lịch khám nào trong những ngày tới?"
with col3:
    if st.button("📋 Bệnh viện có khám BHYT không?"):
        st.session_state.prompt_input = "Bệnh viện có khám BHYT không và thủ tục cần giấy tờ gì?"

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# Xử lý input từ người dùng
user_prompt = st.chat_input("Nhập câu hỏi triệu chứng, thủ tục hoặc yêu cầu đặt lịch khám...")

# Nếu người dùng bấm nút gợi ý
if "prompt_input" in st.session_state and st.session_state.prompt_input:
    user_prompt = st.session_state.prompt_input
    del st.session_state.prompt_input

if user_prompt:
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # 2. Xử lý qua LangGraph Agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Trợ lý y tế đang phân tích và truy vấn dữ liệu..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                input_msg = HumanMessage(content=user_prompt)
                response = hospital_agent.invoke({"messages": [input_msg]}, config)
                bot_reply = response["messages"][-1].content

                st.markdown(bot_reply)

                # Nếu phản hồi có chứa mã phiếu hẹn thành công, làm nổi bật thêm
                if "KB-" in bot_reply and "ĐẶT LỊCH" in bot_reply:
                    st.balloons()

                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                error_msg = f"❌ Đã xảy ra lỗi khi kết nối Agent: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer miễn trừ trách nhiệm y tế
st.markdown("---")
st.caption(
    "⚠️ **Lưu ý y tế:** Trợ lý AI chỉ đóng vai trò tư vấn thông tin và phân luồng ban đầu dựa trên tài liệu YouMed. "
    "Mọi thông tin không thay thế chẩn đoán chuyên sâu trực tiếp của bác sĩ tại bệnh viện."
)
