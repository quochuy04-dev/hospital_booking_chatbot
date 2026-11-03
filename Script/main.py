import sys
import uuid
from pathlib import Path

# Đảm bảo console Windows hiển thị đúng tiếng Việt UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục Script vào sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from agent import hospital_agent
from langchain_core.messages import HumanMessage


def main():
    print("=" * 65)
    print("🏥 HỆ THỐNG TRỢ LÝ Y TẾ & ĐẶT LỊCH KHÁM (LANGGRAPH REACT AGENT V2)")
    print("   • Cơ sở dữ liệu: MySQL (Laragon localhost:3306)")
    print("   • Tri thức y khoa: 100 bài viết YouMed & PDF Quy trình bệnh viện")
    print("=" * 65)
    print("💡 Mẹo: Bạn có thể hỏi thủ tục, mô tả triệu chứng hoặc yêu cầu đặt lịch khám.")
    print("   (Gõ 'exit', 'quit' hoặc 'q' để kết thúc cuộc trò chuyện)\n")

    # Tạo phiên hội thoại duy nhất cho phiên làm việc này
    session_thread_id = f"session_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_thread_id}}

    while True:
        try:
            user_input = input("👤 Bệnh nhân: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Cảm ơn bạn đã sử dụng dịch vụ. Chúc bạn thật nhiều sức khỏe!")
                break

            print("\n🤖 Trợ lý y tế đang xử lý...", end="\r")

            # Gửi tin nhắn vào LangGraph Agent
            input_message = HumanMessage(content=user_input)
            response = hospital_agent.invoke({"messages": [input_message]}, config)

            # Lấy câu trả lời cuối cùng của Agent
            bot_reply = response["messages"][-1].content
            print(f"🤖 Bot:\n{bot_reply}\n")
            print("-" * 65)

        except KeyboardInterrupt:
            print("\n👋 Đã dừng chương trình. Tạm biệt bạn!")
            break
        except Exception as e:
            print(f"\n❌ Có lỗi xảy ra trong quá trình xử lý: {e}\n")


if __name__ == "__main__":
    main()
