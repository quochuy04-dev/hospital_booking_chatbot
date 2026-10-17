from pathlib import Path
import sys
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Xác định thư mục gốc của project và thư mục Script
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.append(str(SCRIPT_DIR))

from rag_service import answer_hospital_faq, triage_symptom_rag
from triage import classify_intent

_ = load_dotenv(find_dotenv(str(BASE_DIR / ".env")))

# Đảm bảo console Windows hiển thị đúng tiếng Việt
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
  try:
    sys.stdout.reconfigure(encoding="utf-8")
  except Exception:
    pass

# Memory Contextualizer
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Dựa vào lịch sử hội thoại và tin nhắn mới nhất, hãy viết lại thành một câu độc lập (standalone query) có đầy đủ ngữ cảnh để hiểu mà không cần đọc lại lịch sử.

QUY TẮC BẮT BUỘC:
1. TUYỆT ĐỐI KHÔNG trả lời câu hỏi và KHÔNG đóng vai chatbot.
2. Nếu tin nhắn của người dùng là lời chào hỏi, cảm ơn, hoặc một chủ đề mới không liên quan đến cuộc trò chuyện y tế trước đó (ví dụ: chào hỏi, thời tiết, toán học, câu hỏi ngoài lề): HÃY TRẢ VỀ NGUYÊN VĂN tin nhắn đó, KHÔNG thay đổi.
3. CHỈ viết lại câu khi tin nhắn là câu nói tiếp/câu trả lời cụt (ví dụ: 'còn giá thì sao', 'khoảng 3 ngày nay', 'có cần nhịn ăn không') cần kết hợp với ngữ cảnh trước đó để rõ nghĩa.
4. Chỉ xuất ra duy nhất câu độc lập (hoặc câu gốc), không thêm bất kỳ tiền tố hay giải thích nào.
"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

chat_history = []


def handle_patient_message(user_message: str) -> dict:
  global chat_history

  # 1. Tái tạo ngữ cảnh nếu có hội thoại trước đó
  standalone_query = (
      contextualize_chain.invoke({
          "chat_history": chat_history,
          "question": user_message,
      })
      if chat_history
      else user_message
  )

  # 2. Bước Triage: Phân loại yêu cầu
  triage_result = classify_intent(standalone_query)
  intent = triage_result.loai_yeu_cau

  # 3. Điều phối luồng xử lý theo intent
  if intent == "CHAO_HOI_CAM_ON":
    user_lower = user_message.lower()
    if any(k in user_lower for k in ["cảm ơn", "cam on", "thank", "thanks"]):
      bot_reply = (
          "Rất vui được hỗ trợ bạn! Nếu bạn cần thêm thông tin về thủ tục khám "
          "hoặc có bất kỳ thắc mắc sức khỏe nào khác, đừng ngần ngại hỏi tôi nhé. "
          "Chúc bạn thật nhiều sức khỏe!"
      )
      action_type = "THANK_YOU"
    else:
      bot_reply = (
          "Xin chào! Tôi là Trợ lý ảo y tế của bệnh viện.\n"
          "Tôi có thể hỗ trợ bạn:\n"
          "1. Giải đáp quy trình, thủ tục, thời gian khám, chi phí và BHYT.\n"
          "2. Tiếp nhận triệu chứng sức khỏe ban đầu để tư vấn chuyên khoa phù hợp.\n\n"
          "Bạn đang quan tâm đến thông tin nào, hay có triệu chứng gì cần được hỗ trợ không?"
      )
      action_type = "GREETING"

  elif intent == "NGOAI_PHAM_VI":
    bot_reply = (
        "Xin lỗi bạn, tôi là Trợ lý y tế chuyên hỗ trợ về quy trình khám bệnh "
        "và tư vấn sơ bộ chuyên khoa.\n"
        "Câu hỏi của bạn hiện nằm ngoài phạm vi hỗ trợ của tôi. "
        "Vui lòng đặt câu hỏi liên quan đến thủ tục bệnh viện hoặc sức khỏe để được hỗ trợ tốt nhất."
    )
    action_type = "OUT_OF_SCOPE"

  elif intent == "QUY_TRINH_KHAM":
    # Đẩy sang RAG: FAQ quy trình
    bot_reply = answer_hospital_faq(standalone_query)
    action_type = "FAQ_ANSWER"

  else:  # intent == "TRIEU_CHUNG_LAM_SANG"
    # Luồng triệu chứng: Truy xuất tài liệu triệu chứng y khoa -> Gợi ý chuyên khoa
    bot_reply = triage_symptom_rag(
        query=standalone_query, symptoms=triage_result.trieu_chung_ghi_nhan
    )
    action_type = "BOOKING_REDIRECT"

  # Cập nhật Memory
  chat_history.append(HumanMessage(content=user_message))
  chat_history.append(AIMessage(content=bot_reply))
  if len(chat_history) > 10:
    chat_history = chat_history[-10:]

  return {
      "status": "success",
      "action_type": action_type,
      "bot_reply": bot_reply,
      "intent": intent,
  }


if __name__ == "__main__":
  print("=" * 60)
  print("HỆ THỐNG TRỢ LÝ Y TẾ THEO KIẾN TRÚC ROUTER - RAG")
  print("=" * 60)
  while True:
    msg = input("\n👤 Bệnh nhân: ").strip()
    if msg.lower() in ["exit", "quit"]:
      break
    res = handle_patient_message(msg)
    print(f"\n[{res['action_type']}]")
    print(f"🤖 Bot:\n{res['bot_reply']}")