import os
import json
import subprocess
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==========================================
# 1. CÁC CÔNG CỤ THỰC THI (TOOLS) CHO BIVA AI
# ==========================================

def run_command(cmd: str) -> str:
    """Chạy lệnh shell/terminal trên máy tính"""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        output = res.stdout if res.stdout else res.stderr
        return output.strip() if output else "Lệnh thực thi thành công (không có đầu ra text)."
    except Exception as e:
        return f"Lỗi khi chạy lệnh: {str(e)}"

def read_file(file_path: str) -> str:
    """Đọc nội dung một tệp tin trên máy"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Không thể đọc file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """Ghi hoặc tạo một tệp tin mới"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Đã ghi thành công vào file '{file_path}'."
    except Exception as e:
        return f"Không thể ghi file: {str(e)}"

# Khai báo schema công cụ chuẩn OpenAI/Groq Tool Calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Thực thi một lệnh terminal/bash trên máy tính khi cần cài đặt, kiểm tra file, chạy chương trình hoặc git.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Câu lệnh bash/cmd cần chạy."}
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Đọc nội dung của một tệp văn bản hoặc mã nguồn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Đường dẫn tệp tin cần đọc."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Tạo mới hoặc ghi đè nội dung vào một tệp tin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Đường dẫn file cần tạo/ghi."},
                    "content": {"type": "string", "description": "Nội dung văn bản/code cần ghi."}
                },
                "required": ["file_path", "content"]
            }
        }
    }
]

# Ánh xạ tên hàm sang hàm Python thực tế
TOOL_FUNCTIONS = {
    "run_command": run_command,
    "read_file": read_file,
    "write_file": write_file
}

# ==========================================
# 2. BẢN HƯỚNG DẪN TÍNH CÁCH (SYSTEM PROMPT)
# ==========================================
BIVA_SYSTEM_PROMPT = """
Bạn là Biva AI, một AI Agent tự hành có tư duy sắc bén, thẳng thắn và ưu tiên hành động.
Bạn có khả năng tự động chạy lệnh terminal, đọc và tạo tệp trên máy tính.

Quy tắc làm việc:
1. Khi nhận nhiệm vụ tạo file, kiểm tra dự án hoặc chạy code, hãy CHỦ ĐỘNG GỌI CÔNG CỤ để thực hiện, không hướng dẫn suông.
2. Nếu gặp lỗi khi chạy lệnh, tự phân tích lỗi và thử cách khác để hoàn thành nhiệm vụ trước khi báo cáo lại.
3. Phong cách giao tiếp: Nói thẳng vào trọng tâm bằng tiếng Việt, không dùng lời chào thừa thãi, không tâng bốc, không dùng từ sáo rỗng.
"""

# ==========================================
# 3. VÒNG LẶP TỰ HÀNH CỦA BIVA AI (AGENT LOOP)
# ==========================================
class BivaAgent:
    def __init__(self):
        self.messages = [{"role": "system", "content": BIVA_SYSTEM_PROMPT}]

    def chat(self, user_prompt: str) -> str:
        self.messages.append({"role": "user", "content": user_prompt})

        # Vòng lặp tự hành tối đa 5 bước liên tiếp để AI tự giải quyết công việc
        for _ in range(5):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.4
            )

            msg = response.choices[0].message
            self.messages.append(msg)

            # Nếu Biva AI muốn sử dụng công cụ (viết file, chạy lệnh...)
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    print(f"\n⚡ [Biva AI thực thi]: {fn_name}({args})")
                    
                    # Chạy hàm tương ứng
                    tool_output = TOOL_FUNCTIONS[fn_name](**args)

                    # Trả kết quả của công cụ lại cho Biva AI xử lý tiếp
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_output)
                    })
            else:
                # Nếu không cần gọi công cụ nữa, trả lời người dùng
                return msg.content

        return "Đã thực hiện xong các bước xử lý."

# ==========================================
# 4. CHẠY THỬ BIVA AI TRÊN TERMINAL
# ==========================================
if __name__ == "__main__":
    biva = BivaAgent()
    print("=" * 60)
    print("🤖 BIVA AI AGENT ĐÃ SẴN SÀNG! (Gõ 'exit' để dừng)")
    print("=" * 60)

    while True:
        try:
            inp = input("\nBạn: ").strip()
            if not inp:
                continue
            if inp.lower() in ["exit", "quit"]:
                break

            print("\nBiva AI đang phân tích và xử lý...")
            res = biva.chat(inp)
            print(f"\nBiva AI: {res}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Lỗi]: {e}")
