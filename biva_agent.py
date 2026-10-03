import os
import json
import subprocess
from dotenv import load_dotenv
from groq import Groq

# 1. Tải biến môi trường và khởi tạo Client an toàn
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", "dien_api_key_cua_ban_vao_day")
client = Groq(api_key=API_KEY)

# Sử dụng mô hình hỗ trợ context lớn và Tool Calling tối ưu
MODEL_NAME = "openai/gpt-oss-120b"

# ==========================================
# 2. HỆ THỐNG CÔNG CỤ TỰ HÀNH (TOOLS)
# ==========================================

def run_command(cmd: str) -> str:
    """Chạy lệnh shell/terminal với giới hạn an toàn 60 giây"""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        out = res.stdout if res.stdout else res.stderr
        return out.strip() if out else "Thực thi thành công (không có đầu ra text)."
    except Exception as e:
        return f"Lỗi chạy lệnh: {str(e)}"

def list_files(path: str = ".") -> str:
    """Liệt kê danh sách file và thư mục trong dự án"""
    try:
        files = os.listdir(path)
        return "\n".join(files) if files else "Thư mục trống."
    except Exception as e:
        return f"Lỗi liệt kê file: {str(e)}"

def read_file(file_path: str) -> str:
    """Đọc toàn bộ nội dung tệp tin để phân tích code"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Không thể đọc file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """Tạo mới hoặc ghi nội dung vào file"""
    try:
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Đã ghi thành công file '{file_path}'."
    except Exception as e:
        return f"Không thể ghi file: {str(e)}"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Thực thi lệnh terminal để kiểm tra file, chạy script Python, hoặc test ứng dụng.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Câu lệnh terminal."}},
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Quét toàn bộ cấu trúc thư mục hiện tại.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Đường dẫn thư mục, mặc định '.'"}}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Đọc nội dung một file cụ thể để tìm lỗi hoặc phân tích logic.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "Đường dẫn file cần đọc."}},
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Tạo hoặc cập nhật mã nguồn vào tệp tin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Tên file (ví dụ: main.py, test.py)."},
                    "content": {"type": "string", "description": "Nội dung hoàn chỉnh của file."}
                },
                "required": ["file_path", "content"]
            }
        }
    }
]

TOOL_MAP = {
    "run_command": run_command,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file
}

# ==========================================
# 3. CHỈ DẪN HỆ THỐNG (SYSTEM INSTRUCTION)
# ==========================================
SYSTEM_PROMPT = """
Bạn là Biva AI, một AI Agent lập trình và cộng tác kỹ thuật cấp cao, đồng hành cùng Bảo sigma.

Nguyên tắc tự chủ và giải quyết vấn đề:
1. Định hướng hành động: Khi nhận nhiệm vụ viết code hay sửa lỗi, hãy tự động dùng công cụ để tạo file và chạy thử nghiệm.
2. Vòng lặp tự sửa lỗi: Nếu chạy lệnh kiểm thử phát hiện lỗi (traceback, syntax error), tự động đọc lại file, chỉnh sửa và chạy lại cho đến khi chương trình hoạt động hoàn chỉnh.
3. Phong cách giao tiếp: Đi thẳng vào kết quả, ngắn gọn và chính xác.
"""

# ==========================================
# 4. VÒNG LẶP SUY LUẬN & THỰC THI (AGENTIC LOOP)
# ==========================================
class BivaAgentPro:
    def __init__(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def execute_task(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})

        for step in range(8):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.history,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3
            )

            message = response.choices[0].message
            self.history.append(message)

            if message.tool_calls:
                for call in message.tool_calls:
                    func_name = call.function.name
                    args = json.loads(call.function.arguments)

                    print(f"⚡ [Biva Action - Bước {step+1}]: {func_name}({args})")
                    action_result = TOOL_MAP[func_name](**args)
                    print(f"👉 [Kết quả]: {action_result}\n")

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(action_result)
                    })
            else:
                return message.content

        return "Nhiệm vụ đã hoàn tất qua nhiều bước xử lý tự động."

# ==========================================
# 5. ĐIỀU KHIỂN TƯƠNG TÁC
# ==========================================
if __name__ == "__main__":
    biva = BivaAgentPro()
    print("=" * 60)
    print("🦾 BIVA AI PRO (AGENT TỰ HÀNH) ĐÃ SẴN SÀNG!")
    print("=" * 60)

    while True:
        try:
            inp = input("\nBảo sigma: ").strip()
            if not inp:
                continue
            if inp.lower() in ["exit", "quit"]:
                print("Tạm biệt Bảo sigma!")
                break

            reply = biva.execute_task(inp)
            print(f"\nBiva AI: {reply}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Lỗi]: {e}")
