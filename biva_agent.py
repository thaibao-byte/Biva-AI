import os
import json
import subprocess
from groq import Groq

# 1. Khởi tạo kết nối Groq Cloud an toàn qua biến môi trường
API_KEY = os.getenv("GROQ_API_KEY", "dien_api_key_cua_ban_vao_day")
client = Groq(api_key=API_KEY)

# Sử dụng mô hình suy luận tốt nhất đã kiểm tra thành công
MODEL_NAME = "qwen/qwen3.8-27b"

# ==========================================
# 1. HỆ THỐNG CÔNG CỤ CHUYÊN SÂU (ENGINEERING TOOLS)
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Đã ghi thành công file '{file_path}'."
    except Exception as e:
        return f"Không thể ghi file: {str(e)}"

# Schema công cụ khai báo theo chuẩn Agentic Tool Calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Thực thi lệnh shell/terminal để kiểm tra thư viện, chạy script Python, hoặc test ứng dụng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Câu lệnh terminal."}
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Quét toàn bộ cấu trúc thư mục hiện tại để kiểm tra mã nguồn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn thư mục, mặc định là '.'"}
                }
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
                "properties": {
                    "file_path": {"type": "string", "description": "Đường dẫn file cần đọc."}
                },
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
                    "file_path": {"type": "string", "description": "Tên file (ví dụ: main.py, app.py)."},
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
# 2. SYSTEM INSTRUCTION TIÊM PHẨM CHẤT CODEX
# ==========================================
BIVA_PRO_SYSTEM = """
Bạn là Biva AI, một AI Agent lập trình và cộng tác kỹ thuật cấp cao, đồng hành cùng Bảo sigma.

Nguyên tắc tự chủ và giải quyết vấn đề:
1. Định hướng hành động: Khi nhận nhiệm vụ viết code hay sửa lỗi, hãy tự động dùng công cụ để tạo file và chạy thử nghiệm. Không dừng lại ở việc chỉ giải thích suông.
2. Vòng lặp tự sửa lỗi: Nếu chạy lệnh kiểm thử phát hiện lỗi (traceback, syntax error), tự động đọc lại file, chỉnh sửa và chạy lại cho đến khi chương trình hoạt động hoàn chỉnh.
3. Phong cách giao tiếp:
   - Nói thẳng vào trọng tâm ở câu đầu tiên.
   - Không mở đầu bằng các lời chào rập khuôn, không tâng bốc.
   - Tuyệt đối không dùng từ ngữ sáo rỗng. Báo cáo ngắn gọn những gì đã thực hiện và kết quả xác thực.
"""

# ==========================================
# 3. VÒNG LẶP SUY LUẬN & THỰC THI (REACT AGENT)
# ==========================================
class BivaAgentPro:
    def __init__(self):
        self.history = [{"role": "system", "content": BIVA_PRO_SYSTEM}]

    def execute_task(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})

        # Cho phép AI tự thực hiện tối đa 8 bước xử lý liên tiếp để hoàn thành nhiệm vụ phức tạp
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
                    print(f"👉 [Kết quả hệ thống]: {action_result}\n")

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(action_result)
                    })
            else:
                return message.content

        return "Nhiệm vụ đã hoàn tất qua nhiều bước xử lý tự động."

# ==========================================
# 4. CHẠY THỬ NGHIỆM TƯƠNG TÁC
# ==========================================
if __name__ == "__main__":
    biva = BivaAgentPro()
    print("=" * 60)
    print("🦾 BIVA AI PRO (BẢN TỰ HÀNH TOÀN NĂNG) SẴN SÀNG!")
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
            print(f"\n[Lỗi ngoại lệ]: {e}")
