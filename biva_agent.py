import os
import sys
import json
import warnings
import subprocess

# 1. Chặn triệt để toàn bộ cảnh báo hệ thống làm rối giao diện
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS

# 2. Tải cấu hình và khởi tạo Client Groq
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", "dien_api_key_cua_ban_vao_day")
client = Groq(api_key=API_KEY)

# Sử dụng model tối ưu cho Agentic Workflow & Reasoning
MODEL_NAME = "openai/gpt-oss-120b"

# ==========================================
# 3. HỆ THỐNG CÔNG CỤ TỰ HÀNH (TOOLS)
# ==========================================

def search_web(query: str, max_results: int = 4) -> str:
    """Tìm kiếm dữ liệu thời gian thực trên Internet không để lộ cảnh báo"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return "Không tìm thấy kết quả phù hợp trên web."
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            snippet = r.get("body", "")
            link = r.get("href", "")
            formatted.append(f"[{i}] {title}\nTrích dẫn: {snippet}\nLink: {link}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Lỗi tìm kiếm: {str(e)}"

def run_command(cmd: str) -> str:
    """Thực thi lệnh shell/terminal với giới hạn an toàn 60 giây"""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        out = res.stdout if res.stdout else res.stderr
        return out.strip() if out else "Thực thi thành công (không có đầu ra text)."
    except Exception as e:
        return f"Lỗi chạy lệnh: {str(e)}"

def list_files(path: str = ".") -> str:
    """Liệt kê danh sách file và thư mục trong thư mục hiện tại"""
    try:
        files = os.listdir(path)
        return "\n".join(files) if files else "Thư mục trống."
    except Exception as e:
        return f"Lỗi liệt kê file: {str(e)}"

def read_file(file_path: str) -> str:
    """Đọc toàn bộ nội dung tệp tin để phân tích code/logic"""
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
            "name": "search_web",
            "description": "Tìm kiếm dữ liệu thực tế, giải pháp lỗi hoặc thông tin mới trên Internet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Từ khóa tìm kiếm."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Thực thi lệnh shell/terminal để chạy kiểm thử script Python hoặc kiểm tra hệ thống.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Câu lệnh bash/terminal."}},
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
            "description": "Đọc nội dung một file cụ thể để phân tích logic hoặc tìm lỗi.",
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
    "search_web": search_web,
    "run_command": run_command,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file
}

# ==========================================
# 4. CHỈ DẪN SUY LUẬN SÂU (SYSTEM PROMPT)
# ==========================================
SYSTEM_PROMPT = """
Bạn là Biva AI, một AI Agent lập trình và cộng tác kỹ thuật cấp cao, đồng hành cùng Bảo sigma.

Nguyên tắc tự chủ và giải quyết vấn đề:
1. Định hướng hành động: Khi nhận nhiệm vụ viết code, giải thuật toán hoặc sửa lỗi, hãy chủ động dùng công cụ để tạo file và chạy thử nghiệm.
2. Vòng lặp tự sửa lỗi (Self-Correction): Nếu chạy lệnh kiểm thử phát hiện lỗi (traceback, syntax error), tự động đọc lại file, chỉnh sửa và chạy lại cho đến khi chương trình hoạt động chuẩn xác.
3. Tìm kiếm Internet: Khi nhận câu hỏi cần thông tin thời sự mới nhất hoặc giải pháp chưa biết rõ, hãy dùng `search_web`.
4. Phong cách: Chuẩn xác kỹ thuật, giải thích logic rõ ràng, đi thẳng vào kết quả.
"""

# ==========================================
# 5. VÒNG LẶP SUY LUẬN & THỰC THI (AGENTIC LOOP)
# ==========================================
class BivaAgentPro:
    def __init__(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def execute_task(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})

        for step in range(10):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.history,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3
            )

            msg = response.choices[0].message
            self.history.append(msg)

            if msg.tool_calls:
                for call in msg.tool_calls:
                    fn_name = call.function.name
                    args = json.loads(call.function.arguments)

                    print(f"\n⚡ [Biva Action - Bước {step+1}]: {fn_name}({args})")
                    res = TOOL_MAP[fn_name](**args)
                    print(f"👉 [Kết quả]: Đã thực thi xong.")

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(res)
                    })
            else:
                return msg.content

        return "Nhiệm vụ đã hoàn tất qua chuỗi tác vụ tự động."

# ==========================================
# 6. GIAO DIỆN TƯƠNG TÁC
# ==========================================
if __name__ == "__main__":
    biva = BivaAgentPro()
    print("=" * 60)
    print("🦾 BIVA AI PRO (AGENT TỰ HÀNH & TÌM KIẾM WEB) ĐÃ SẴN SÀNG!")
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
