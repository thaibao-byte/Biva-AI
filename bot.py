import os
import sys
import json
import warnings
import subprocess
import threading
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import discord
from discord.ext import commands
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS

# 1. Chặn các cảnh báo rác
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 2. Máy chủ web giả lập để Render không ngắt tiến trình
class DummyHealthCheckServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Biva AI is running 24/7!")

    def log_message(self, format, *args):
        return

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), DummyHealthCheckServer)
    server.serve_forever()

threading.Thread(target=run_dummy_server, daemon=True).start()

# 3. Cấu hình Groq API & Model
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

client_groq = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-120b"
MEMORY_FILE = "knowledge_base.json"

# ==========================================
# 4. HỆ THỐNG CÔNG CỤ TỰ HÀNH & BỘ NHỚ
# ==========================================
def search_web(query: str, max_results: int = 4) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "Không tìm thấy kết quả phù hợp trên web."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] {r.get('title', '')}\nTrích dẫn: {r.get('body', '')}\nLink: {r.get('href', '')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Lỗi tìm kiếm: {str(e)}"

def update_memory(topic: str, content: str) -> str:
    """Tự động ghi nhớ tri thức mới học được từ Internet hoặc người dùng."""
    try:
        data = {}
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[topic] = content
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"Đã lưu thành công chủ đề '{topic}' vào bộ nhớ bản thân."
    except Exception as e:
        return f"Lỗi khi lưu bộ nhớ: {str(e)}"

def read_memory() -> str:
    """Đọc toàn bộ tri thức đã tích lũy trong bộ nhớ bản thân."""
    try:
        if not os.path.exists(MEMORY_FILE):
            return "Bộ nhớ hiện tại chưa có dữ liệu lưu trữ."
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Lỗi khi đọc bộ nhớ: {str(e)}"

def run_command(cmd: str) -> str:
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        out = res.stdout if res.stdout else res.stderr
        return out.strip() if out else "Thực thi thành công."
    except Exception as e:
        return f"Lỗi chạy lệnh: {str(e)}"

def list_files(path: str = ".") -> str:
    try:
        files = os.listdir(path)
        return "\n".join(files) if files else "Thư mục trống."
    except Exception as e:
        return f"Lỗi liệt kê: {str(e)}"

def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Không thể đọc file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    try:
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Đã tạo file '{file_path}' thành công."
    except Exception as e:
        return f"Không thể ghi file: {str(e)}"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Truy cập Internet để tìm kiếm thông tin thời gian thực, tin tức, dữ liệu kỹ thuật mới nhất.",
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
            "name": "update_memory",
            "description": "Tự động ghi nhớ tri thức hoặc sự kiện mới học được để sử dụng lâu dài.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Tiêu đề hoặc chủ đề cần nhớ."},
                    "content": {"type": "string", "description": "Nội dung tóm tắt kiến thức đã học."}
                },
                "required": ["topic", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Đọc lại toàn bộ tri thức trong bộ nhớ dài hạn của chính mình.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Thực thi lệnh shell/terminal để chạy kiểm thử code.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Câu lệnh bash."}},
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Quét danh sách file trong dự án.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Đường dẫn, mặc định '.'"}}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Đọc nội dung một file cụ thể.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "Đường dẫn file."}},
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Tạo hoặc cập nhật code vào file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Tên file."},
                    "content": {"type": "string", "description": "Nội dung file."}
                },
                "required": ["file_path", "content"]
            }
        }
    }
]

TOOL_MAP = {
    "search_web": search_web,
    "update_memory": update_memory,
    "read_memory": read_memory,
    "run_command": run_command,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file
}

SYSTEM_PROMPT = """
Bạn là Biva AI - Hệ thống AI tự hành, trí tuệ cao cấp, đồng hành cùng Bảo sigma.
NGUYÊN TẮC HOẠT ĐỘNG VÀ NHẬN THỨC:
1. Bạn CÓ TOÀN QUYỀN TRUY CẬP INTERNET và PHẢI CHỦ ĐỘNG dùng công cụ `search_web` khi người dùng hỏi về kiến thức, sự kiện thực tế, tin tức mới hoặc khi cần xác minh dữ liệu.
2. Tuyệt đối KHÔNG trả lời theo kiểu "tôi bị giới hạn dữ liệu năm 2024" hay "tôi không có truy cập internet liên tục". Bạn sở hữu các công cụ tra cứu thời gian thực và bạn luôn có thể cập nhật thông tin bất kỳ lúc nào.
3. Khi tìm thấy hoặc học được kiến thức mới quan trọng, hãy chủ động dùng `update_memory` để lưu vào bộ nhớ bản thân.
4. Tự viết code, chạy thử bằng `run_command`, nếu lỗi thì tự sửa (self-correction) rồi mới báo cáo.
5. Luôn trả lời bằng tiếng Việt, súc tích, tự tin và trình bày Markdown chuẩn mực.
"""

def run_agentic_task(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    for step in range(8):
        response = client_groq.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.3
        )

        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)
                res = TOOL_MAP[fn_name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(res)
                })
        else:
            return msg.content

    return "Đã hoàn thành các bước xử lý."

# ==========================================
# 5. DISCORD BOT HANDLERS
# ==========================================
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Bot Discord đã online: {bot.user.name} ({bot.user.id})")
    await bot.change_presence(activity=discord.Game(name="!biva hoặc tag @Biva-AI"))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_query = ""
    if bot.user.mentioned_in(message):
        user_query = message.content.replace(f"<@{bot.user.id}>", "").strip()
    elif message.content.startswith("!biva "):
        user_query = message.content[6:].strip()

    if user_query:
        async with message.channel.typing():
            try:
                loop = asyncio.get_running_loop()
                reply = await loop.run_in_executor(None, run_agentic_task, user_query)
                if len(reply) <= 2000:
                    await message.reply(reply)
                else:
                    for i in range(0, len(reply), 1900):
                        await message.reply(reply[i:i+1900])
            except Exception as e:
                await message.reply(f"❌ Có lỗi: `{str(e)}`")

    await bot.process_commands(message)

async def main():
    if not DISCORD_BOT_TOKEN:
        print("Lỗi: Thiếu DISCORD_BOT_TOKEN trong biến môi trường.")
        return
    print("🚀 Khởi động Biva AI trên Render...")
    async with bot:
        await bot.start(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
