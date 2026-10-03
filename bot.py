import os
import sys
import json
import warnings
import subprocess
import nest_asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS

# Áp dụng patch async cho môi trường đa luồng/Jupyter
nest_asyncio.apply()

# 1. Chặn các cảnh báo hệ thống
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 2. Tải biến môi trường
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

client_groq = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-120b"

# ==========================================
# 3. HỆ THỐNG CÔNG CỤ TỰ HÀNH (TOOLS)
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
            "description": "Tìm kiếm dữ liệu thực tế trên Internet.",
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
    "run_command": run_command,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file
}

SYSTEM_PROMPT = """
Bạn là Biva AI - Hệ thống AI Agent tự hành cấp cao trên Discord, đồng hành cùng Bảo sigma.
Khi nhận yêu cầu, bạn có thể:
1. Tự động tìm kiếm web nếu cần thông tin thực tế mới nhất.
2. Tự tạo file code và chạy thử bằng terminal để chắc chắn code hoạt động trước khi gửi kết quả.
3. Nếu code lỗi, tự sửa (self-correction) rồi mới báo cáo.
Trả lời bằng tiếng Việt, ngắn gọn, chuẩn Markdown đẹp mắt.
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
# 4. THIẾT LẬP DISCORD CLIENT
# ==========================================
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Bot Discord đã online với tên: {bot.user.name} ({bot.user.id})")
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
                reply = run_agentic_task(user_query)
                if len(reply) <= 2000:
                    await message.reply(reply)
                else:
                    for i in range(0, len(reply), 1900):
                        await message.reply(reply[i:i+1900])
            except Exception as e:
                await message.reply(f"❌ Có lỗi xảy ra trong quá trình xử lý: `{str(e)}`")

    await bot.process_commands(message)

# ==========================================
# 5. KHỞI CHẠY BOT
# ==========================================
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("Lỗi: Chưa tìm thấy DISCORD_BOT_TOKEN trong biến môi trường hoặc file .env")
    else:
        print("🚀 Đang khởi động Biva AI trên Discord...")
        bot.run(DISCORD_BOT_TOKEN)
