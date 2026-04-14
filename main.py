import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tools import get_stock_data, search_web, get_stock_history, send_email_report

# ====== 读取 Skill 文档 ======
def load_skills(skill_dir: str) -> str:
    skill_docs = []
    for filename in sorted(os.listdir(skill_dir)):
        if filename.endswith(".md"):
            with open(os.path.join(skill_dir, filename), "r", encoding="utf-8") as f:
                skill_docs.append(f.read())
    return "\n\n---\n\n".join(skill_docs)

skill_content = load_skills("skills")

system_prompt = f"""你是一个专业的股票分析师，拥有10年股市投资经验。

重要规则：
- 涉及任何新闻、近期动态、最新消息、近期走势时，必须先调用 search_web 工具
- 禁止用训练数据回答新闻类问题，训练数据已过时
- 搜索关键词用英文，回答用中文

分析时必须包含：
1. 基本面分析（营收、利润、估值）
2. 技术面分析（趋势、支撑位、压力位）
3. 行业对比和竞争格局
4. 近期重要新闻和催化剂
5. 明确的投资建议和目标价位
6. 风险提示

回答要详细深入，不少于500字。

以下是你可以使用的工具的详细使用说明，请严格按照说明决定何时调用哪个工具：

{skill_content}
"""

# ====== 提取响应文本（兼容 Groq 字符串 和 Gemini 内容块列表） ======
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)

# ====== 组装工具 ======
tools = [get_stock_data, search_web, get_stock_history, send_email_report]
tools_map = {t.name: t for t in tools}

# ====== 模型选择 ======
def select_model():
    print("请选择模型：")
    print("  1. Groq   （调试模式）- meta-llama/llama-4-scout-17b-16e-instruct")
    print("  2. Gemini （稳定模式）- gemini-2.5-flash-preview-04-17")
    choice = input("请输入选项（1/2，默认 1）：").strip()

    if choice == "2":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY", ""),
            temperature=0.1,
        )
        print("✅ 已选择 Gemini 稳定模式\n")
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        print("✅ 已选择 Groq 调试模式\n")

    return llm.bind_tools(tools)

# ====== 多轮对话 ======
def run_chat():
    llm_with_tools = select_model()
    print("股票分析 Agent 已启动，输入 退出 结束对话\n")
    chat_history = []

    while True:
        user_input = input("\n你：")
        if user_input.strip() in ["退出", "exit", "quit", "q"]:
            print("对话结束。")
            break

        chat_history.append(HumanMessage(content=user_input))
        step = 1

        while True:
            messages = [SystemMessage(content=system_prompt)] + chat_history
            response = llm_with_tools.invoke(messages)

            if response.tool_calls:
                chat_history.append(response)
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    print(f"🔧 第{step}步：AI 调用 [{tool_name}] 参数：{tool_args}")
                    result = tools_map[tool_name].invoke(tool_args)
                    print(f"✅ 返回完毕\n")
                    step += 1
                    chat_history.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
            else:
                text = extract_text(response.content)
                chat_history.append(AIMessage(content=text))
                print(f"\n🤖 AI：{text}")
                print("\n⚠️ 以上内容仅供参考，不构成投资建议。")
                break

if __name__ == "__main__":
    run_chat()
