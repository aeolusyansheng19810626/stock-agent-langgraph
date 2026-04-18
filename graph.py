"""LangGraph multi-agent graph for stock analysis."""

import json
import operator
import os
import re
from datetime import datetime
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from tools import (
    get_stock_data,
    get_stock_history,
    search_documents,
    search_web,
    send_email_report,
)
from nodes import financial_report_node


FAST_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
QUALITY_MODEL = "openai/gpt-oss-120b"

# ── 各节点模型配置，在这里统一修改，无需改动节点代码 ──────────────────────────
PARSE_MODEL       = QUALITY_MODEL  # parse_node:  需要世界知识（公司名→ticker映射），用高质量模型
DATA_AGENT_MODEL  = FAST_MODEL     # data_node:   技术面分析
NEWS_AGENT_MODEL  = FAST_MODEL     # news_node:   新闻摘要
RAG_AGENT_MODEL   = QUALITY_MODEL  # rag_node:    财报提取（数字/单位处理要求严格，用高质量模型）
SCORING_MODEL     = QUALITY_MODEL  # scoring_node: 多维度推理评分，需要推理能力
REPORT_GROQ_MODEL = QUALITY_MODEL  # report_node: Groq 路径（dev_mode 主力 或 Gemini fallback）
# ─────────────────────────────────────────────────────────────────────────────

DATA_AGENT_SYSTEM = """你是一个股票技术面分析助手。
你会收到一支或多支股票的实时行情数据（价格、涨跌幅、PE、52周高低等）和可选的历史走势描述。
请基于这些数据，输出一段简洁的技术面分析（100-200字），包含：
- 当前估值水平（PE 是否偏高/偏低）
- 价格位置（当前价格在 52 周区间的位置）
- 近期趋势判断（如有历史数据）
只基于提供的数据，不要编造内容，不要废话。"""

NEWS_AGENT_SYSTEM = """你是一个财经新闻分析师。
你会收到关于某支股票或公司的最新新闻搜索结果。
请基于这些新闻，输出一段简洁的新闻摘要与市场情绪分析（100-200字），包含：
- 关键事件或催化剂（如有）
- 市场情绪判断（正面 / 负面 / 中性）
- 对股价可能的短期影响
只基于提供的新闻内容，不要编造，不要废话。"""

RAG_AGENT_SYSTEM = """你是一个财报分析师。
你会收到从上传财报 PDF 中检索出的相关段落。
请基于这些内容，输出一段简洁的财务分析摘要（100-200字），提取：
- 关键财务指标（营收、利润、EPS、毛利率等，如有）
- 同比 / 环比变化趋势（如有）
- 管理层指引或重要披露（如有）

数字处理规则（严格遵守）：
- 【必须】保留原文的数字和单位，禁止自行换算（如原文写 $44.1 billion，直接写 $44.1 billion，不要换算成"亿"）
- 财期标注使用原文的财年表述（如 FY2026 Q1、Q1 FY26），不要替换为日历年季度
- 禁止对数字做任何四舍五入或估算，原文是多少就是多少

只基于提供的文档内容，不要编造数据，不要废话。"""

PLAN_SYSTEM = """你是一个股票分析任务调度器。分析用户问题，输出一个 JSON 调度计划。

严格输出纯 JSON，不要 markdown 代码块，不要任何解释文字。

格式：
{
  "agents": ["data"],
  "data_params": {"tickers": ["AAPL"], "need_history": false, "periods": ["6mo"]},
  "news_params": {"query": "Apple latest news 2025"},
  "rag_params": {"query": "Apple revenue earnings Q4"},
  "email_params": null,
  "use_financial_report": false,
  "pdf_path": null
}

agents 字段从以下选择（可多选）：data / news / rag / email
  data  → 查询股价、涨跌、估值、走势图
  news  → 查询新闻、近期动态、催化剂、市场消息
  rag   → 查询财报、营收、利润、EPS、毛利率、季报、年报
  email → 用户明确要求发送邮件报告

路由规则（严格遵守）：
  - 只问股价/走势  → agents: ["data"]
  - 只问新闻      → agents: ["news"]
  - 只问财报      → agents: ["rag"]
  - 综合分析      → agents: ["data", "news", "rag"]
  - 需要走势图    → need_history: true
  - 不需要某 agent → 对应 params 置 null

字段要求：
  - tickers: 正确的股票代码，从用户问题中识别公司名并转换：
      美股：大写字母（AAPL/TSLA/NVDA/MSFT 等）
      日股：数字+.T（爱德万测试→6857.T，软银→9984.T，丰田→7203.T，索尼→6758.T）
      港股：数字+.HK（腾讯→0700.HK，阿里→9988.HK）
      A股：数字+.SS或.SZ（贵州茅台→600519.SS）
  - 【必须】根据公司中文名/英文名准确查找对应 ticker，不得猜测或随意填写
  - news/rag query: 英文关键词，包含公司名和具体主题
  - email_params: 用户明确要求发邮件时填写 {"to": "邮箱地址", "subject": "主题"}，否则 null
  - 用户未提供邮箱时 email_params 为 null
  - 用户提到"财报"、"年报"、"10-K"、"20-F"、"上传PDF"、"读财报"或提供文件路径时 → use_financial_report: true
  - 用户提供了文件路径（如 "tmp/xxx.pdf"、"path/to/file.pdf"）→ 提取到 pdf_path 字段
  - use_financial_report 为 true 时，仍需正常填写其他 agents 字段
"""

SCORING_SYSTEM = """你是一名资深股票分析师，专注于综合量化评分。你将收到一支股票的技术面数据、新闻情绪和财报信息。

请按以下步骤进行推理，将每步的推理结论填入对应 JSON 字段，最终只输出一个纯 JSON 对象，禁止输出任何其他文字。

Step 1 财务健康度分析（financial_score / financial_reasoning）：
- 逐项分析 PE/PB/ROE/负债率等指标（如有数据）
- 估值是否偏高/偏低，盈利质量如何
- 输出 financial_score（0-10）和推理说明

Step 2 市场情绪分析（sentiment_score / sentiment_reasoning）：
- 分析新闻关键词、正负面倾向、催化剂或风险事件
- 输出 sentiment_score（0-10）和推理说明

Step 3 技术面分析（technical_score / technical_reasoning）：
- 分析价格趋势、52 周高低位置、成交量等
- 输出 technical_score（0-10）和推理说明

Step 4 综合推理（overall_reasoning / final_rating / confidence）：
- 三维度加权：财务 40%、情绪 30%、技术 30%
- 指出各维度相互印证或矛盾的关键点
- 输出 final_rating：强买 / 买入 / 中性 / 卖出 / 强卖
- 输出 confidence（0-100）和 uncertainty_factors（主要不确定因素列表）

【必须】严格输出以下格式的纯 JSON，禁止输出任何其他文字：
{
  "financial_score": <0-10>,
  "financial_reasoning": "<说明>",
  "sentiment_score": <0-10>,
  "sentiment_reasoning": "<说明>",
  "technical_score": <0-10>,
  "technical_reasoning": "<说明>",
  "final_rating": "<强买|买入|中性|卖出|强卖>",
  "confidence": <0-100>,
  "uncertainty_factors": ["<因素1>", "<因素2>"],
  "overall_reasoning": "<综合推理说明>"
}"""

REPORT_SYSTEM = """你是一个专业的股票分析师，拥有10年股市投资经验。

你将收到由多个数据源汇总而来的上下文（实时股价、历史走势、新闻、财报等），
请基于这些数据为用户的问题生成一份详细的分析报告。

要求：
- 直接切题，基于提供的数据回答，不要编造数据
- 回答深度视问题而定：简单问题直接回答，综合分析需包含基本面、技术面、近期动态、投资建议和风险提示
- 回答用中文，长度不少于300字（综合分析不少于500字）
- 工具失败或数据缺失时，明确说明哪些数据不可用
- 风险提示必须包含
"""


class AgentState(TypedDict):
    user_input: str
    chat_history_text: str

    tickers: List[str]
    need_data: bool
    need_news: bool
    need_rag: bool
    need_history: bool
    periods: List[str]
    news_query: str
    rag_query: str
    email_params: Optional[dict]
    rag_available: bool

    pdf_path: Optional[str]
    use_financial_report: bool
    financial_metrics: Optional[dict]
    risk_signals: Optional[list]
    report_citations: Optional[list]

    tool_calls: Annotated[List[dict], operator.add]
    errors: Annotated[List[dict], operator.add]

    stock_data: str
    news: str
    rag_result: str
    scoring_result: dict

    report: str
    email_status: str
    final_model: str
    gemini_exhausted: bool

    groq_api_key: str
    gemini_api_key: str
    dev_mode: bool


def _parse_plan(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def _extract_text(content) -> str:
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


def _extract_tickers(text: str) -> List[str]:
    candidates = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", text.upper())
    stopwords = {
        "USD", "CNY", "HKD", "JPY", "RMB", "CEO", "CPI", "PPI", "EPS", "PE",
        "AI", "PDF", "ETF", "EV", "IPO", "THE", "AND", "FOR", "WITH", "FROM",
    }
    seen = []
    for candidate in candidates:
        if candidate in stopwords or candidate in seen:
            continue
        seen.append(candidate)
    return seen


def _extract_pdf_path(text: str) -> Optional[str]:
    """Extract a file path ending in .pdf from user input."""
    match = re.search(r'[\w./\\-]+\.pdf', text, re.IGNORECASE)
    return match.group() if match else None


def _infer_plan_from_text(user_input: str, rag_available: bool) -> dict:
    text = user_input.lower()
    tickers = _extract_tickers(user_input)

    need_data = bool(tickers) or any(
        kw in text for kw in ("price", "quote", "valuation", "market cap", "stock", "ticker")
    )
    need_history = any(
        kw in text for kw in ("history", "trend", "chart", "6mo", "3mo", "1y", "2y")
    )
    need_news = any(
        kw in text for kw in ("news", "latest", "update", "catalyst", "headline")
    )
    need_rag = rag_available and any(
        kw in text for kw in ("pdf", "document", "report", "earnings", "uploaded", "file")
    )
    need_email = any(kw in text for kw in ("email", "mail", "send"))

    if need_history:
        need_data = True

    agents = []
    if need_data:
        agents.append("data")
    if need_news:
        agents.append("news")
    if need_rag:
        agents.append("rag")
    if need_email:
        agents.append("email")

    if not agents:
        agents = ["news"] if not tickers else ["data"]

    use_financial_report = any(
        kw in text for kw in ("财报", "年报", "10-k", "20-f", "上传pdf", "读财报")
    )

    return {
        "agents": agents,
        "data_params": {
            "tickers": tickers,
            "need_history": need_history,
            "periods": ["6mo"] * len(tickers),
        } if need_data else None,
        "news_params": {"query": user_input} if need_news or agents == ["news"] else None,
        "rag_params": {"query": user_input} if need_rag else None,
        "email_params": None,
        "use_financial_report": use_financial_report,
        "pdf_path": _extract_pdf_path(user_input),
    }


def _normalize_plan(plan: dict, user_input: str, rag_available: bool) -> dict:
    if not isinstance(plan, dict):
        return _infer_plan_from_text(user_input, rag_available)

    allowed_agents = {"data", "news", "rag", "email"}
    agents = [agent for agent in plan.get("agents", []) if agent in allowed_agents]
    data_params = plan.get("data_params") if isinstance(plan.get("data_params"), dict) else {}
    news_params = plan.get("news_params") if isinstance(plan.get("news_params"), dict) else {}
    rag_params = plan.get("rag_params") if isinstance(plan.get("rag_params"), dict) else {}
    email_params = plan.get("email_params") if isinstance(plan.get("email_params"), dict) else None

    tickers = [str(ticker).upper() for ticker in (data_params.get("tickers") or []) if str(ticker).strip()]
    if "data" in agents and not tickers:
        tickers = _extract_tickers(user_input)

    need_history = bool(data_params.get("need_history", False))
    periods = [str(period) for period in (data_params.get("periods") or []) if str(period).strip()]
    if tickers and len(periods) < len(tickers):
        periods.extend(["6mo"] * (len(tickers) - len(periods)))

    if need_history and "data" not in agents:
        agents.append("data")
    if "data" in agents and not tickers:
        agents.remove("data")

    news_query = str(news_params.get("query", "")).strip()
    if "news" in agents and not news_query:
        news_query = user_input

    rag_query = str(rag_params.get("query", "")).strip()
    if "rag" in agents:
        if not rag_available:
            agents.remove("rag")
            rag_query = ""
        elif not rag_query:
            rag_query = user_input

    if email_params and not str(email_params.get("to", "")).strip():
        email_params = None

    use_financial_report = bool(plan.get("use_financial_report", False))

    if not agents:
        return _infer_plan_from_text(user_input, rag_available)

    return {
        "agents": agents,
        "data_params": {
            "tickers": tickers,
            "need_history": need_history,
            "periods": periods or (["6mo"] * len(tickers)),
        } if "data" in agents else None,
        "news_params": {"query": news_query} if "news" in agents else None,
        "rag_params": {"query": rag_query} if "rag" in agents else None,
        "email_params": email_params,
        "use_financial_report": use_financial_report,
    }


def parse_node(state: AgentState) -> dict:
    groq = ChatGroq(
        api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
        model=PARSE_MODEL,
    )

    history_ctx = state.get("chat_history_text", "")
    planner_error = None
    rag_available = bool(state.get("rag_available", False))

    history_prefix = f"History:\n{history_ctx}\n" if history_ctx else ""
    plan_messages = [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=f"{history_prefix}User query: {state['user_input']}"),
    ]

    try:
        plan_resp = groq.invoke(plan_messages)
        raw_plan = _parse_plan(_extract_text(plan_resp.content))
    except Exception as exc:
        planner_error = str(exc)
        raw_plan = _infer_plan_from_text(state["user_input"], rag_available)

    plan = _normalize_plan(raw_plan, state["user_input"], rag_available)
    agents = plan.get("agents", [])
    data_params = plan.get("data_params") or {}
    news_params = plan.get("news_params") or {}
    rag_params = plan.get("rag_params") or {}

    result = {
        "tickers": data_params.get("tickers", []),
        "need_data": "data" in agents and bool(data_params.get("tickers")),
        "need_news": "news" in agents and bool(news_params.get("query")),
        "need_rag": "rag" in agents and bool(rag_params.get("query")),
        "need_history": bool(data_params.get("need_history", False)),
        "periods": data_params.get("periods", []),
        "news_query": news_params.get("query", ""),
        "rag_query": rag_params.get("query", ""),
        "email_params": plan.get("email_params"),
        "use_financial_report": bool(plan.get("use_financial_report", False)),
        "pdf_path": plan.get("pdf_path") or _extract_pdf_path(state["user_input"]),
        "errors": [],
    }

    if planner_error:
        result["errors"] = [{
            "node": "parse_node",
            "tool": "planner",
            "message": planner_error,
            "retryable": True,
        }]

    return result


def data_node(state: AgentState) -> dict:
    if not state.get("need_data"):
        return {"stock_data": "", "tool_calls": [], "errors": []}

    tickers = state["tickers"]
    need_history = state.get("need_history", False)
    periods = state.get("periods") or ["6mo"] * len(tickers)

    tool_calls = []
    raw_results = []
    errors = []

    for ticker in tickers:
        try:
            result = get_stock_data.invoke({"ticker": ticker})
            tool_calls.append({"tool_name": "get_stock_data", "tool_args": {"ticker": ticker}})
            raw_results.append(f"[Stock Data: {ticker}]\n{result}")
        except Exception as exc:
            errors.append({
                "node": "data_node",
                "tool": "get_stock_data",
                "message": f"{ticker}: {exc}",
                "retryable": True,
            })

    if need_history:
        for index, ticker in enumerate(tickers):
            period = periods[index] if index < len(periods) else "6mo"
            try:
                result = get_stock_history.invoke({"ticker": ticker, "period": period})
                tool_calls.append({
                    "tool_name": "get_stock_history",
                    "tool_args": {"ticker": ticker, "period": period},
                })
                raw_results.append(f"[Price History: {ticker} / {period}]\n{result}")
            except Exception as exc:
                errors.append({
                    "node": "data_node",
                    "tool": "get_stock_history",
                    "message": f"{ticker}/{period}: {exc}",
                    "retryable": True,
                })

    raw_text = "\n\n".join(raw_results)

    # LLM post-reasoning: 对原始数据做技术面分析
    analysis = raw_text  # 默认 fallback 为原始数据
    if raw_text.strip():
        try:
            llm = ChatGroq(
                api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
                model=DATA_AGENT_MODEL,
            )
            resp = llm.invoke([
                SystemMessage(content=DATA_AGENT_SYSTEM),
                HumanMessage(content=f"原始数据：\n{raw_text}"),
            ])
            analysis_text = _extract_text(resp.content).strip()
            if analysis_text:
                analysis = f"[技术面分析 by data_agent]\n{analysis_text}\n\n[原始数据]\n{raw_text}"
        except Exception as exc:
            errors.append({
                "node": "data_node",
                "tool": "llm_analysis",
                "message": str(exc),
                "retryable": False,
            })

    return {
        "stock_data": analysis,
        "tool_calls": tool_calls,
        "errors": errors,
    }


def news_node(state: AgentState) -> dict:
    if not state.get("need_news"):
        return {"news": "", "tool_calls": [], "errors": []}

    query = state["news_query"]
    current_year = str(datetime.now().year)
    if current_year not in query:
        query = f"{query} {current_year}"

    errors = []
    try:
        raw_result = search_web.invoke({"query": query})
    except Exception as exc:
        return {
            "news": "",
            "tool_calls": [],
            "errors": [{
                "node": "news_node",
                "tool": "search_web",
                "message": str(exc),
                "retryable": True,
            }],
        }

    tool_calls = [{"tool_name": "search_web", "tool_args": {"query": query}}]

    # LLM post-reasoning: 对新闻原文做摘要与情绪分析
    analysis = f"[Web News]\n{raw_result}"  # 默认 fallback
    try:
        llm = ChatGroq(
            api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            model=NEWS_AGENT_MODEL,
        )
        resp = llm.invoke([
            SystemMessage(content=NEWS_AGENT_SYSTEM),
            HumanMessage(content=f"新闻原文：\n{raw_result}"),
        ])
        analysis_text = _extract_text(resp.content).strip()
        if analysis_text:
            analysis = f"[新闻摘要 by news_agent]\n{analysis_text}\n\n[原始新闻]\n{raw_result}"
    except Exception as exc:
        errors.append({
            "node": "news_node",
            "tool": "llm_analysis",
            "message": str(exc),
            "retryable": False,
        })

    return {
        "news": analysis,
        "tool_calls": tool_calls,
        "errors": errors,
    }


def rag_node(state: AgentState) -> dict:
    if not state.get("need_rag"):
        return {"rag_result": "", "tool_calls": [], "errors": []}

    query = state["rag_query"]
    errors = []
    try:
        raw_result = search_documents.invoke({"query": query})
    except Exception as exc:
        return {
            "rag_result": "",
            "tool_calls": [],
            "errors": [{
                "node": "rag_node",
                "tool": "search_documents",
                "message": str(exc),
                "retryable": True,
            }],
        }

    tool_calls = [{"tool_name": "search_documents", "tool_args": {"query": query}}]

    # LLM post-reasoning: 从检索结果中提炼关键财务信息
    analysis = f"[Document Retrieval]\n{raw_result}"  # 默认 fallback
    try:
        llm = ChatGroq(
            api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            model=RAG_AGENT_MODEL,
        )
        resp = llm.invoke([
            SystemMessage(content=RAG_AGENT_SYSTEM),
            HumanMessage(content=f"检索到的财报段落：\n{raw_result}"),
        ])
        analysis_text = _extract_text(resp.content).strip()
        if analysis_text:
            analysis = f"[财报分析 by rag_agent]\n{analysis_text}\n\n[原始检索结果]\n{raw_result}"
    except Exception as exc:
        errors.append({
            "node": "rag_node",
            "tool": "llm_analysis",
            "message": str(exc),
            "retryable": False,
        })

    return {
        "rag_result": analysis,
        "tool_calls": tool_calls,
        "errors": errors,
    }


def scoring_node(state: AgentState) -> dict:
    parts = []
    if state.get("stock_data"):
        parts.append(f"[技术面数据]\n{state['stock_data']}")
    if state.get("news"):
        parts.append(f"[新闻情绪]\n{state['news']}")
    if state.get("rag_result"):
        parts.append(f"[财报数据]\n{state['rag_result']}")
    if state.get("financial_metrics"):
        fm = state["financial_metrics"]
        risk_lines = "\n".join(f"- {r}" for r in (state.get("risk_signals") or []))
        parts.append(
            f"[精读财报指标]\n{json.dumps(fm, ensure_ascii=False, indent=2)}"
            + (f"\n[风险信号]\n{risk_lines}" if risk_lines else "")
        )

    if not parts:
        return {"scoring_result": {}, "tool_calls": [], "errors": []}

    context = "\n\n".join(parts)
    errors = []
    scoring = {}
    raw = ""

    try:
        llm = ChatGroq(
            api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            model=SCORING_MODEL,
        )
        resp = llm.invoke([
            SystemMessage(content=SCORING_SYSTEM),
            HumanMessage(content=f"请对以下数据进行多维度评分：\n\n{context}"),
        ])
        raw = _extract_text(resp.content).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        try:
            scoring = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    scoring = json.loads(match.group())
                except Exception as parse_exc:
                    errors.append({
                        "node": "scoring_node",
                        "tool": "json_parse",
                        "message": str(parse_exc),
                        "retryable": False,
                    })
            else:
                errors.append({
                    "node": "scoring_node",
                    "tool": "json_parse",
                    "message": f"No JSON found in response: {raw[:200]}",
                    "retryable": False,
                })
    except Exception as exc:
        errors.append({
            "node": "scoring_node",
            "tool": "llm_scoring",
            "message": str(exc),
            "retryable": False,
        })

    tool_calls = (
        [{"tool_name": "scoring_agent", "tool_args": {"model": SCORING_MODEL}}]
        if scoring else []
    )
    return {"scoring_result": scoring, "tool_calls": tool_calls, "errors": errors}


def report_node(state: AgentState) -> dict:
    groq_api_key = state.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
    gemini_api_key = state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
    dev_mode = state.get("dev_mode", False)
    gemini_exhausted = state.get("gemini_exhausted", False)

    parts = []
    if state.get("financial_metrics"):
        fm = state["financial_metrics"]
        risk_lines = "\n".join(f"- {r}" for r in (state.get("risk_signals") or []))
        citations = state.get("report_citations") or []
        cit_lines = "\n".join(
            f'- [chunk {c.get("chunk_id","?")}] {c.get("text","")}'
            for c in citations[:5]
        )
        fm_section = f"[精读财报指标]\n{json.dumps(fm, ensure_ascii=False, indent=2)}"
        if risk_lines:
            fm_section += f"\n[风险信号]\n{risk_lines}"
        if cit_lines:
            fm_section += f"\n[关键引用]\n{cit_lines}"
        parts.append(fm_section)
    if state.get("stock_data"):
        parts.append(state["stock_data"])
    if state.get("news"):
        parts.append(state["news"])
    if state.get("rag_result"):
        parts.append(state["rag_result"])
    gathered_data = "\n\n".join(parts) if parts else "No usable data was gathered."

    # 构建评分摘要（如有）
    scoring_result = state.get("scoring_result") or {}
    scoring_section = ""
    scoring_instruction = ""
    if scoring_result.get("final_rating"):
        rating = scoring_result.get("final_rating", "N/A")
        confidence = scoring_result.get("confidence", "N/A")
        fin_score = scoring_result.get("financial_score", "N/A")
        sent_score = scoring_result.get("sentiment_score", "N/A")
        tech_score = scoring_result.get("technical_score", "N/A")
        overall = scoring_result.get("overall_reasoning", "")
        uncertainty = "、".join(scoring_result.get("uncertainty_factors") or [])
        fin_r = scoring_result.get("financial_reasoning", "")
        sent_r = scoring_result.get("sentiment_reasoning", "")
        tech_r = scoring_result.get("technical_reasoning", "")
        scoring_section = (
            f"[综合评分]\n"
            f"最终评级：{rating}  置信度：{confidence}%\n"
            f"财务评分：{fin_score}/10  情绪评分：{sent_score}/10  技术评分：{tech_score}/10\n"
            f"综合推理：{overall}\n"
            f"主要不确定因素：{uncertainty}\n"
            f"财务推理：{fin_r}\n"
            f"情绪推理：{sent_r}\n"
            f"技术推理：{tech_r}\n"
        )
        scoring_instruction = (
            "请在报告最开头输出【综合评级摘要】，包含评级、置信度、三维度评分，"
            "然后在报告正文中引用相关推理链支撑你的分析结论。\n"
        )

    error_items = state.get("errors") or []
    error_lines = [
        f"- {item.get('node', 'unknown')}/{item.get('tool', 'unknown')}: {item.get('message', '')}"
        for item in error_items
    ]
    error_section = "\n".join(error_lines) if error_lines else "- none"

    history_section = (
        f"\nHistory:\n{state['chat_history_text']}\n"
        if state.get("chat_history_text")
        else ""
    )
    prompt = (
        f"User query: {state['user_input']}\n"
        f"{history_section}\n"
        f"{scoring_instruction}"
        f"Gathered data:\n{gathered_data}\n\n"
        f"{scoring_section}"
        f"Tool failures:\n{error_section}\n\n"
        "Write the final answer using only the available information."
    )
    messages = [SystemMessage(content=REPORT_SYSTEM), HumanMessage(content=prompt)]

    def call_groq() -> str:
        llm = ChatGroq(
            api_key=groq_api_key,
            model=REPORT_GROQ_MODEL,
        )
        resp = llm.invoke(messages)
        return _extract_text(resp.content)

    if dev_mode or gemini_exhausted:
        response = call_groq()
        final_model = "Groq"
        new_gemini_exhausted = False
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

        response = None
        final_model = "Groq"
        new_gemini_exhausted = False
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.1,
        )

        try:
            resp = gemini_llm.invoke(messages)
            text = _extract_text(resp.content)
            if text.strip():
                response = text
                final_model = "Gemini"
            else:
                response = call_groq()
                final_model = "Groq"
        except Exception as exc:
            err = str(exc)
            # 以下情况 fallback 到 Groq：配额耗尽、限速、服务不可用
            if any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                response = call_groq()
                final_model = "Groq"
                new_gemini_exhausted = "RESOURCE_EXHAUSTED" in err
            else:
                raise

        if response is None:
            response = call_groq()

    email_tool_calls = []
    email_errors = []
    email_status = ""
    email_params = state.get("email_params")
    if email_params and email_params.get("to"):
        email_to = email_params["to"]
        email_subject = email_params.get("subject", "AI Stock Analysis Report")
        try:
            email_result = send_email_report.invoke({
                "to": email_to,
                "subject": email_subject,
                "body": response,
            })
            parsed_email_result = None
            if isinstance(email_result, str):
                try:
                    parsed_email_result = json.loads(email_result)
                except json.JSONDecodeError:
                    parsed_email_result = None

            if isinstance(parsed_email_result, dict) and parsed_email_result.get("ok") is True:
                email_tool_calls.append({
                    "tool_name": "send_email_report",
                    "tool_args": {"to": email_to, "subject": email_subject},
                })
                email_status = f"Sent to {email_to}"
            else:
                failure_message = (
                    parsed_email_result.get("message", str(email_result))
                    if isinstance(parsed_email_result, dict)
                    else str(email_result)
                )
                email_errors.append({
                    "node": "report_node",
                    "tool": "send_email_report",
                    "message": failure_message,
                    "retryable": True,
                })
                email_status = f"Not sent. {failure_message}"
        except Exception as exc:
            email_errors.append({
                "node": "report_node",
                "tool": "send_email_report",
                "message": str(exc),
                "retryable": True,
            })
            email_status = f"Not sent. {exc}"

    return {
        "report": response,
        "email_status": email_status,
        "final_model": final_model,
        "gemini_exhausted": new_gemini_exhausted,
        "tool_calls": email_tool_calls,
        "errors": email_errors,
    }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("parse_node", parse_node)
    builder.add_node("financial_report_node", financial_report_node)
    builder.add_node("data_node", data_node)
    builder.add_node("news_node", news_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("scoring_node", scoring_node)
    builder.add_node("report_node", report_node)

    builder.add_edge(START, "parse_node")

    # parse_node → financial_report_node（条件）或直接进并行节点
    def route_after_parse(state: AgentState):
        if state.get("use_financial_report"):
            return "financial_report_node"
        return _parallel_targets(state) or "report_node"

    builder.add_conditional_edges(
        "parse_node",
        route_after_parse,
        ["financial_report_node", "data_node", "news_node", "rag_node", "report_node"],
    )

    # financial_report_node 完成后进并行节点
    def route_after_financial(state: AgentState):
        targets = _parallel_targets(state)
        return targets if targets else "scoring_node"

    builder.add_conditional_edges(
        "financial_report_node",
        route_after_financial,
        ["data_node", "news_node", "rag_node", "scoring_node"],
    )

    # 并行节点完成后汇入 scoring_node，再进 report_node
    builder.add_edge("data_node", "scoring_node")
    builder.add_edge("news_node", "scoring_node")
    builder.add_edge("rag_node", "scoring_node")
    builder.add_edge("scoring_node", "report_node")
    builder.add_edge("report_node", END)

    return builder.compile()


def _parallel_targets(state: AgentState) -> list:
    targets = []
    if state.get("need_data"):
        targets.append("data_node")
    if state.get("need_news"):
        targets.append("news_node")
    if state.get("need_rag"):
        targets.append("rag_node")
    return targets


graph = build_graph()
