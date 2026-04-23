"""LangGraph multi-agent graph for stock analysis."""

import json
import logging
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

    need_scoring: bool
    need_risk: bool
    need_comparison: bool
    need_hypothesis: bool
    need_deep_read: bool
    need_reflection: bool
    comparison_dimensions: List[str]
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
    risk_result: Optional[dict]
    comparison_result: Optional[dict]
    hypothesis_result: Optional[dict]
    deep_read_result: Optional[dict]
    reflection_result: Optional[str]

    report: str
    final_report: str
    email_status: str
    final_model: str
    gemini_exhausted: bool

    groq_api_key: str
    gemini_api_key: str
    dev_mode: bool


logger = logging.getLogger(__name__)

# ── 5-tier 模型配置 ───────────────────────────────────────────────────────────
TIER_TOP       = "openai/gpt-oss-120b"
TIER_UPPER_MID = "openai/gpt-oss-20b"
TIER_MID       = "qwen/qwen3-32b"
TIER_LOW       = "meta-llama/llama-4-scout-17b-16e-instruct"
TIER_DEBUG     = "llama-3.1-8b-instant"

# Quality 节点从上到下依次降级；Fast 节点直接用 TIER_LOW，不降级
QUALITY_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW, TIER_DEBUG]


RATE_LIMIT_KEYWORDS = ("429", "rate_limit", "rate limit", "503", "over_capacity", "model_overloaded")

# Fast 节点（data/news）直接使用的单一模型
DATA_AGENT_MODEL = TIER_LOW
NEWS_AGENT_MODEL = TIER_LOW

# 各节点专用 cascade（替代固定单模型，兼顾质量与限速容错）
RISK_MODEL_CASCADE       = [TIER_TOP, TIER_UPPER_MID, TIER_MID]
COMPARISON_MODEL_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID]
REFLECTION_MODEL_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID]
DEEP_READ_S1_CASCADE     = [TIER_MID, TIER_LOW, TIER_DEBUG]   # 提取为主，不需要强模型
DEEP_READ_S2_CASCADE     = [TIER_TOP, TIER_UPPER_MID, TIER_MID]  # 批判推理，保持高质量

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

PLAN_SYSTEM = """股票分析任务调度器。严格输出纯 JSON，禁止任何其他文字。

{
  "agents": ["data"|"news"|"rag"|"email"],
  "data_params": {"tickers": ["AAPL"], "need_history": false, "periods": ["6mo"]},
  "news_params": {"query": "Apple news 2026"},
  "rag_params": {"query": "Apple revenue Q4"},
  "email_params": null,
  "need_scoring": false,
  "need_risk": false,
  "need_comparison": false,
  "need_hypothesis": false,
  "need_deep_read": false,
  "comparison_dimensions": [],
  "need_reflection": false,
  "use_financial_report": false,
  "pdf_path": null
}

agents 选择规则：
  data → 股价/走势/估值  news → 新闻/动态  rag → 财报/营收/利润/知识库/向量数据库  email → 明确要求发邮件
  综合分析 → ["data","news","rag"]  单一问题 → 单个agent

tickers 转换规则：
  美股：AAPL/TSLA/NVDA/MSFT  日股：数字.T（丰田→7203.T）  港股：数字.HK（腾讯→0700.HK）  A股：数字.SS/.SZ

布尔字段规则（true 条件）：
  need_scoring     → 综合分析/投资建议/买卖评级
  need_risk        → 风险/隐患/担忧/risk
  need_comparison  → 2个以上ticker/对比/比较/PK/compare
  need_hypothesis  → 如果/假设/若/what if/假如
  need_deep_read   → 精读/质疑/批判/论文
  need_reflection  → 深度/严谨/全面 或 need_scoring=true
  use_financial_report → 财报/年报/10-K/20-F/PDF路径

其他：
  pdf_path → 从输入提取 .pdf 路径
  comparison_dimensions → 用户指定维度，否则 []
  不需要的 agent 对应 params 置 null
  use_financial_report=true 时，仍需正常填写其他 agents 字段
"""

DEEP_READ_STAGE1_SYSTEM = """你是一个专业文档分析师。
基于以下文档内容，提取：
{
  "doc_type": "财报|论文|研究报告",
  "summary": "核心内容100字摘要",
  "key_metrics": [
    {"name": "指标名", "value": "数值", "significance": "重要性说明"}
  ],
  "key_claims": ["核心论点1", "核心论点2"],
  "time_period": "时间范围"
}
只输出纯 JSON。"""

DEEP_READ_STAGE2_SYSTEM = """你是一个批判性思维专家。
基于以下文档的核心论点，进行深度质疑：
{
  "critical_questions": [
    {
      "question": "质疑问题",
      "basis": "质疑依据",
      "severity": "高|中|低"
    }
  ],
  "logical_gaps": ["逻辑漏洞1"],
  "missing_info": ["缺失信息1"],
  "reliability_score": 0-10,
  "conclusion": "综合可信度评估"
}
只输出纯 JSON。"""

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

COMPARISON_SYSTEM = """你是一名股票对比分析师。你会收到多只股票的行情/技术面数据和新闻信息。

请对以下股票按用户指定维度逐项对比，输出纯 JSON，禁止输出任何其他文字。

输出格式：
{
  "dimensions": ["估值", "技术面", "新闻情绪"],
  "rankings": [
    {"rank": 1, "ticker": "AAPL", "score": 8.2, "summary": "..."},
    {"rank": 2, "ticker": "MSFT", "score": 7.5, "summary": "..."}
  ],
  "comparison_table": {
    "估值": {"AAPL": "PE 28，略高", "MSFT": "PE 26，合理"},
    "技术面": {"AAPL": "...", "MSFT": "..."}
  },
  "winner": "AAPL",
  "winner_reasoning": "综合来看..."
}"""

RISK_SYSTEM = """你是一名股票风险分析师。你将收到一支股票的财务指标、技术面数据和新闻信息。

请只基于输入数据识别关键风险，输出纯 JSON，禁止输出任何其他文字。

输出格式：
{
  "risk_factors": [
    {
      "id": 1,
      "category": "财务风险|市场风险|宏观风险",
      "title": "...",
      "reasoning": "数据依据 → 传导路径 → 最终影响",
      "severity": "高|中|低",
      "trigger": "什么情况下会爆发"
    }
  ],
  "risk_summary": "整体风险评估一句话"
}"""

REFLECTION_SYSTEM = """你是一个严苛的金融分析审核员。
下面是一份股票分析报告，请从以下角度批判：
1. 数据是否有遗漏或矛盾
2. 结论是否有逻辑跳跃
3. 风险提示是否充分
4. 投资建议是否过于武断

请输出纯 JSON，禁止输出任何其他文字：
{
  "issues": ["问题1", "问题2"],
  "severity": "高/中/低",
  "revised_sections": "需要修订的段落建议",
  "revised_report": "基于以上批判修订后的完整报告"
}"""

HYPOTHESIS_SYSTEM = """你是一个宏观经济与股票分析专家，擅长多步推理。
用户提出了一个假设性问题，请按以下步骤推演：

Step 1: 识别假设条件（What）
Step 2: 分析直接影响（传导路径）
Step 3: 分析对目标股票/行业的二阶影响
Step 4: 给出概率加权的情景结论

输出纯 JSON：
{
  "hypothesis": "用户的假设条件",
  "transmission_path": ["路径1", "路径2"],
  "scenarios": [
    {"name": "基准情景", "probability": 50, "impact": "...", "price_impact": "+5%~+8%"},
    {"name": "悲观情景", "probability": 30, "impact": "...", "price_impact": "-3%~-5%"},
    {"name": "乐观情景", "probability": 20, "impact": "...", "price_impact": "+10%~+15%"}
  ],
  "conclusion": "综合推演结论"
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
def hypothesis_node(state: AgentState) -> dict:
    if not state.get("need_hypothesis"):
        return {"hypothesis_result": {}, "tool_calls": [], "errors": []}

    parts = []
    if state.get("stock_data"):
        parts.append(f"[技术面数据]\n{state['stock_data']}")
    if state.get("news"):
        parts.append(f"[新闻信息]\n{state['news']}")
    if state.get("rag_result"):
        parts.append(f"[财报数据]\n{state['rag_result']}")

    context = "\n\n".join(parts) if parts else "无背景数据"
    errors = []
    hypothesis_result = {}
    # 强制使用 TIER_TOP，但会进入 cascade 降级逻辑
    hypothesis_model = TIER_TOP
    _hyp_usage = None

    try:
        raw, hypothesis_model, _hyp_usage = _invoke_with_cascade(
            [
                SystemMessage(content=HYPOTHESIS_SYSTEM),
                HumanMessage(content=f"背景数据：\n{context}\n\n用户假设请求：{state['user_input']}")
            ],
            state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            [TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW],
        )
        hypothesis_result = _parse_json_from_text(raw)
        if not hypothesis_result:
             errors.append({
                "node": "hypothesis_node",
                "tool": "json_parse",
                "message": f"No JSON found in response: {raw[:200]}",
                "retryable": False,
            })
    except Exception as exc:
        errors.append({
            "node": "hypothesis_node",
            "tool": "llm_hypothesis",
            "message": str(exc),
            "retryable": False,
        })

    tool_calls = (
        [{"tool_name": "llm", "tool_args": {"node": "hypothesis", "model": hypothesis_model}}]
        if hypothesis_result else []
    )
    if _hyp_usage:
        tool_calls.append({"node": "hypothesis", "token_usage": _hyp_usage})
    return {"hypothesis_result": hypothesis_result, "tool_calls": tool_calls, "errors": errors}


def deep_read_node(state: AgentState) -> dict:
    if not state.get("need_deep_read"):
        return {"deep_read_result": {}, "tool_calls": [], "errors": []}

    parts = []
    if state.get("financial_metrics"):
        parts.append("[财务指标]\n" + json.dumps(state.get("financial_metrics"), ensure_ascii=False, indent=2))
    if state.get("rag_result"):
        parts.append(f"[财报数据]\n{state['rag_result']}")

    context = "\n\n".join(parts) if parts else "无背景数据"
    errors = []
    api_key = state.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")

    # 阶段 1: 摘要与指标提取
    s1_result = {}
    s1_model = QUALITY_CASCADE[0]
    _s1_usage = None
    try:
        raw1, s1_model, _s1_usage = _invoke_with_cascade(
            [
                SystemMessage(content=DEEP_READ_STAGE1_SYSTEM),
                HumanMessage(content=f"内容：\n{context}")
            ],
            api_key,
            DEEP_READ_S1_CASCADE
        )
        s1_result = _parse_json_from_text(raw1)
    except Exception as exc:
        errors.append({"node": "deep_read_node", "tool": "s1_llm", "message": str(exc), "retryable": False})

    # 阶段 2: 批判质疑
    s2_result = {}
    s2_model = TIER_TOP
    _s2_usage = None
    if s1_result:
        try:
            raw2, s2_model, _s2_usage = _invoke_with_cascade(
                [
                    SystemMessage(content=DEEP_READ_STAGE2_SYSTEM),
                    HumanMessage(content=f"核心内容：\n{json.dumps(s1_result, ensure_ascii=False)}")
                ],
                api_key,
                DEEP_READ_S2_CASCADE
            )
            s2_result = _parse_json_from_text(raw2)
        except Exception as exc:
            errors.append({"node": "deep_read_node", "tool": "s2_llm", "message": str(exc), "retryable": False})

    deep_read_result = {**s1_result, **s2_result}
    tool_calls = []
    if deep_read_result:
        tool_calls.append({"tool_name": "llm", "tool_args": {"node": "deep_read", "model": f"S1:{s1_model} S2:{s2_model}"}})
    _dr_usage_parts = [u for u in (_s1_usage, _s2_usage) if u]
    if _dr_usage_parts:
        tool_calls.append({"node": "deep_read", "token_usage": {
            "prompt_tokens":     sum(u["prompt_tokens"] for u in _dr_usage_parts),
            "completion_tokens": sum(u["completion_tokens"] for u in _dr_usage_parts),
            "total_tokens":      sum(u["total_tokens"] for u in _dr_usage_parts),
        }})

    return {"deep_read_result": deep_read_result, "tool_calls": tool_calls, "errors": errors}


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


def _invoke_with_cascade(messages: list, api_key: str, tiers: list, temperature: float = 0.1) -> tuple[str, str, dict | None]:
    """Try each model tier in order on rate-limit errors. Returns (text, model_used, usage_or_None)."""
    last_exc: Exception = Exception("no tiers provided")
    for model in tiers:
        try:
            llm = ChatGroq(api_key=api_key, model=model, temperature=temperature)
            resp = llm.invoke(messages)
            text = _extract_text(resp.content)
            um = getattr(resp, "usage_metadata", None)
            usage = {
                "prompt_tokens":     um.get("input_tokens", 0),
                "completion_tokens": um.get("output_tokens", 0),
                "total_tokens":      um.get("total_tokens", 0),
            } if um else None
            return text, model, usage
        except Exception as exc:
            if any(k in str(exc).lower() for k in RATE_LIMIT_KEYWORDS):
                logger.warning("Groq %s rate-limited, trying next tier", model)
                last_exc = exc
                continue
            raise
    raise last_exc


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


def _extract_comparison_dimensions(text: str) -> List[str]:
    dimension_keywords = [
        ("估值", ("估值", "pe", "pb", "valuation")),
        ("技术面", ("技术面", "走势", "趋势", "chart", "technical")),
        ("新闻情绪", ("新闻情绪", "情绪", "新闻", "消息", "news", "sentiment")),
        ("财务", ("财务", "财报", "营收", "利润", "earnings", "financial")),
        ("风险", ("风险", "隐患", "担忧", "risk")),
    ]
    lowered = text.lower()
    dimensions = []
    for dimension, keywords in dimension_keywords:
        if any(keyword in lowered for keyword in keywords):
            dimensions.append(dimension)
    return dimensions


def _infer_plan_from_text(state: dict, rag_available: bool) -> dict:
    user_input = state["user_input"]
    text = user_input.lower()
    tickers = _extract_tickers(user_input)
    comparison_dimensions = _extract_comparison_dimensions(user_input)
    need_comparison = len(tickers) >= 2 or any(
        kw in text for kw in ("对比", "比较", "pk", "哪个更好", "compare")
    )

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
        kw in text for kw in (
            "pdf", "document", "report", "earnings", "uploaded", "file",
            "知识库", "向量数据库", "财报", "营收", "利润", "年报",
        )
    )
    need_email = any(kw in text for kw in ("email", "mail", "send"))

    if need_history:
        need_data = True
    if need_comparison and tickers:
        need_data = True
        if not comparison_dimensions or "新闻情绪" in comparison_dimensions:
            need_news = True

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

    use_financial_report = state.get("use_financial_report", False) or any(
        kw in text for kw in ("财报", "年报", "10-k", "20-f", "上传pdf", "读财报")
    )
    need_scoring = any(
        kw in text for kw in ("综合", "分析", "建议", "评级", "值不值", "买入", "卖出", "投资")
    ) or len(agents) >= 2
    need_risk = any(
        kw in text for kw in ("风险", "隐患", "担忧", "有什么问题", "risk")
    )
    need_deep_read = any(
        kw in text for kw in ("精读", "深度分析", "质疑", "批判", "论文", "研究报告")
    )
    need_reflection = need_scoring or any(
        kw in text for kw in ("深度分析", "详细", "严谨", "全面")
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
        "need_scoring": need_scoring,
        "need_risk": need_risk,
        "need_comparison": need_comparison,
        "need_hypothesis": any(kw in text for kw in ("如果", "假设", "若", "what if", "假如", "倘若")),
        "need_deep_read": need_deep_read,
        "comparison_dimensions": comparison_dimensions,
        "need_reflection": need_reflection,
        "use_financial_report": use_financial_report,
        "pdf_path": state.get("pdf_path") or _extract_pdf_path(user_input),
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
    need_scoring = bool(plan.get("need_scoring", False))
    need_risk = bool(plan.get("need_risk", False)) or any(
        kw in user_input.lower() for kw in ("风险", "隐患", "担忧", "有什么问题", "risk")
    )
    need_hypothesis = bool(plan.get("need_hypothesis", False)) or any(
        kw in user_input.lower() for kw in ("如果", "假设", "若", "what if", "假如", "倘若")
    )
    need_deep_read = bool(plan.get("need_deep_read", False)) or any(
        kw in user_input.lower() for kw in ("精读", "深度分析", "质疑", "批判", "论文", "研究报告")
    )
    comparison_dimensions = [
        str(item) for item in (plan.get("comparison_dimensions") or []) if str(item).strip()
    ] or _extract_comparison_dimensions(user_input)
    need_comparison = bool(plan.get("need_comparison", False)) or len(tickers) >= 2 or any(
        kw in user_input.lower() for kw in ("对比", "比较", "pk", "哪个更好", "compare")
    )
    need_reflection = (
        bool(plan.get("need_reflection", False))
        or need_scoring
        or need_deep_read
        or any(kw in user_input.lower() for kw in ("深度分析", "详细", "严谨", "全面"))
    )

    if need_comparison and tickers and "data" not in agents:
        agents.append("data")
    if need_comparison and tickers and ("新闻情绪" in comparison_dimensions or not comparison_dimensions) and "news" not in agents:
        agents.append("news")
        if not news_query:
            news_query = user_input

    # 只要 rag_available 且问题涉及财报分析，强制触发 rag（不依赖 LLM 判断）
    _financial_kws = ("财报", "营收", "利润", "年报", "季报", "业绩", "earnings", "revenue",
                      "report", "10-k", "20-f", "知识库", "向量")
    if rag_available and "rag" not in agents and any(kw in user_input.lower() for kw in _financial_kws):
        agents.append("rag")
        if not rag_query:
            rag_query = user_input

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
        "need_scoring": need_scoring,
        "need_risk": need_risk,
        "need_comparison": need_comparison,
        "need_hypothesis": need_hypothesis,
        "need_deep_read": need_deep_read,
        "comparison_dimensions": comparison_dimensions,
        "need_reflection": need_reflection,
        "use_financial_report": use_financial_report,
    }


def parse_node(state: AgentState) -> dict:
    api_key = state.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
    history_ctx = state.get("chat_history_text", "")
    planner_error = None
    rag_available = bool(state.get("rag_available", False))

    model_used = QUALITY_CASCADE[0]

    history_prefix = f"History:\n{history_ctx}\n" if history_ctx else ""
    plan_messages = [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=f"{history_prefix}User query: {state['user_input']}"),
    ]

    _parse_usage = None
    try:
        raw_text, model_used, _parse_usage = _invoke_with_cascade(plan_messages, api_key, QUALITY_CASCADE)
        raw_plan = _parse_plan(raw_text)
    except Exception as exc:
        planner_error = str(exc)
        raw_plan = _infer_plan_from_text(state, rag_available)

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
        "need_scoring": bool(plan.get("need_scoring", False)),
        "need_risk": bool(plan.get("need_risk", False)),
        "need_comparison": bool(plan.get("need_comparison", False)),
        "need_hypothesis": bool(plan.get("need_hypothesis", False)),
        "need_deep_read": bool(plan.get("need_deep_read", False)),
        "comparison_dimensions": plan.get("comparison_dimensions", []),
        "need_reflection": bool(plan.get("need_reflection", False)),
        "use_financial_report": state.get("use_financial_report", False) or bool(plan.get("use_financial_report", False)),
        "pdf_path": state.get("pdf_path") or plan.get("pdf_path") or _extract_pdf_path(state["user_input"]),
        "tool_calls": [{"tool_name": "llm", "tool_args": {"node": "parse", "model": model_used}}]
            + ([{"node": "parse", "token_usage": _parse_usage}] if _parse_usage else []),
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
                tool_calls.append({"tool_name": "llm", "tool_args": {"node": "data", "model": DATA_AGENT_MODEL}})
                _um = getattr(resp, "usage_metadata", None)
                if _um:
                    tool_calls.append({"node": "data", "token_usage": {
                        "prompt_tokens":     _um.get("input_tokens", 0),
                        "completion_tokens": _um.get("output_tokens", 0),
                        "total_tokens":      _um.get("total_tokens", 0),
                    }})
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
            tool_calls.append({"tool_name": "llm", "tool_args": {"node": "news", "model": NEWS_AGENT_MODEL}})
            _um = getattr(resp, "usage_metadata", None)
            if _um:
                tool_calls.append({"node": "news", "token_usage": {
                    "prompt_tokens":     _um.get("input_tokens", 0),
                    "completion_tokens": _um.get("output_tokens", 0),
                    "total_tokens":      _um.get("total_tokens", 0),
                }})
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
        err = {"node": "rag_node", "tool": "search_documents", "message": str(exc), "retryable": True}
        return {
            "rag_result": "",
            "tool_calls": [],
            "errors": [err],
        }

    tool_calls = [{"tool_name": "search_documents", "tool_args": {"query": query}}]

    # LLM post-reasoning: 从检索结果中提炼关键财务信息
    analysis = f"[Document Retrieval]\n{raw_result}"  # 默认 fallback
    _rag_usage = None
    try:
        analysis_text, rag_model, _rag_usage = _invoke_with_cascade(
            [SystemMessage(content=RAG_AGENT_SYSTEM), HumanMessage(content=f"检索到的财报段落：\n{raw_result}")],
            state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            QUALITY_CASCADE,
        )
        analysis_text = analysis_text.strip()
        if analysis_text:
            analysis = f"[财报分析 by rag_agent]\n{analysis_text}\n\n[原始检索结果]\n{raw_result}"
            tool_calls.append({"tool_name": "llm", "tool_args": {"node": "rag", "model": rag_model}})
            if _rag_usage:
                tool_calls.append({"node": "rag", "token_usage": _rag_usage})
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
    if not state.get("need_scoring"):
        return {"scoring_result": {}, "tool_calls": [], "errors": []}

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
    scoring_model = QUALITY_CASCADE[0]
    _scoring_usage = None

    try:
        raw, scoring_model, _scoring_usage = _invoke_with_cascade(
            [SystemMessage(content=SCORING_SYSTEM), HumanMessage(content=f"请对以下数据进行多维度评分：\n\n{context}")],
            state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            QUALITY_CASCADE,
        )
        raw = raw.strip()
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
        [{"tool_name": "llm", "tool_args": {"node": "scoring", "model": scoring_model}}]
        if scoring else []
    )
    if _scoring_usage:
        tool_calls.append({"node": "scoring", "token_usage": _scoring_usage})
    return {"scoring_result": scoring, "tool_calls": tool_calls, "errors": errors}


def risk_node(state: AgentState) -> dict:
    if not state.get("need_risk"):
        return {"risk_result": {}, "tool_calls": [], "errors": []}

    parts = []
    if state.get("financial_metrics"):
        parts.append("[财务指标]\n" + json.dumps(state.get("financial_metrics"), ensure_ascii=False, indent=2))
    if state.get("stock_data"):
        parts.append(f"[技术面数据]\n{state['stock_data']}")
    if state.get("news"):
        parts.append(f"[新闻信息]\n{state['news']}")

    if not parts:
        return {"risk_result": {}, "tool_calls": [], "errors": []}

    context = "\n\n".join(parts)
    errors = []
    risk_result = {}
    risk_model = TIER_TOP
    _risk_usage = None

    try:
        raw, risk_model, _risk_usage = _invoke_with_cascade(
            [SystemMessage(content=RISK_SYSTEM), HumanMessage(content=f"请分析以下数据中的风险：\n\n{context}")],
            state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            RISK_MODEL_CASCADE,
        )
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        try:
            risk_result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    risk_result = json.loads(match.group())
                except Exception as parse_exc:
                    errors.append({
                        "node": "risk_node",
                        "tool": "json_parse",
                        "message": str(parse_exc),
                        "retryable": False,
                    })
            else:
                errors.append({
                    "node": "risk_node",
                    "tool": "json_parse",
                    "message": f"No JSON found in response: {raw[:200]}",
                    "retryable": False,
                })
    except Exception as exc:
        errors.append({
            "node": "risk_node",
            "tool": "llm_risk",
            "message": str(exc),
            "retryable": False,
        })

    tool_calls = (
        [{"tool_name": "llm", "tool_args": {"node": "risk", "model": risk_model}}]
        if risk_result else []
    )
    if _risk_usage:
        tool_calls.append({"node": "risk", "token_usage": _risk_usage})
    return {"risk_result": risk_result, "tool_calls": tool_calls, "errors": errors}


def comparison_node(state: AgentState) -> dict:
    if not state.get("need_comparison"):
        return {"comparison_result": {}, "tool_calls": [], "errors": []}

    parts = []
    if state.get("tickers"):
        parts.append("[股票列表]\n" + ", ".join(state.get("tickers") or []))
    if state.get("comparison_dimensions"):
        parts.append("[指定维度]\n" + json.dumps(state.get("comparison_dimensions"), ensure_ascii=False))
    else:
        parts.append("[指定维度]\n[]（代表全维度）")
    if state.get("stock_data"):
        parts.append(f"[行情与技术面数据]\n{state['stock_data']}")
    if state.get("news"):
        parts.append(f"[新闻信息]\n{state['news']}")

    if not state.get("stock_data") and not state.get("news"):
        return {"comparison_result": {}, "tool_calls": [], "errors": []}

    context = "\n\n".join(parts)
    errors = []
    comparison_result = {}
    comparison_model = TIER_TOP
    _cmp_usage = None

    try:
        raw, comparison_model, _cmp_usage = _invoke_with_cascade(
            [SystemMessage(content=COMPARISON_SYSTEM), HumanMessage(content=f"请对以下股票进行逐项对比：\n\n{context}")],
            state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            COMPARISON_MODEL_CASCADE,
        )
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        try:
            comparison_result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    comparison_result = json.loads(match.group())
                except Exception as parse_exc:
                    errors.append({
                        "node": "comparison_node",
                        "tool": "json_parse",
                        "message": str(parse_exc),
                        "retryable": False,
                    })
            else:
                errors.append({
                    "node": "comparison_node",
                    "tool": "json_parse",
                    "message": f"No JSON found in response: {raw[:200]}",
                    "retryable": False,
                })
    except Exception as exc:
        errors.append({
            "node": "comparison_node",
            "tool": "llm_comparison",
            "message": str(exc),
            "retryable": False,
        })

    tool_calls = (
        [{"tool_name": "llm", "tool_args": {"node": "comparison", "model": comparison_model}}]
        if comparison_result else []
    )
    if _cmp_usage:
        tool_calls.append({"node": "comparison", "token_usage": _cmp_usage})
    return {"comparison_result": comparison_result, "tool_calls": tool_calls, "errors": errors}


def _format_comparison_section(comparison_result: Optional[dict]) -> str:
    if not comparison_result:
        return ""

    rankings = comparison_result.get("rankings") or []
    table = comparison_result.get("comparison_table") or {}
    if not rankings and not table:
        return ""

    lines = ["", "## 对比排名与表格", ""]

    if rankings:
        lines.extend([
            "### 综合排名",
            "",
            "| 排名 | 股票 | 分数 | 摘要 |",
            "|---:|---|---:|---|",
        ])
        for item in rankings:
            rank = item.get("rank", "")
            ticker = str(item.get("ticker", "")).replace("|", "/")
            score = item.get("score", "")
            summary = str(item.get("summary", "")).replace("|", "/")
            lines.append(f"| {rank} | {ticker} | {score} | {summary} |")

    winner = str(comparison_result.get("winner", "")).strip()
    winner_reasoning = str(comparison_result.get("winner_reasoning", "")).strip()
    if winner or winner_reasoning:
        lines.extend(["", f"**胜出方：** {winner or 'N/A'}"])
        if winner_reasoning:
            lines.append(f"**胜出理由：** {winner_reasoning}")

    if table:
        tickers = []
        for values in table.values():
            if isinstance(values, dict):
                for ticker in values:
                    if ticker not in tickers:
                        tickers.append(ticker)

        if tickers:
            lines.extend(["", "### 维度对比", ""])
            header = "| 维度 | " + " | ".join(str(ticker).replace("|", "/") for ticker in tickers) + " |"
            separator = "|---" + "|---" * len(tickers) + "|"
            lines.extend([header, separator])
            for dimension, values in table.items():
                if not isinstance(values, dict):
                    continue
                row = [str(dimension).replace("|", "/")]
                for ticker in tickers:
                    row.append(str(values.get(ticker, "")).replace("|", "/"))
                lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _format_risk_matrix(risk_result: Optional[dict]) -> str:
    if not risk_result:
        return ""

    factors = risk_result.get("risk_factors") or []
    if not factors:
        return ""

    lines = [
        "",
        "## 风险矩阵",
        "",
        "| ID | 类别 | 风险 | 严重性 | 触发条件 |",
        "|---:|---|---|---|---|",
    ]
    for index, item in enumerate(factors, start=1):
        risk_id = item.get("id", index)
        category = str(item.get("category", "")).replace("|", "/")
        title = str(item.get("title", "")).replace("|", "/")
        severity = str(item.get("severity", "")).replace("|", "/")
        trigger = str(item.get("trigger", "")).replace("|", "/")
        reasoning = str(item.get("reasoning", "")).strip()
        if reasoning:
            title = f"{title}<br>{reasoning}" if title else reasoning
        lines.append(f"| {risk_id} | {category} | {title} | {severity} | {trigger} |")

    summary = str(risk_result.get("risk_summary", "")).strip()
    if summary:
        lines.extend(["", f"**整体风险评估：** {summary}"])

    return "\n".join(lines)


def _format_hypothesis_section(hypothesis_result: Optional[dict]) -> str:
    if not hypothesis_result:
        return ""

    hypothesis = hypothesis_result.get("hypothesis", "")
    path = hypothesis_result.get("transmission_path") or []
    scenarios = hypothesis_result.get("scenarios") or []
    conclusion = hypothesis_result.get("conclusion", "")

    if not hypothesis and not scenarios:
        return ""

    lines = ["", "## 假设情景推演", ""]
    if hypothesis:
        lines.append(f"**假设条件：** {hypothesis}")

    if path:
        lines.extend(["", "### 传导路径", ""])
        for i, p in enumerate(path, 1):
            lines.append(f"{i}. {p}")

    if scenarios:
        lines.extend([
            "",
            "### 情景分析报告",
            "",
            "| 情景 | 概率 | 核心影响 | 股价预期 |",
            "|---|---:|---|---|",
        ])
        for s in scenarios:
            name = str(s.get("name", "")).replace("|", "/")
            prob = f"{s.get('probability', '')}%"
            impact = str(s.get("impact", "")).replace("|", "/")
            price = str(s.get("price_impact", "")).replace("|", "/")
            lines.append(f"| {name} | {prob} | {impact} | {price} |")

    if conclusion:
        lines.extend(["", f"**推演结论：** {conclusion}"])

    return "\n".join(lines)


def _format_deep_read_section(deep_read_result: Optional[dict]) -> str:
    if not deep_read_result:
        return ""

    summary = deep_read_result.get("summary", "")
    metrics = deep_read_result.get("key_metrics") or []
    claims = deep_read_result.get("key_claims") or []
    questions = deep_read_result.get("critical_questions") or []

    if not summary and not questions:
        return ""

    lines = ["", "## 深度分析与批判报告", ""]
    if summary:
        lines.extend(["### 核心摘要", summary, ""])

    if metrics:
        lines.extend([
            "### 关键指标提取",
            "",
            "| 指标 | 数值 | 意义 |",
            "|---|---|---|",
        ])
        for m in metrics:
            name = str(m.get("name", "")).replace("|", "/")
            val = str(m.get("value", "")).replace("|", "/")
            sig = str(m.get("significance", "")).replace("|", "/")
            lines.append(f"| {name} | {val} | {sig} |")
        lines.append("")

    if claims:
        lines.extend(["### 核心论点", ""])
        for c in claims:
            lines.append(f"- {c}")
        lines.append("")

    if questions:
        lines.extend([
            "### 深度质疑清单",
            "",
            "| 质疑问题 | 依据 | 严重性 |",
            "|---|---|---|",
        ])
        for q in questions:
            ques = str(q.get("question", "")).replace("|", "/")
            basis = str(q.get("basis", "")).replace("|", "/")
            sev = str(q.get("severity", "")).replace("|", "/")
            lines.append(f"| {ques} | {basis} | {sev} |")
        lines.append("")

    rel_score = deep_read_result.get("reliability_score")
    if rel_score is not None:
        lines.append(f"**综合可信度评分：** {rel_score}/10")

    conclusion = deep_read_result.get("conclusion")
    if conclusion:
        lines.append(f"**分析结论：** {conclusion}")

    return "\n".join(lines)


def _parse_json_from_text(text: str) -> dict:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except Exception:
            return {}


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
        stock_text = state["stock_data"]
        if "[原始数据]" in stock_text:
            stock_text = stock_text.split("[原始数据]")[0].strip()
        parts.append(stock_text)
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

    # 构建假设推演摘要（如有）
    hypothesis_result = state.get("hypothesis_result") or {}
    hypothesis_prompt_ctx = ""
    if hypothesis_result:
        h_conclusion = hypothesis_result.get("conclusion", "")
        h_scenarios = hypothesis_result.get("scenarios", [])
        scenarios_text = "\n".join(
            f"- {s.get('name')}（概率{s.get('probability')}%）：{s.get('impact')} 股价影响：{s.get('price_impact')}"
            for s in h_scenarios
        )
        hypothesis_prompt_ctx = f"[假设推演结论]\n{h_conclusion}\n情景：\n{scenarios_text}\n"

    error_items = state.get("errors") or []
    error_lines = [
        f"- {item.get('node', 'unknown')}/{item.get('tool', 'unknown')}: {item.get('message', '')}"
        for item in error_items
    ]
    error_section = "\n".join(error_lines) if error_lines else "- none"

    if state.get("chat_history_text"):
        history_lines = state["chat_history_text"].strip().split("\n")
        truncated = "\n".join(history_lines[-12:])
        history_section = f"\nHistory（最近3轮）:\n{truncated}\n"
    else:
        history_section = ""
    prompt = (
        f"User query: {state['user_input']}\n"
        f"{history_section}\n"
        f"{scoring_instruction}"
        f"Gathered data:\n{gathered_data}\n\n"
        f"{scoring_section}"
        f"{hypothesis_prompt_ctx}"
        f"Tool failures:\n{error_section}\n\n"
        "Write the final answer using only the available information."
    )
    messages = [SystemMessage(content=REPORT_SYSTEM), HumanMessage(content=prompt)]

    _report_usage = None

    def call_groq() -> tuple[str, str, dict | None]:
        text, model, usage = _invoke_with_cascade(messages, groq_api_key, QUALITY_CASCADE)
        return text, model, usage

    if dev_mode or gemini_exhausted:
        try:
            response, final_model, _report_usage = call_groq()
            if not response.strip():
                response = "⚠️ 报告生成失败：模型返回空响应，请重试。"
                final_model = "none"
        except Exception as exc:
            response = f"⚠️ 报告生成失败，所有 Groq 模型均不可用：{exc}"
            final_model = "none"
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
                _um = getattr(resp, "usage_metadata", None)
                if _um:
                    _report_usage = {
                        "prompt_tokens":     _um.get("input_tokens", 0),
                        "completion_tokens": _um.get("output_tokens", 0),
                        "total_tokens":      _um.get("total_tokens", 0),
                    }
            else:
                response, final_model, _report_usage = call_groq()
        except Exception as exc:
            err = str(exc)
            # 以下情况 fallback 到 Groq：配额耗尽、限速、服务不可用
            if any(k in err for k in ("RESOURCE_EXHAUSTED", "UNAVAILABLE") + RATE_LIMIT_KEYWORDS):
                response, final_model, _report_usage = call_groq()
                new_gemini_exhausted = "RESOURCE_EXHAUSTED" in err
            else:
                raise

        if response is None:
            response, final_model, _report_usage = call_groq()

    comparison_section = _format_comparison_section(state.get("comparison_result"))
    if comparison_section:
        response = f"{response.rstrip()}\n\n{comparison_section}"

    risk_matrix = _format_risk_matrix(state.get("risk_result"))
    if risk_matrix:
        response = f"{response.rstrip()}\n\n{risk_matrix}"

    hypothesis_section = _format_hypothesis_section(state.get("hypothesis_result"))
    if hypothesis_section:
        response = f"{response.rstrip()}\n\n{hypothesis_section}"

    deep_read_section = _format_deep_read_section(state.get("deep_read_result"))
    if deep_read_section:
        response = f"{response.rstrip()}\n\n{deep_read_section}"

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

    report_model_record = {"tool_name": "llm", "tool_args": {"node": "report", "model": final_model}}
    _report_token_entry = [{"node": "report", "token_usage": _report_usage}] if _report_usage else []
    return {
        "report": response,
        "final_report": response,
        "email_status": email_status,
        "final_model": final_model,
        "gemini_exhausted": new_gemini_exhausted,
        "tool_calls": [report_model_record] + _report_token_entry + email_tool_calls,
        "errors": email_errors,
    }


def reflection_node(state: AgentState) -> dict:
    report = state.get("report") or state.get("final_report") or ""
    if not state.get("need_reflection") or not report.strip():
        return {
            "final_report": report,
            "reflection_result": None,
            "tool_calls": [],
            "errors": [],
        }

    reflection_model = TIER_TOP
    _ref_usage = None
    errors = []
    try:
        raw, reflection_model, _ref_usage = _invoke_with_cascade(
            [SystemMessage(content=REFLECTION_SYSTEM), HumanMessage(content=f"股票分析报告：\n\n{report}")],
            state.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
            REFLECTION_MODEL_CASCADE,
        )
        parsed = _parse_json_from_text(raw)
        if not parsed:
            errors.append({
                "node": "reflection_node",
                "tool": "json_parse",
                "message": f"No JSON found in response: {raw[:200]}",
                "retryable": False,
            })
            return {
                "final_report": report,
                "reflection_result": None,
                "tool_calls": [],
                "errors": errors,
            }

        revised_report = (
            parsed.get("revised_report")
            or parsed.get("final_report")
            or parsed.get("report")
            or report
        )
        reflection_payload = {
            "issues": parsed.get("issues") or [],
            "severity": parsed.get("severity", ""),
            "revised_sections": parsed.get("revised_sections", ""),
        }
        _ref_tc = [{"tool_name": "llm", "tool_args": {"node": "reflection", "model": reflection_model}}]
        if _ref_usage:
            _ref_tc.append({"node": "reflection", "token_usage": _ref_usage})
        return {
            "final_report": str(revised_report),
            "reflection_result": json.dumps(reflection_payload, ensure_ascii=False),
            "tool_calls": _ref_tc,
            "errors": [],
        }
    except Exception as exc:
        errors.append({
            "node": "reflection_node",
            "tool": "llm_reflection",
            "message": str(exc),
            "retryable": False,
        })
        return {
            "final_report": report,
            "reflection_result": None,
            "tool_calls": [],
            "errors": errors,
        }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("parse_node", parse_node)
    builder.add_node("financial_report_node", financial_report_node)
    builder.add_node("data_node", data_node)
    builder.add_node("news_node", news_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("scoring_node", scoring_node)
    builder.add_node("risk_node", risk_node)
    builder.add_node("comparison_node", comparison_node)
    builder.add_node("hypothesis_node", hypothesis_node)
    builder.add_node("deep_read_node", deep_read_node)
    builder.add_node("reflection_node", reflection_node)
    builder.add_node("report_node", report_node)

    builder.add_edge(START, "parse_node")

    # parse_node → financial_report_node（条件）或直接进并行节点
    def route_after_parse(state: AgentState):
        if state.get("use_financial_report"):
            return "financial_report_node"
        return _parallel_targets(state) or _analysis_targets(state) or "report_node"

    builder.add_conditional_edges(
        "parse_node",
        route_after_parse,
        ["financial_report_node", "data_node", "news_node", "rag_node", "scoring_node", "risk_node", "comparison_node", "hypothesis_node", "deep_read_node", "reflection_node", "report_node"],
    )

    # financial_report_node 完成后进 deep_read_node 或并行节点
    def route_after_financial(state: AgentState):
        if state.get("need_deep_read"):
            return "deep_read_node"
        targets = _parallel_targets(state)
        return targets if targets else _analysis_targets(state)

    builder.add_conditional_edges(
        "financial_report_node",
        route_after_financial,
        ["deep_read_node", "data_node", "news_node", "rag_node", "scoring_node", "risk_node", "comparison_node", "hypothesis_node", "reflection_node"],
    )

    # deep_read_node 完成后进并行节点
    def route_after_deep_read(state: AgentState):
        targets = _parallel_targets(state)
        return targets if targets else _analysis_targets(state)

    builder.add_conditional_edges(
        "deep_read_node",
        route_after_deep_read,
        ["data_node", "news_node", "rag_node", "scoring_node", "risk_node", "comparison_node", "hypothesis_node", "reflection_node"],
    )

    # 并行采集节点完成后进入分析节点
    builder.add_edge("data_node", "scoring_node")
    builder.add_edge("news_node", "scoring_node")
    builder.add_edge("rag_node", "scoring_node")
    builder.add_edge("data_node", "risk_node")
    builder.add_edge("news_node", "risk_node")
    builder.add_edge("rag_node", "risk_node")
    builder.add_edge("data_node", "comparison_node")
    builder.add_edge("news_node", "comparison_node")
    builder.add_edge("rag_node", "comparison_node")
    builder.add_edge("data_node", "hypothesis_node")
    builder.add_edge("news_node", "hypothesis_node")
    builder.add_edge("rag_node", "hypothesis_node")
    builder.add_edge("data_node", "reflection_node")
    builder.add_edge("news_node", "reflection_node")
    builder.add_edge("rag_node", "reflection_node")

    # 所有分析节点汇入 report_node
    builder.add_edge(["scoring_node", "risk_node", "comparison_node", "hypothesis_node", "reflection_node"], "report_node")

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


def _analysis_targets(state: AgentState) -> list:
    targets = []
    if state.get("need_scoring") or state.get("use_financial_report"):
        targets.append("scoring_node")
    if state.get("need_risk"):
        targets.append("risk_node")
    if state.get("need_comparison"):
        targets.append("comparison_node")
    if state.get("need_hypothesis"):
        targets.append("hypothesis_node")
    if state.get("need_deep_read"):
        targets.append("deep_read_node")
    if state.get("need_reflection"):
        targets.append("reflection_node")
    return targets


graph = build_graph()
