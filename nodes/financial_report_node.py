"""财报精读节点：下载/读取财报 PDF，Map-Reduce 提取财务指标和风险信号。"""

import base64
import json
import logging
import os
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

FAST_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
QUALITY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
CHUNK_TOKENS = 3000  # approximate chars per chunk (1 token ≈ 4 chars → ~12000 chars)
CHUNK_CHARS = CHUNK_TOKENS * 4

MAP_SYSTEM = """你是财报分析师助手。你会收到财报的一个文本分块。
请从中提取所有关键财务数据和风险信息，输出纯 JSON：
{
  "metrics": {"revenue": "...", "net_profit": "...", "gross_margin": "...", "debt_ratio": "...", "eps": "...", "operating_cash_flow": "...", "revenue_yoy": "...", "net_profit_yoy": "..."},
  "risks": ["风险描述1", "风险描述2"],
  "citations": [{"text": "原文片段", "chunk_id": "chunk_XXX"}]
}
没有的字段填 null，只输出 JSON，禁止其他文字。"""

REDUCE_SYSTEM = """你是资深财报分析师。你会收到多个财报分块的提取结果（JSON 列表）。
请合并去重，输出一份最终的结构化财务摘要，纯 JSON 格式：
{
  "financial_metrics": {
    "revenue": "...", "revenue_yoy": "...",
    "net_profit": "...", "net_profit_yoy": "...",
    "gross_margin": "...", "debt_ratio": "...",
    "eps": "...", "operating_cash_flow": "..."
  },
  "risk_signals": ["风险1", "风险2"],
  "report_citations": [{"text": "...", "page": null, "chunk_id": "chunk_XXX"}]
}
数字保留原文单位，禁止换算，只输出 JSON。"""

VISION_SYSTEM = """你是财报表格解析助手。图片是财报的一页，请提取其中所有财务数据，输出纯 JSON：
{"metrics": {...}, "risks": [], "citations": []}"""


def _invoke_groq(messages: list, api_key: str, model: str, temperature: float = 0.1) -> str:
    """Call Groq with automatic fallback to FALLBACK_MODEL on rate-limit errors."""
    try:
        llm = ChatGroq(api_key=api_key, model=model, temperature=temperature)
        resp = llm.invoke(messages)
        return _extract_text(resp.content)
    except Exception as exc:
        err = str(exc).lower()
        if model != FALLBACK_MODEL and any(
            k in err for k in ("429", "rate_limit", "rate limit", "503", "over_capacity", "model_overloaded")
        ):
            logger.warning("Groq %s rate-limited, falling back to %s", model, FALLBACK_MODEL)
            llm_fb = ChatGroq(api_key=api_key, model=FALLBACK_MODEL, temperature=temperature)
            return _extract_text(llm_fb.invoke(messages).content)
        raise


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


def _parse_json_safe(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def _read_pdf(pdf_path: str) -> tuple[list[str], list]:
    """Extract text chunks and raw pages from PDF. Returns (chunks, pages)."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required: pip install pdfplumber")

    full_text = ""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            # Try extracting tables too
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    if row:
                        table_text += " | ".join(str(cell or "") for cell in row) + "\n"
            combined = page_text + ("\n[表格]\n" + table_text if table_text else "")
            full_text += combined + "\n"
            pages.append({"index": i, "text": combined, "raw_page": page})

    # Split into chunks of ~CHUNK_CHARS characters
    chunks = []
    for start in range(0, len(full_text), CHUNK_CHARS):
        chunks.append(full_text[start: start + CHUNK_CHARS])

    return chunks, pages


def _vision_fallback(page, api_key: str) -> Optional[dict]:
    """Use gpt-oss-120b vision to parse a page image when text extraction fails."""
    try:
        import pdfplumber
        import PIL.Image
        import io

        img = page.to_image(resolution=150).original
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        llm = ChatGroq(api_key=api_key, model=QUALITY_MODEL, temperature=0.0)
        resp = llm.invoke([
            SystemMessage(content=VISION_SYSTEM),
            HumanMessage(content=[
                {"type": "text", "text": "请解析此财报页面中的财务数据："},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ])
        return _parse_json_safe(_extract_text(resp.content))
    except Exception as exc:
        logger.warning("Vision fallback failed: %s", exc)
        return None


def _map_chunk(chunk: str, chunk_id: str, api_key: str) -> dict:
    """Extract financial data from a single chunk via LLM (Map phase)."""
    messages = [
        SystemMessage(content=MAP_SYSTEM),
        HumanMessage(content=f"[{chunk_id}]\n{chunk}"),
    ]
    try:
        raw = _invoke_groq(messages, api_key, FAST_MODEL)
        result = _parse_json_safe(raw)
        if result:
            # Stamp chunk_id on citations
            for c in result.get("citations", []):
                c["chunk_id"] = chunk_id
            return result
    except Exception as exc:
        logger.warning("Map chunk %s failed: %s", chunk_id, exc)
    return {"metrics": {}, "risks": [], "citations": []}


def _reduce(map_results: list[dict], api_key: str) -> dict:
    """Merge all chunk results into a final structured summary (Reduce phase)."""
    serialized = json.dumps(map_results, ensure_ascii=False)
    messages = [
        SystemMessage(content=REDUCE_SYSTEM),
        HumanMessage(content=f"以下是各分块提取结果（JSON 列表），请合并：\n{serialized}"),
    ]
    try:
        raw = _invoke_groq(messages, api_key, QUALITY_MODEL)
        result = _parse_json_safe(raw)
        if result:
            return result
    except Exception as exc:
        logger.error("Reduce phase failed: %s", exc)

    # Fallback: naive merge
    merged_metrics = {}
    merged_risks = []
    merged_citations = []
    for r in map_results:
        for k, v in (r.get("metrics") or {}).items():
            if v and k not in merged_metrics:
                merged_metrics[k] = v
        merged_risks.extend(r.get("risks") or [])
        merged_citations.extend(r.get("citations") or [])
    return {
        "financial_metrics": merged_metrics,
        "risk_signals": list(dict.fromkeys(merged_risks)),
        "report_citations": merged_citations,
    }


def financial_report_node(state: dict) -> dict:
    """条件触发节点：读取/下载财报 PDF，Map-Reduce 提取财务指标、风险信号和引用。"""
    if not state.get("use_financial_report"):
        return {
            "financial_metrics": None,
            "risk_signals": None,
            "report_citations": None,
        }

    api_key = state.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
    errors = []

    # Determine PDF path
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        tickers = state.get("tickers") or []
        ticker = tickers[0] if tickers else state.get("ticker", "")
        if not ticker:
            return {
                "financial_metrics": None,
                "risk_signals": None,
                "report_citations": None,
                "errors": [{"node": "financial_report_node", "tool": "fetch", "message": "No ticker or pdf_path", "retryable": False}],
            }
        try:
            # Detect market by ticker suffix
            t_upper = ticker.upper()
            if t_upper.endswith(".SS") or t_upper.endswith(".SZ") or t_upper.endswith(".HK"):
                from tools.cn_report_fetcher import fetch_cn_report
                pdf_path = fetch_cn_report(ticker)
            else:
                from tools.sec_fetcher import fetch_sec_filing
                pdf_path = fetch_sec_filing(ticker)
        except Exception as exc:
            return {
                "financial_metrics": None,
                "risk_signals": None,
                "report_citations": None,
                "errors": [{"node": "financial_report_node", "tool": "fetch", "message": str(exc), "retryable": True}],
            }

    # Parse PDF
    try:
        chunks, pages = _read_pdf(pdf_path)
    except Exception as exc:
        return {
            "financial_metrics": None,
            "risk_signals": None,
            "report_citations": None,
            "errors": [{"node": "financial_report_node", "tool": "pdfplumber", "message": str(exc), "retryable": False}],
        }

    # Map phase
    map_results = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i:03d}"
        result = _map_chunk(chunk, chunk_id, api_key)
        # Vision fallback for empty chunks (likely scanned pages)
        if not result.get("metrics") and i < len(pages):
            vision_result = _vision_fallback(pages[i].get("raw_page"), api_key)
            if vision_result:
                result = vision_result
        map_results.append(result)

    # Reduce phase
    final = _reduce(map_results, api_key)

    return {
        "financial_metrics": final.get("financial_metrics"),
        "risk_signals": final.get("risk_signals"),
        "report_citations": final.get("report_citations"),
        "tool_calls": [{"tool_name": "llm", "tool_args": {"node": "financial_report", "model": f"Map:{FAST_MODEL} → Reduce:{QUALITY_MODEL}"}}],
        "errors": errors,
    }
