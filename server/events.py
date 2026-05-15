"""SSE event type constants — must mirror web/src/types/sse.ts."""
from __future__ import annotations

# Lifecycle / routing
NODE_START    = "node.start"      # {node, label}
NODE_COMPLETE = "node.complete"   # {node, payload?}
TOOL_CALL     = "tool.call"       # {step, tool_name, tool_args, retries}
ERROR         = "error"           # {node, tool, message}

# Report streaming
REPORT_TOKEN   = "report.token"   # {delta}
REPORT_SECTION = "report.section" # {type, markdown}

# Side effects
CHART = "chart"                   # {path}

# Terminal
DONE = "done"                     # {final_model, final_report, email_status, gemini_exhausted, elapsed}

# Human-readable label per LangGraph node (mirrors app.py:1315-1327)
NODE_LABELS: dict[str, str] = {
    "parse_node":            "正在分析问题，制定调度计划…",
    "data_node":             "正在获取股票数据…",
    "news_node":             "正在搜索最新新闻…",
    "rag_node":              "正在检索财报文档…",
    "scoring_node":          "正在计算综合评分…",
    "risk_node":             "正在分析风险因素…",
    "comparison_node":       "正在做横向对比…",
    "hypothesis_node":       "正在做情景推演…",
    "deep_read_node":        "正在精读财报…",
    "financial_report_node": "正在解析财报 PDF…",
    "report_node":           "正在生成分析报告…",
    "reflection_node":       "正在自我反思与修订…",
}

# Structural sections appended by report_node (graph.py:1701-1719)
SECTION_TYPES = ("comparison", "risk_matrix", "hypothesis", "deep_read")
