"""POST /api/analyze — SSE stream wrapping graph.stream.

M1: returns a fixed mock event sequence so the front-end can be wired up before
the graph integration lands in M2.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server import events as ev
from server.sse import EventEmitter

router = APIRouter()


class AnalyzeRequest(BaseModel):
    user_input: str
    chat_history: list[dict[str, str]] = []  # [{role, content}]
    dev_mode: bool = False
    gemini_exhausted: bool = False
    image_b64: Optional[str] = None


def _mock_run(emitter: EventEmitter, req: AnalyzeRequest) -> None:
    """Deterministic mock event chain — drop-in replacement until M2 wires graph.stream."""
    try:
        for node in ("parse_node", "data_node", "news_node"):
            emitter.emit(ev.NODE_START, {"node": node, "label": ev.NODE_LABELS[node]})
            time.sleep(0.15)
            emitter.emit(ev.NODE_COMPLETE, {"node": node, "payload": None})

        emitter.emit(ev.TOOL_CALL, {
            "step": 1, "tool_name": "get_stock_data",
            "tool_args": {"ticker": "NVDA"}, "retries": 0,
        })

        emitter.emit(ev.NODE_START, {"node": "report_node", "label": ev.NODE_LABELS["report_node"]})
        for chunk in ("# 分析报告\n\n", "这是 ", "一个 ", "**mock** ", "流式输出 ", "示例。\n"):
            emitter.emit(ev.REPORT_TOKEN, {"delta": chunk})
            time.sleep(0.05)
        emitter.emit(ev.NODE_COMPLETE, {"node": "report_node", "payload": None})

        emitter.emit(ev.DONE, {
            "final_model":      "mock",
            "final_report":     "# 分析报告\n\n这是 一个 **mock** 流式输出 示例。\n",
            "email_status":     "",
            "gemini_exhausted": False,
            "elapsed":          1.0,
        })
    except Exception as exc:  # pragma: no cover - defensive
        emitter.emit(ev.ERROR, {"node": "mock", "tool": "mock", "message": str(exc)})
    finally:
        emitter.close()


@router.post("/analyze")
async def analyze(req: AnalyzeRequest) -> StreamingResponse:
    emitter = EventEmitter()

    # Default: run the real LangGraph pipeline. Set STOCKAI_USE_MOCK=1 to keep the M1 mock
    # (useful for front-end work without paying API quota).
    from server.sse import run_in_thread
    if os.getenv("STOCKAI_USE_MOCK") == "1":
        run_in_thread(lambda: _mock_run(emitter, req))
    else:
        from server.routes._analyze_graph import run_graph
        run_in_thread(lambda: run_graph(emitter, req))

    return StreamingResponse(
        emitter.frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx/proxy buffering
            "Connection":    "keep-alive",
        },
    )
