"""Real graph.stream → SSE event bridge. Replaces the mock in analyze.py.

Mirrors the Streamlit threading model from app.py:1310-1352:
  - install per-request streaming callback via graph.set_streaming_cb()
  - run graph.stream(..., stream_mode="updates") in a worker thread
  - re-emit each LangGraph update as SSE events
  - tokens flow through the streaming callback as report.token events
  - structural sections appended after the LLM response are detected by
    the same callback (it fires once more inside report_node).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import graph as graph_module
from history import make_record, save_history
from server import events as ev
from server.events import NODE_LABELS
from server.sse import EventEmitter

logger = logging.getLogger("stockai.server.analyze")

# state.<key> values forwarded inside node.complete.payload (front-end uses these
# for progressive KPI / verdict-card / risk-matrix rendering).
PAYLOAD_KEYS = (
    "financial_metrics",
    "risk_signals",
    "report_citations",
    "scoring_result",
    "risk_result",
    "comparison_result",
    "hypothesis_result",
    "deep_read_result",
)


def _build_initial_state(req: Any) -> dict:
    """Translate API request → AgentState input dict (mirrors app.py:1292-1305)."""
    chat_history_text = ""
    for msg in req.chat_history[-4:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "human"):
            chat_history_text += f"用户: {str(content)[:300]}\n"
        elif role in ("assistant", "ai"):
            chat_history_text += f"助手: {str(content)[:300]}\n"

    # PDF: pick the most recently uploaded one when the user has any.
    pdf_path = None
    use_financial_report = False
    try:
        from server.services.pdf_ingest import load_processed_registry
        registry = load_processed_registry()
    except Exception:
        registry = {}
    if registry:
        first = next(iter(registry.keys()))
        pdf_path = os.path.join("./tmp", first)
        use_financial_report = True

    return {
        "user_input":           req.user_input,
        "chat_history_text":    chat_history_text,
        "groq_api_key":         os.getenv("GROQ_API_KEY", ""),
        "gemini_api_key":       os.getenv("GEMINI_API_KEY", ""),
        "dev_mode":             req.dev_mode,
        "gemini_exhausted":     req.gemini_exhausted,
        "rag_available":        bool(registry),
        "pdf_path":             pdf_path,
        "image_data":           req.image_b64,
        "use_financial_report": use_financial_report,
        "tool_calls":           [],
        "errors":               [],
    }


def _strip_keys(d: dict, keys: tuple[str, ...]) -> dict:
    """Return a sub-dict with only the requested keys present (and non-None)."""
    return {k: d[k] for k in keys if k in d and d[k] is not None}


def run_graph(emitter: EventEmitter, req: Any) -> None:
    """Worker entry point: pump graph.stream events into the SSE emitter."""
    import glob

    cb_token = graph_module.set_streaming_cb(lambda t: emitter.emit(ev.REPORT_TOKEN, {"delta": t}))
    initial_state = _build_initial_state(req)
    charts_before = set(glob.glob("charts/*.png"))
    started = time.time()

    final_report     = ""
    final_model      = "Groq"
    email_status     = ""
    new_gemini_exhausted = False
    all_tool_calls: list[dict] = []
    all_errors:     list[dict] = []
    seen_tool_steps = 0

    try:
        graph = graph_module.graph
        for update in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_out in update.items():
                if not isinstance(node_out, dict):
                    continue

                emitter.emit(ev.NODE_START, {
                    "node":  node_name,
                    "label": NODE_LABELS.get(node_name, node_name),
                })

                # tool.call events — flush newly-appended entries since last node
                for tc in node_out.get("tool_calls") or []:
                    all_tool_calls.append(tc)
                    if tc.get("tool_name") and tc["tool_name"] != "llm":
                        seen_tool_steps += 1
                        emitter.emit(ev.TOOL_CALL, {
                            "step":      seen_tool_steps,
                            "tool_name": tc["tool_name"],
                            "tool_args": tc.get("tool_args", {}),
                            "retries":   tc.get("retries", 0),
                        })

                for err in node_out.get("errors") or []:
                    all_errors.append(err)
                    if isinstance(err, dict):
                        emitter.emit(ev.ERROR, {
                            "node":    err.get("node", node_name),
                            "tool":    err.get("tool", ""),
                            "message": err.get("message", str(err)),
                        })
                    else:
                        emitter.emit(ev.ERROR, {"node": node_name, "tool": "", "message": str(err)})

                # Capture report fields as they arrive
                if node_name == "report_node":
                    final_report         = node_out.get("final_report") or node_out.get("report", "") or final_report
                    email_status         = node_out.get("email_status") or email_status
                    final_model          = node_out.get("final_model")  or final_model
                    new_gemini_exhausted = node_out.get("gemini_exhausted", new_gemini_exhausted)
                elif node_name == "reflection_node":
                    final_report = node_out.get("final_report", final_report)

                # Structural payload (front-end uses for KPI / verdict / matrix rendering)
                payload = _strip_keys(node_out, PAYLOAD_KEYS)
                emitter.emit(ev.NODE_COMPLETE, {
                    "node":    node_name,
                    "payload": payload or None,
                })

        # New chart files produced during this run
        charts_after = set(glob.glob("charts/*.png"))
        for chart_path in sorted(charts_after - charts_before):
            emitter.emit(ev.CHART, {"path": chart_path.replace("\\", "/")})

        # Persist history (best effort, mirrors app.py:1413-1421)
        try:
            tickers = list({
                tc.get("tool_args", {}).get("ticker")
                for tc in all_tool_calls
                if tc.get("tool_name") == "get_stock_data" and tc.get("tool_args", {}).get("ticker")
            })
            save_history(make_record(
                user_input  = req.user_input,
                tool_calls  = all_tool_calls,
                final_model = final_model,
                elapsed     = round(time.time() - started, 1),
                has_error   = bool(all_errors),
                tickers     = tickers,
            ))
        except Exception as exc:
            logger.warning("save_history failed: %s", exc)

        emitter.emit(ev.DONE, {
            "final_model":      final_model,
            "final_report":     final_report,
            "email_status":     email_status,
            "gemini_exhausted": new_gemini_exhausted,
            "elapsed":          round(time.time() - started, 2),
        })
    except Exception as exc:
        import traceback
        logger.error("graph run failed: %s\n%s", exc, traceback.format_exc())
        emitter.emit(ev.ERROR, {"node": "graph", "tool": "", "message": str(exc)})
        emitter.emit(ev.DONE, {
            "final_model":      "error",
            "final_report":     "",
            "email_status":     "",
            "gemini_exhausted": False,
            "elapsed":          round(time.time() - started, 2),
        })
    finally:
        graph_module.reset_streaming_cb(cb_token)
        emitter.close()
