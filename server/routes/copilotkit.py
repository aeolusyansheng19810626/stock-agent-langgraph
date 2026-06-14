from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request

logger = logging.getLogger("stockai.copilotkit")

SYSTEM_PROMPT = """你是股票分析界面的智能副驾驶（CopilotChat）。

你可以通过以下操作帮助用户：
- 切换界面主题（amber=墨黑琥珀 / cyan=电光青 / ink=账簿墨绿）
- 添加或删除自选股（如 NVDA、AAPL、600519.SS）
- 切换右侧面板的标签页（行情/资讯/公告/副驾驶）
- 开启或关闭开发模式（跳过 Gemini 仅用 Groq）

如果用户想做深度股票分析（基本面/技术面/风险报告），请告知他们在主对话框中输入问题——那里有完整的多节点 AI 分析管线。
"""

# Module-level refs populated by mount(); used by the root route handler below.
_sdk = None
_ck_handler = None


def _build_info_response() -> dict:
    """Build the info response in the format @copilotkit/react-core v1.60 expects.

    The Python SDK (0.1.94) returns agents as an array:
        {"agents": [{"name": "default", ...}]}
    But the frontend v1.60 expects agents as a name-keyed object:
        {"agents": {"default": {"description": ..., "capabilities": {}}}}

    We transform the SDK's array format to the correct object format here.
    """
    if _sdk is None:
        return {"agents": {}}
    raw = _sdk.info(context={"properties": {}, "frontend_url": None, "headers": {}})
    agents_list = raw.get("agents", [])
    if isinstance(agents_list, list):
        agents_map = {
            a["name"]: {"description": a.get("description", "")}
            for a in agents_list
            if isinstance(a, dict) and "name" in a
        }
    else:
        agents_map = agents_list  # already object format
    return {"agents": agents_map}


async def copilotkit_info_root(request: Request):
    """GET/POST /api/copilotkit (no trailing slash).

    Handles two patterns:
    - GET, or POST {method:"info"}  → return runtime info (agent capability list)
    - POST {method:"agent/run", ...} → delegate to SDK's handle_execute_agent (single-endpoint mode)
      The frontend v1.60 routes ALL requests to the same URL using body-method dispatch.
    """
    from fastapi.responses import JSONResponse
    from copilotkit.integrations.fastapi import handle_execute_agent

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    method = body.get("method", "")

    if request.method in ("GET", "OPTIONS") or method == "info":
        return JSONResponse(_build_info_response())

    if method == "agent/run":
        # Delegate to the AG-UI LangGraphAgent directly.
        # CopilotKitRemoteEndpoint.execute_agent() calls agent.execute() which
        # LangGraphAGUIAgent doesn't implement. Instead we build a RunAgentInput
        # from the body-method payload and stream via LangGraphAGUIAgent.run().
        run_body = body.get("body", {})
        from ag_ui.core.types import RunAgentInput
        from ag_ui.encoder import EventEncoder
        from starlette.responses import StreamingResponse

        # Get the underlying LangGraphAgent from our SDK's agent list
        agents = _sdk.agents({}) if callable(_sdk.agents) else _sdk.agents
        agent = next((a for a in agents if a.name == body.get("params", {}).get("agentId", "default")), None)
        if agent is None:
            return JSONResponse({"error": "Agent not found"}, status_code=404)

        # Build RunAgentInput (AG-UI protocol) from the single-endpoint body
        input_data = RunAgentInput(
            thread_id=run_body.get("threadId", ""),
            run_id=run_body.get("runId", ""),
            messages=run_body.get("messages", []),
            tools=run_body.get("tools", []),   # frontend actions
            context=run_body.get("context", []),
            state=run_body.get("state", {}),
            forwarded_props=run_body.get("forwardedProps", {}),
        )
        accept = request.headers.get("accept", "")
        encoder = EventEncoder(accept=accept)
        request_agent = agent.clone()

        async def event_generator():
            async for event in request_agent.run(input_data):
                yield encoder.encode(event)

        return StreamingResponse(event_generator(), media_type=encoder.get_content_type())

    # Fallback: unknown method, return info
    return JSONResponse(_build_info_response())


async def copilotkit_info_slash(request: Request):
    """GET /api/copilotkit/info — REST-mode runtime info probe."""
    from fastapi.responses import JSONResponse
    return JSONResponse(_build_info_response())


def _build_graph():
    from langchain_core.messages import SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import StateGraph
    from copilotkit import CopilotKitState

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    async def chatbot(state: CopilotKitState) -> dict:
        ck = state.get("copilotkit") or {}
        frontend_actions = ck.get("actions") or []

        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system_msg] + list(state.get("messages") or [])

        if frontend_actions:
            openai_tools = [
                a if (isinstance(a, dict) and a.get("type") == "function")
                else {"type": "function", "function": a}
                for a in frontend_actions
            ]
            response = await llm.bind_tools(openai_tools).ainvoke(messages)
        else:
            response = await llm.ainvoke(messages)

        return {"messages": [response]}

    from langgraph.checkpoint.memory import MemorySaver
    graph = StateGraph(CopilotKitState)
    graph.add_node("chatbot", chatbot)
    graph.set_entry_point("chatbot")
    graph.set_finish_point("chatbot")
    return graph.compile(checkpointer=MemorySaver())


def mount(app: FastAPI) -> None:
    """挂载 CopilotKit 端点。

    - GET/POST /api/copilotkit   → copilotkit_info_root → handle_info（能力清单）
    - /api/copilotkit/{path:path} → add_fastapi_endpoint catch-all（agent 执行）
    """
    global _sdk, _ck_handler

    try:
        from copilotkit import CopilotKitRemoteEndpoint, LangGraphAGUIAgent
        from copilotkit.integrations.fastapi import add_fastapi_endpoint
        from copilotkit.integrations.fastapi import handler as ck_handler
    except ImportError as e:
        logger.error("copilotkit SDK not available: %s", e)
        raise

    graph = _build_graph()
    agent = LangGraphAGUIAgent(name="default", graph=graph, description="股票分析界面副驾驶")
    _sdk = CopilotKitRemoteEndpoint(agents=[agent])
    _ck_handler = ck_handler

    # Register info routes BEFORE the catch-all so they take precedence.
    # GET+POST /api/copilotkit — no-trailing-slash probe by AgentRegistry
    app.add_api_route("/api/copilotkit", copilotkit_info_root, methods=["GET", "POST", "OPTIONS"])
    # GET /api/copilotkit/info — REST-mode probe (fetchRuntimeInfoAutoDetect tries this first)
    app.add_api_route("/api/copilotkit/info", copilotkit_info_slash, methods=["GET", "OPTIONS"])
    # Catch-all: handles /api/copilotkit/{path:path} (agent execute, state, etc.)
    add_fastapi_endpoint(app, _sdk, "/api/copilotkit")
    logger.info("CopilotKit endpoint mounted at /api/copilotkit (agent: copilot)")
