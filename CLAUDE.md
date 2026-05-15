# CLAUDE.md — stock-agent-langgraph

## 项目概述
基于 LangGraph 的股票分析多 Agent 系统。**FastAPI 后端 + React 前端**（Vite + TS + CSS Modules-style 全局命名空间），用户通过浏览器提问，多节点并行分析后流式生成报告。设计稿在 `design_handoff_stockai/`。

## 技术栈
- **后端**：FastAPI + LangGraph，SSE（`text/event-stream`）流式协议
- **前端**：Vite + React 18 + TypeScript + zustand + `@microsoft/fetch-event-source`
- **LLM**：Groq API（主）+ Google Gemini（report_node 优先）
- **向量数据库**：ChromaDB（`vectorstore/`）
- **PDF 解析**：财报上传至 `tmp/`，由 `financial_report_node` 处理

## 关键文件职责

| 文件 | 职责 |
|------|------|
| `graph.py` | 所有节点定义 + LangGraph 图构建，**核心文件** |
| `nodes/financial_report_node.py` | PDF 财报精读节点（Map-Reduce + Vision fallback） |
| `tools.py` | LangChain Tool 定义（get_stock_data / search_web 等） |
| `tools/sec_fetcher.py` | SEC 财报抓取 |
| `tools/cn_report_fetcher.py` | A股财报抓取 |
| `server/main.py` | FastAPI app + 路由注册 + StaticFiles |
| `server/routes/analyze.py` + `_analyze_graph.py` | `POST /api/analyze` SSE 流式分析入口（包 `graph.stream`） |
| `server/routes/{docs,quote,history,email}.py` | 周边接口 |
| `server/services/pdf_ingest.py` | PDF → ChromaDB 入库（无 streamlit 依赖） |
| `server/sse.py` | 线程→asyncio.Queue→SSE 帧 桥接 |
| `server/events.py` | SSE 事件常量（与 `web/src/types/sse.ts` 镜像） |
| `web/src/App.tsx` | 三栏 grid + 主题 provider |
| `web/src/styles/stockai.css` | 设计系统（三主题 CSS 变量） |
| `web/src/{layout,sidebar,chat,report,context}/` | 按设计稿 §9 拆分的组件 |
| `web/src/store/{chat,settings,docs,watchlist}.ts` | zustand 状态切片 |

## 图结构（LangGraph）

```
START → parse_node
  → [use_financial_report=True] → financial_report_node
      → [need_deep_read=True] → deep_read_node → 并行采集/分析节点
  → 并行采集：data_node / news_node / rag_node
  → 并行分析：scoring_node / risk_node / comparison_node / hypothesis_node / deep_read_node* / reflection_node
  → fan-in → report_node → END

* deep_read_node 条件触发（need_deep_read=True）：
  - use_financial_report=True 时，在 financial_report_node 之后优先运行，再进采集节点
  - 否则作为并行分析节点之一，与 scoring/risk 等同级 fan-in 到 report_node
```

## AgentState 关键字段

```python
# 路由控制
need_data / need_news / need_rag / need_scoring
need_risk / need_comparison / need_hypothesis
need_reflection / need_deep_read
use_financial_report / pdf_path

# 数据流
stock_data / news / rag_result
scoring_result / risk_result / comparison_result
hypothesis_result / reflection_result / deep_read_result
financial_metrics / risk_signals / report_citations

# 输出
report / final_report / email_status
```

## 模型分级（5-tier）

```python
TIER_TOP       = "openai/gpt-oss-120b"   # 复杂推理节点
TIER_UPPER_MID = "openai/gpt-oss-20b"
TIER_MID       = "qwen/qwen3-32b"
TIER_LOW       = "meta-llama/llama-4-scout-17b-16e-instruct"  # data/news 快速节点
TIER_DEBUG     = "llama-3.1-8b-instant"
```

- `parse_node`：QUALITY_CASCADE（从 TOP 开始降级）
- `data_node` / `news_node`：固定 TIER_LOW
- `scoring_node`：QUALITY_CASCADE（全档降级）
- `risk_node` / `comparison_node` / `reflection_node`：[TIER_TOP, TIER_UPPER_MID, TIER_MID]（限速时降级）
- `hypothesis_node`：[TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW]（部分降级）
- `deep_read_node`：S1 [TIER_MID, TIER_LOW, TIER_DEBUG]，S2 [TIER_TOP, TIER_UPPER_MID, TIER_MID]
- `report_node`：Gemini 优先，失败 fallback Groq QUALITY_CASCADE

## 开发规范

### 新增节点
1. 在 `graph.py` 中定义节点函数
2. 在 `AgentState` 中添加对应字段（`need_xxx: bool` + `xxx_result: Optional[...]`）
3. 在 `PLAN_SYSTEM` prompt 中添加路由规则
4. 在 `_normalize_plan` 的 return dict 中添加字段（**容易漏，必须检查**）
5. 在 `parse_node` 的 result dict 中添加字段
6. 在 `build_graph()` 中注册节点和边
7. 在 `report_node` 中读取结果并追加到报告

### 节点函数模板
```python
def xxx_node(state: AgentState) -> dict:
    if not state.get("need_xxx"):
        return {"xxx_result": None, "tool_calls": [], "errors": []}
    # ... 逻辑
    return {"xxx_result": result, "tool_calls": [...], "errors": [...]}
```

### 禁止事项
- ❌ **禁止重写整个文件**：只允许精确修改涉及变更的函数或段落，不得整体替换 `graph.py` / `server/main.py` / `web/src/App.tsx` 等核心文件
- ❌ 不要修改 `QUALITY_CASCADE` 顺序
- ❌ 不要在节点函数里直接读取环境变量，统一用 `state.get("groq_api_key") or os.getenv(...)`
- ❌ 不要在 `_normalize_plan` return 里漏字段（历史上 `need_hypothesis` 因此出过 bug）
- ❌ 不要改动 `financial_report_node.py` 的 Map-Reduce 结构，除非明确需要
- ❌ 不要在 `graph.py` 里直接赋值 `_report_streaming_cb = ...`：它是 `ContextVar`，必须用 `set_streaming_cb()` / `reset_streaming_cb()` 配对调用
- ❌ 不要把样式写进 React 组件 `style={{}}` inline（除占位 placeholder）；走 `web/src/styles/stockai.css` 的 `sx-*` class 体系，保持设计稿对齐

### 流式输出（report_node + SSE）
- `_report_streaming_cb` 是 `contextvars.ContextVar[Optional[Callable[[str], None]]]`（`graph.py:98` 周围）
- 调用方：`server/routes/_analyze_graph.run_graph()` 用 `set_streaming_cb(lambda t: emitter.emit("report.token", {"delta": t}))` 装回调
- 子线程必须用 `contextvars.copy_context().run(target)` 启动（`server/sse.run_in_thread()` 已封装），否则子线程读到的 ContextVar 是默认 None
- `_stream_with_cb(llm_obj, messages, cb)` 辅助函数兼容 Gemini / Groq，逐 chunk 调用 cb
- `call_groq()` 和 Gemini 路径均检测 `cb = _report_streaming_cb.get()` 是否非 None，有则走 `llm.stream()`，否则退回 `llm.invoke()`
- 追加的结构化段落（comparison / risk_matrix 等）在主体生成后计算差量，一并 push 到 cb（最终也是 SSE `report.token` 事件）
- `node.complete.payload` 仅携带白名单字段（见 `server/routes/_analyze_graph.PAYLOAD_KEYS`），前端用于 KPI / 估值卡分段渲染

### Token 优化约定（report_node prompt 构建）
- `stock_data`：只传 `[技术面分析 by data_agent]` 部分，截掉 `[原始数据]` 以下内容（原始数据已由 data_node LLM 提炼，重复传入无意义）
- `hypothesis_result`：只传 `conclusion` + `scenarios` 一行摘要，不要 `json.dumps` 全量传入（`transmission_chain` / `key_assumptions` 对 report_node 无用）
- `chat_history_text`：截断到最近 12 行（约 3 轮对话），长对话后 history 线性增长是最大 token 泄漏点

### 调试 / 自动测试规范
- 测试时写临时脚本（如 `test_stream.py`），在 `import graph` 后立即 patch 所有模型变量为 `TIER_DEBUG`，运行后删除脚本：
  ```python
  import graph as G
  _D = G.TIER_DEBUG
  G.QUALITY_CASCADE = [_D]; G.DATA_AGENT_MODEL = _D; G.NEWS_AGENT_MODEL = _D
  G.RISK_MODEL_CASCADE = [_D]; G.COMPARISON_MODEL_CASCADE = [_D]
  G.REFLECTION_MODEL_CASCADE = [_D]; G.DEEP_READ_S1_CASCADE = [_D]; G.DEEP_READ_S2_CASCADE = [_D]
  ```
- patch 只在进程内生效，不会污染 graph.py 源文件
- 提交前确认 graph.py 中无残留的 `TIER_DEBUG` 临时赋值

### Commit 规范
commit message 必须包含修改者名字和修改内容摘要：
```
Claude Code: add deep_read_node and update AgentState fields
Gemini: fix scoring_node cascade model assignment
Codex: refactor report_node to support hypothesis_result
```

### JSON 解析
所有节点的 LLM 输出用 `_parse_json_from_text()` 解析，不要自己写解析逻辑。

## 环境变量（.env）
```
GROQ_API_KEY=
GEMINI_API_KEY=
TAVILY_API_KEY=
```
Gmail 发送用 OAuth，首次运行生成 `token.pickle`，无需配置密码。

## 常用命令
```bash
# 后端 dev
python -m uvicorn server.main:app --reload --port 8000

# 后端 dev（不烧 Groq quota，返回固定 SSE mock）
STOCKAI_USE_MOCK=1 STOCKAI_SKIP_WARMUP=1 python -m uvicorn server.main:app --port 8000

# 前端 dev（自动 proxy /api 到 :8000）
cd web && npm run dev

# 前端构建（生产部署用）
cd web && npm run build       # → web/dist/

# 单口生产模式（FastAPI 同时 serve API + SPA）
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

# 临时测试脚本（用完即删，TIER_DEBUG patch 在脚本内完成）
PYTHONIOENCODING=utf-8 python test_stream.py
```

## 目录说明
- `server/` — FastAPI 后端（M1-M3 重构产出，替代旧 `app.py`）
- `web/` — Vite + React 前端（M4-M9 重构产出，替代旧 Streamlit UI）
- `design_handoff_stockai/` — UI 设计交付（重构源材料 / 视觉真理）
- `tmp/` — 用户上传的 PDF 财报（不提交 git）
- `vectorstore/` — ChromaDB 向量数据库（不提交 git）
- `charts/` — 生成的图表临时文件，FastAPI 静态托管 `/charts/*.png`
- `reports/` — 生成的报告临时文件
- `skills/` — 工具使用说明 .md（注入 system prompt 用）
