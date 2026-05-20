---
title: StockAI 股票分析 Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# StockAI 股票分析 Agent

基于 **LangGraph** 的多 Agent 并行股票分析助手；**FastAPI 后端 + React 前端**，SSE 流式协议驱动报告分段渲染。

---

## 启动方式

### 开发模式（前后端分离）

```bash
# 后端（终端 1）
pip install -r requirements.txt
python -m uvicorn server.main:app --reload --port 8000

# 前端（终端 2）
cd web
npm install
npm run dev          # → http://localhost:5173 ，自动 proxy /api 到 :8000
```

### 生产模式（单口）

```bash
cd web && npm run build && cd ..
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
# FastAPI 同时 serve API 与 web/dist 静态资源 → http://localhost:8000
```

### Docker

```bash
docker build -t stockai .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=xxx \
  -e TAVILY_API_KEY=zzz \
  -e GCP_SA_KEY="$(base64 -w0 /path/to/sa-key.json)" \
  -e GOOGLE_CLOUD_PROJECT=your-project \
  -e GOOGLE_CLOUD_REGION=global \
  stockai
```

---

## 项目结构

```
stock-agent-langgraph/
├── server/                      # FastAPI 后端
│   ├── main.py                  # app 实例 + 路由注册 + StaticFiles
│   ├── sse.py                   # 线程→asyncio.Queue→SSE 帧 桥接
│   ├── events.py                # SSE 事件类型常量（与 web/src/types/sse.ts 镜像）
│   ├── routes/
│   │   ├── analyze.py           # POST /api/analyze（SSE 流，包 graph.stream）
│   │   ├── _analyze_graph.py    # graph→SSE 事件分发逻辑
│   │   ├── docs.py              # PDF 上传 / 列表 / 删除
│   │   ├── quote.py             # yfinance 批量行情
│   │   ├── history.py           # 历史记录 read / clear
│   │   └── email.py             # 手动邮件发送
│   └── services/
│       ├── pdf_ingest.py        # PDF → ChromaDB 入库（无 streamlit 依赖）
│       └── image.py             # 图片→512px PNG→base64
├── web/                         # Vite + React 18 + TS 前端
│   ├── src/
│   │   ├── App.tsx              # 三栏 grid + 主题切换
│   │   ├── styles/stockai.css   # 设计系统（三主题 CSS 变量）
│   │   ├── layout/              # TopNav / Ticker / Sidebar / MainArea / ContextPanel / SettingsDrawer
│   │   ├── sidebar/             # Watchlist / UploadZone / DocumentList
│   │   ├── chat/                # ChatStream / Composer / Markdown / StepRow / SuggestionGrid
│   │   ├── report/              # ReportCard / KpiGrid / CandleChart / VerdictCards / Sparkline
│   │   ├── context/             # QuoteCard / NewsList / FilingsList
│   │   ├── icons/Icon.tsx       # 16 个内联 SVG
│   │   ├── api/                 # client.ts（fetch wrapper）+ sse.ts（fetch-event-source）
│   │   ├── store/               # zustand: chat / settings / docs / watchlist
│   │   └── types/sse.ts         # SSE 事件 TS 镜像
│   ├── vite.config.ts           # /api 与 /charts 反向代理到 :8000
│   └── package.json
├── graph.py                     # LangGraph 多 Agent 图（核心）
├── tools.py                     # 工具定义（不要修改）
├── history.py                   # 对话历史记录（history.json）
├── nodes/
│   └── financial_report_node.py # PDF 财报精读节点（Map-Reduce + Vision fallback）
├── tools/
│   ├── sec_fetcher.py           # SEC 财报抓取
│   └── cn_report_fetcher.py     # A股财报抓取
├── skills/                      # 工具使用说明（注入 system prompt）
├── charts/                      # 走势图输出目录（运行时自动创建，FastAPI 静态托管 /charts）
├── vectorstore/                 # ChromaDB 向量库（运行时自动创建，不上传 git）
├── design_handoff_stockai/      # UI 设计交付包（重构源材料）
├── Dockerfile                   # 多阶段：node build web/ → python runtime
├── .env                         # API Keys（不上传 git，参考 .env.example）
└── requirements.txt
```

> 旧版 `app.py`（Streamlit）已于 UI 重构时移除；如需对照请查看 git history 4b16b4b 之前的版本。

---

## API Keys / 环境变量

```
GROQ_API_KEY=          # Groq API key（Gemini 失败时 fallback）
TAVILY_API_KEY=        # Tavily 新闻搜索
GOOGLE_CLOUD_PROJECT=  # GCP 项目 ID（默认 yansheng-project）
GOOGLE_CLOUD_REGION=   # Vertex AI 区域（默认 global，可设为 us-central1 等）
```

**GCP 认证**：本地开发使用 `gcloud auth application-default login`（ADC）；
生产/Docker 用 Service Account JSON，base64 编码后设为环境变量 `GCP_SA_KEY`，
容器启动时由 `entrypoint.sh` 自动 decode 并注入 `GOOGLE_APPLICATION_CREDENTIALS`。

> `GEMINI_API_KEY` 已废弃（原 Google AI Studio 接入），现已切换至 Vertex AI。

---

## LangGraph 多 Agent 架构

```
用户提问
   ↓
parse_node（QUALITY_CASCADE）
  └─ 分析问题，生成调度计划（JSON），含 need_xxx 路由字段
   ↓ 条件路由
   ├─ [条件] financial_report_node → pdfplumber Map + QUALITY_CASCADE Reduce
   ↓ 条件路由（并行）
   ├─ [条件] data_node    → yfinance 获取数据   → Gemini Flash 技术面分析
   ├─ [条件] news_node    → Tavily 搜索新闻    → Gemini Flash 新闻摘要+情绪判断
   ├─ [条件] rag_node     → ChromaDB 检索财报  → Gemini Flash 财务指标提取
   └─ [条件] image_analysis → Gemini 2.5 Pro 原生读取上传图片（如有）
   ↓ fan-in（并行分析节点）
[条件] deep_read_node     → need_deep_read=true，双阶段精读批判
[条件] scoring_node       → need_scoring=true，Chain-of-Thought 多维度评分
[条件] risk_node          → need_risk=true，结构化风险矩阵
[条件] comparison_node    → need_comparison=true，多股对比排名与表格
[条件] hypothesis_node    → need_hypothesis=true，假设推演情景分析
[条件] reflection_node    → need_reflection=true，报告审核与修订建议
   ↓
report_node
  ├─ 运行模式：Gemini 2.5 Pro（Vertex AI，原生支持多模态图文）
  ├─ 兜底机制：Gemini 失败时自动 Fallback 至 Groq（含图片多模态降级）
  ├─ 开发模式：Groq QUALITY_CASCADE
  ├─ 逐 token 流式输出（_report_streaming_cb 注入）
  └─ 追加 comparison / risk_matrix / hypothesis / deep_read 结构化段落
```

每个中间节点都是真正的 Agent：先调用工具拿到原始数据，再用 LLM 做领域分析，
`report_node` 接收的是各 Agent 的预分析结论，而不是裸数据。

---

## 多模态支持 (Image Analysis)

系统支持 **JPG/PNG/WEBP** 格式图片的上传与分析：
- **上传方式**：Composer 工具栏「附件」按钮，浏览器原生 `<input type=file>`。
- **处理流程**：前端读为 base64，随 `POST /api/analyze` 一并提交（`image_b64` 字段）；后端写入 `AgentState["image_data"]`。前端不再压缩，过 2MB 直接拒绝。
- **智能调度**：`parse_node` 识别到图片后，会自动在提示词中注入视觉分析指令，并引导 `report_node` 开启多模态链路。
- **模型限制**：有图片时，`report_node` 会自动跳过不支持视觉的 Llama 模型，优先尝试顶级视觉模型层级。

---

## 模型配置

### 主力：Vertex AI Gemini（所有节点首选）

```python
GEMINI_FLASH = "google/gemini-3.5-flash"        # 速度优先节点
GEMINI_PRO   = "google/gemini-3.1-pro-preview"  # 质量优先节点
```

| 节点 | 主力模型 | Groq Fallback |
|------|----------|---------------|
| data_node / news_node / rag_node / scoring_node / deep_read S1 | Gemini **Flash** | TIER_LOW / QUALITY_CASCADE |
| parse_node / risk_node / comparison_node / hypothesis_node / deep_read S2 / reflection_node | Gemini **Pro** | QUALITY_CASCADE / 专属 CASCADE |
| report_node | Gemini **Pro**（流式，支持多模态） | call_groq()（含图片降级） |

### Fallback：Groq（Gemini 异常时自动切换）

```python
TIER_TOP       = "openai/gpt-oss-120b"
TIER_UPPER_MID = "openai/gpt-oss-20b"
TIER_MID       = "qwen/qwen3-32b"
TIER_LOW       = "meta-llama/llama-4-scout-17b-16e-instruct"
TIER_DEBUG     = "llama-3.1-8b-instant"  # 调试专用

QUALITY_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW, TIER_DEBUG]
```

---

## 流式输出架构（SSE 协议）

`report_node` 支持逐 token 流式输出。整条链路：

```
浏览器（fetch-event-source）           server/routes/analyze.py            graph.py
─────────────────────────────         ─────────────────────────────       ──────────────────────────────────
fetch /api/analyze (POST + body)  →   set_streaming_cb(emit)         →    _report_streaming_cb (ContextVar)
                                                                          ↓ 子线程 contextvars.copy_context()
                                      threading.Thread(graph.stream)      report_node 内部 llm.stream()
                                          ↓ 每个 update                       每个 token → cb()
                                      EventEmitter.emit(event, data) ←─┘ → emit(report.token, {delta})
                                          ↓
                                      asyncio.Queue → SSE 帧
                                          ↓
浏览器 useChat.appendToken(id, delta)  ← StreamingResponse
```

**SSE 事件协议**（`server/events.py` ↔ `web/src/types/sse.ts`）：

| event | data | 用途 |
|---|---|---|
| `node.start`     | `{node, label}`                                    | 主区头部状态文字 |
| `node.complete`  | `{node, payload?}`                                 | 结构化 payload 推给前端做 KPI / 估值卡分段渲染 |
| `tool.call`      | `{step, tool_name, tool_args, retries}`            | 步骤卡片实时追加 |
| `report.token`   | `{delta}`                                          | 主报告 token 流 |
| `report.section` | `{type, markdown}`                                 | comparison / risk_matrix / hypothesis / deep_read 段落 |
| `error`          | `{node, tool, message}`                            | 节点错误折叠区 |
| `chart`          | `{path}`                                           | `charts/*.png` 推给前端从 `/charts/...` 加载 |
| `done`           | `{final_model, final_report, email_status, ...}`   | 终态 |

**关键点**：
- `_report_streaming_cb` 是 `contextvars.ContextVar`，子线程用 `contextvars.copy_context().run(worker)` 启动以继承
- LangGraph 的 `graph.stream(..., stream_mode="updates")` 是同步迭代器，FastAPI 用 worker 线程 + asyncio.Queue 桥接到 SSE
- `node.complete.payload` 仅携带 `financial_metrics` / `scoring_result` / `risk_result` / `hypothesis_result` / `deep_read_result` 等结构化字段（白名单见 `_analyze_graph.PAYLOAD_KEYS`）

---

## Gemini 配额

现已切换至 **Vertex AI**，按 token 计费，无 RPM/RPD 免费额度限制。
如遇 `RESOURCE_EXHAUSTED`（项目配额耗尽）→ SSE `done.gemini_exhausted=true`，
前端自动切换后续请求走 Groq，设置抽屉关掉「开发模式」+ 刷新页面可重置。

---

## 工具说明

| 工具 | 数据源 | 说明 |
|------|--------|------|
| `get_stock_data(ticker)` | yfinance | 实时价格、涨跌幅、52周高低、PE、成交量 |
| `search_web(query)` | Tavily API | 3条结果，news_node 自动注入当前年份 |
| `get_stock_history(ticker, period)` | yfinance | 走势图保存 `charts/`，dpi=100 |
| `search_documents(query)` | ChromaDB 本地 | 多语言向量检索，首次自动下载约 120MB 模型 |
| `send_email_report(to, subject, body)` | Gmail OAuth | 首次需授权生成 `token.pickle` |

---

## UI 行为

- **三栏布局**：左 240px 自选股 + PDF 文档；中央对话 + 报告卡；右 360px 上下文栏（行情/资讯/公告 三 tab）
- **三套主题**：琥珀（默认）/ 电光青 / 墨绿，通过 CSS 变量切换，零重渲
- **涨跌色翻转**：设置抽屉「红涨绿跌」开关一键切换 A 股 / 美股惯例
- **流式渲染**：步骤卡片随 SSE `tool.call` 事件实时追加；报告 token 实时累积；`node.complete` 带结构化 payload 时渐进出 KPI / 估值卡
- **重试徽章**：工具调用瞬态错误后自动重试，步骤卡片显示 `⟳ N/3`
- **设置抽辑**：右上角齿轮按钮打开（220ms 三次贝塞尔滑入），含开发模式 / 引用必标注 / 红涨绿跌 / 自动邮件 / 数据源开关

---

## 重试机制

`graph.py` 中所有工具调用（`get_stock_data` / `get_stock_history` / `search_web` / `search_documents`）均通过 `_invoke_with_retry` 包装：

- **最多重试 3 次**，指数退避：1s → 2s → 4s
- **可重试错误**：timeout / timed out / connection / network / reset by peer / ssl / 429 / rate limit
- **不可重试错误**（API key 无效、解析失败等）：直接失败，记录到错误面板
- 中间失败但最终成功时，步骤卡片显示 `⟳ 重试 N/3` 橙色徽章，历史记录标 ❌（因错误曾发生）

## PDF 财报管理

- **上传**：侧栏「财报文档」拖放区或点击浏览，`POST /api/docs`（multipart）→ 自动写入 `tmp/` 并向量化到 ChromaDB
- **删除**：文档条目右侧 × 按钮，`DELETE /api/docs/{name}` 同时清理：
  - `tmp/` 中的原始 PDF 文件
  - ChromaDB 中该文件的所有向量片段
  - `vectorstore/processed_files.json` 中的注册记录
- **扫描件处理**：若 PDF 无可提取文本，跳过向量化但保留文件，仍可由 `financial_report_node` 通过 pdfplumber + Vision 精读

## 注意事项

- `tools.py` 不要修改，工具签名变更会影响 LangGraph 节点绑定
- `graph.py` 修改后需重启 uvicorn（`--reload` 模式下会自动重启）
- `skills/*.md` 修改后立即生效（注入 system prompt，非 graph 节点）
- 前端改 `web/src/`，dev 模式下 Vite HMR 自动更新；生产部署需重新 `npm run build`

## 部署到 HuggingFace Spaces

本项目通过 **GitHub Actions 自动同步**到 HF Space（`.github/workflows/sync-to-hf.yml`），
每次 push master 分支自动触发 Docker build。

**前置步骤**：
1. HF 创建 **Docker（Blank）** 类型 Space
2. GitHub repo → Settings → Secrets 添加 `HF_TOKEN`（HF write token）
3. HF Space → Settings → Repository secrets 添加：

| Secret | 说明 |
|--------|------|
| `GROQ_API_KEY` | Groq API key |
| `TAVILY_API_KEY` | Tavily 新闻搜索 |
| `GCP_SA_KEY` | GCP Service Account JSON 的 base64 编码（`base64 -w0 key.json`） |
| `GOOGLE_CLOUD_PROJECT` | GCP 项目 ID |
| `GOOGLE_CLOUD_REGION` | Vertex AI 区域（默认 `global`，可设 `us-central1` 等） |

push master 后 GitHub Actions 将代码同步到 HF，HF 自动 Docker build，监听 7860 端口。

**已知约束**：
- HF Free Tier 容器是 **ephemeral** 的：`tmp/`、`vectorstore/`、`charts/` 重启后丢失
- Gmail 邮件发送依赖 `token.pickle`，需本地完成首次 OAuth 授权后随代码一并提交（私有 Space）
