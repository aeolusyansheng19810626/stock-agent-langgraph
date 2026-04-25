# StockAI 股票分析 Agent

基于 **LangGraph** 的多 Agent 并行股票分析助手，Streamlit Web UI，支持 report_node 逐字流式输出。

---

## 启动方式

```bash
# 激活虚拟环境
source venv/Scripts/activate   # Windows (bash)
# 或
venv\Scripts\activate          # Windows (cmd/PowerShell)

# 安装依赖（首次）
pip install -r requirements.txt

# 启动
python -m streamlit run app.py
```

---

## 项目结构

```
stock-agent-langgraph/
├── app.py                       # Streamlit Web UI（主程序）
├── graph.py                     # LangGraph 多 Agent 图（核心）
├── tools.py                     # 工具定义（不要修改）
├── history.py                   # 对话历史记录（history.json）
├── nodes/
│   └── financial_report_node.py # PDF 财报精读节点（Map-Reduce + Vision fallback）
├── tools/
│   ├── sec_fetcher.py           # SEC 财报抓取
│   └── cn_report_fetcher.py     # A股财报抓取
├── components/
│   └── stock_ticker.py          # 实时股价侧边栏组件（@st.fragment 30s刷新）
├── skills/                      # 工具使用说明（注入 system prompt）
├── charts/                      # 走势图输出目录（运行时自动创建）
├── vectorstore/                 # ChromaDB 向量库（运行时自动创建，不上传 git）
├── .env                         # API Keys（不上传 git，参考 .env.example）
└── requirements.txt
```

---

## API Keys

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

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
   ├─ [条件] data_node    → yfinance 获取数据   → LLaMA 技术面分析
   ├─ [条件] news_node    → Tavily 搜索新闻    → LLaMA 新闻摘要+情绪判断
   ├─ [条件] rag_node     → ChromaDB 检索财报  → QUALITY_CASCADE 财务指标提取
   └─ [条件] image_analysis → Gemini 2.5 Flash 原生读取上传图片（如有）
   ↓ fan-in（并行分析节点）
[条件] deep_read_node     → need_deep_read=true，双阶段精读批判
[条件] scoring_node       → need_scoring=true，Chain-of-Thought 多维度评分
[条件] risk_node          → need_risk=true，结构化风险矩阵
[条件] comparison_node    → need_comparison=true，多股对比排名与表格
[条件] hypothesis_node    → need_hypothesis=true，假设推演情景分析
[条件] reflection_node    → need_reflection=true，报告审核与修订建议
   ↓
report_node
  ├─ 运行模式：Gemini 2.5 Flash（原生支持多模态图文）
  ├─ 兜底机制：Gemini 失败或不支持图片时，自动 Fallback 至 Groq 纯文字分析（QUALITY_CASCADE）
  ├─ 开发模式：Groq QUALITY_CASCADE
  ├─ 逐 token 流式输出（_report_streaming_cb 注入）
  └─ 追加 comparison / risk_matrix / hypothesis / deep_read 结构化段落
```

每个中间节点都是真正的 Agent：先调用工具拿到原始数据，再用 LLM 做领域分析，
`report_node` 接收的是各 Agent 的预分析结论，而不是裸数据。

---

## 多模态支持 (Image Analysis)

系统现在支持 **JPG/PNG/WEBP** 格式图片的上传与分析：
- **上传方式**：Streamlit 聊天输入框内嵌的 `+` 号按钮（利用 `st.chat_input(accept_file=True)`）。
- **处理流程**：图片在前端自动压缩（512x512）并转为 Base64，存入 `AgentState["image_data"]`。
- **智能调度**：`parse_node` 识别到图片后，会自动在提示词中注入视觉分析指令，并引导 `report_node` 开启多模态链路。
- **模型限制**：有图片时，`report_node` 会自动跳过不支持视觉的 Llama 模型，优先尝试顶级视觉模型层级。

---

## 模型配置（5-tier）

```python
TIER_TOP       = "openai/gpt-oss-120b"
TIER_UPPER_MID = "openai/gpt-oss-20b"
TIER_MID       = "qwen/qwen3-32b"
TIER_LOW       = "meta-llama/llama-4-scout-17b-16e-instruct"  # Fast 节点固定
TIER_DEBUG     = "llama-3.1-8b-instant"                       # 调试专用

QUALITY_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW, TIER_DEBUG]
```

| 节点 | 模型策略 |
|------|----------|
| parse / rag / scoring / report(Groq-Text) / financial_report Reduce | QUALITY_CASCADE |
| report_node (Vision) | [TOP, UPPER_MID, MID] (跳过 Llama) |
| risk / comparison / reflection | [TOP, UPPER_MID, MID] |
| hypothesis | [TOP, UPPER_MID, MID, LOW] |
| deep_read S1 | [MID, LOW, DEBUG] |
| deep_read S2 | [TOP, UPPER_MID, MID] |
| data / news / financial_report Map | TIER_LOW 固定 |
| report_node | Gemini 2.5 Flash → Groq fallback |

---

## 流式输出架构

`report_node` 支持逐 token 流式输出，让报告边生成边显示在 UI 中：

```
graph.py                                  app.py（主线程）
────────────────────────────              ──────────────────────────────
_report_streaming_cb = None    ←──注入── lambda t: token_q.put(t)
                                          │
report_node 调用 llm.stream()             _steps_ph = st.empty()  ← 步骤卡片占位符（chat_message 之前）
  每个 token → _report_streaming_cb       │
  → token 入队 token_q                   with st.chat_message("assistant"):
                                              st.write_stream(tok_gen())  ← 消费队列
finally: token_q.put(None)  ────────→        阻塞直到队列 None 信号
```

- **后台线程**运行 `graph.stream()`，通过 `add_script_run_ctx` 传播 Streamlit session 上下文
- **步骤卡片**：流式结束后一次性写入 `_steps_ph`（位于 chat_message 上方），脚本自然向下继续渲染，无需 `st.rerun()`，消除推理结束时的闪烁

---

## Gemini 配额

| 限制 | 额度 |
|------|------|
| RPM | 5 |
| RPD | 20 |
| 重置时间 | 北京时间 15:00 |

- `RESOURCE_EXHAUSTED` → 自动标记 `gemini_exhausted=True`，后续直接走 Groq
- 侧边栏「🔴 Gemini 已耗尽」按钮可手动恢复

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

- **执行阶段**：状态栏实时更新节点进度 → 报告逐字流式输出
- **完成后**：步骤卡片出现在报告上方，预设快捷提问卡片渲染在最新内容之后
- **重试徽章**：工具调用遇到瞬态错误（超时 / 限速 / 网络）后自动重试，步骤卡片显示 `⟳ 重试 N/3`
- **自我反思**：`need_reflection=True` 时，报告下方显示折叠的「自我反思」区块

---

## 重试机制

`graph.py` 中所有工具调用（`get_stock_data` / `get_stock_history` / `search_web` / `search_documents`）均通过 `_invoke_with_retry` 包装：

- **最多重试 3 次**，指数退避：1s → 2s → 4s
- **可重试错误**：timeout / timed out / connection / network / reset by peer / ssl / 429 / rate limit
- **不可重试错误**（API key 无效、解析失败等）：直接失败，记录到错误面板
- 中间失败但最终成功时，步骤卡片显示 `⟳ 重试 N/3` 橙色徽章，历史记录标 ❌（因错误曾发生）

## PDF 财报管理

- **上传**：侧边栏「📄 财报文档」区域上传 PDF，自动写入 `tmp/` 并向量化到 ChromaDB
- **删除**：每个已上传文件旁有 🗑 按钮，点击后同时清理：
  - `tmp/` 中的原始 PDF 文件
  - ChromaDB 中该文件的所有向量片段
  - `vectorstore/processed_files.json` 中的注册记录
- **扫描件处理**：若 PDF 无可提取文本（扫描件），跳过向量化但保留文件，仍可由 `financial_report_node` 通过 pdfplumber + Vision 精读

## 注意事项

- `tools.py` 不要修改，工具签名变更会影响 LangGraph 节点绑定
- `graph.py` 修改后需重启 Streamlit（`@st.cache_resource` 不会自动失效）
- `skills/*.md` 修改后立即生效（注入 system prompt，非 graph 节点）
