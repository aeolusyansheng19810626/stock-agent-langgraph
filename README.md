# StockAI 股票分析 Agent

基于 **LangGraph** 的多 Agent 并行股票分析助手，Streamlit Web UI。

---

## 启动方式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动
python -m streamlit run app.py
```

---

## 项目结构

```
stock-agent-langgraph/
├── app.py              # Streamlit Web UI（主程序）
├── graph.py            # LangGraph 多 Agent 图（核心）
├── tools.py            # 工具定义（不要修改）
├── components/
│   └── stock_ticker.py # 实时股价侧边栏组件（@st.fragment 30s刷新）
├── skills/             # 工具使用说明（注入 system prompt）
├── charts/             # 走势图输出目录（运行时自动创建）
├── vectorstore/        # ChromaDB 向量库（运行时自动创建，不上传 git）
├── .env                # API Keys（不上传 git，参考 .env.example）
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
parse_node（LLaMA via Groq）
  └─ 分析问题，生成调度计划（JSON）
   ↓ 条件路由（并行）
   ├─ data_node   → yfinance 获取数据 → LLaMA 技术面分析
   ├─ news_node   → Tavily 搜索新闻  → LLaMA 新闻摘要+情绪判断
   └─ rag_node    → ChromaDB 检索财报 → LLaMA 财务指标提取
   ↓ fan-in（等待所有节点完成，每个节点输出已预分析的结论）
report_node
  ├─ 运行模式：Gemini 2.5 Flash（失败时 fallback → Groq）
  └─ 开发模式：openai/gpt-oss-120b via Groq
```

每个中间节点都是真正的 Agent：先调用工具拿到原始数据，再用 LLM 做领域分析，
`report_node` 接收的是各 Agent 的预分析结论，而不是裸数据。

### AgentState 关键字段

```python
class AgentState(TypedDict):
    user_input: str
    chat_history_text: str
    tickers: List[str]
    need_data: bool; need_news: bool; need_rag: bool; need_history: bool
    stock_data: str; news: str; rag_result: str
    report: str; email_status: str; final_model: str
    gemini_exhausted: bool
    tool_calls: Annotated[List[dict], operator.add]  # 并行节点自动合并
    errors:    Annotated[List[dict], operator.add]
```

### 并行执行原理

- LangGraph 通过 `add_conditional_edges` 实现 fan-out，多节点同时调度
- `Annotated[List, operator.add]` reducer 自动合并各节点的 tool_calls
- `stream_mode="updates"` 实时推送每个节点完成状态到 Streamlit status box
- 总耗时 ≈ 最慢节点的时间，而非各节点之和

### 条件路由逻辑（parse_node → 各节点）

```python
def route_to_agents(state):
    targets = []
    if state.get("need_data"):  targets.append("data_node")
    if state.get("need_news"):  targets.append("news_node")
    if state.get("need_rag"):   targets.append("rag_node")
    return targets or "report_node"  # 无需数据时直接生成报告
```

---

## 模型配置

所有节点的模型在 `graph.py` 顶部集中配置，改一行即可切换，无需改动节点代码：

```python
FAST_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
QUALITY_MODEL = "openai/gpt-oss-120b"

PARSE_MODEL       = FAST_MODEL     # parse_node
DATA_AGENT_MODEL  = FAST_MODEL     # data_node
NEWS_AGENT_MODEL  = FAST_MODEL     # news_node
RAG_AGENT_MODEL   = FAST_MODEL     # rag_node
REPORT_GROQ_MODEL = QUALITY_MODEL  # report_node（Groq 路径）
```

## 双模式运行

| 模式 | parse/data/news/rag | report_node |
|------|---------------------|-------------|
| 开发模式（dev_mode=True） | LLaMA（fast） | openai/gpt-oss-120b |
| 运行模式（dev_mode=False） | LLaMA（fast） | Gemini 2.5 Flash → Groq fallback |

切换：侧边栏「🛠️ 开发模式」toggle

---

## Gemini 配额

| 限制 | 额度 |
|------|------|
| RPM | 5 |
| RPD | 20 |
| 重置时间 | 北京时间 15:00 |

- `RESOURCE_EXHAUSTED` → 标记 `gemini_exhausted=True`，后续直接走 Groq
- 侧边栏「🔴 Gemini 已耗尽」按钮可手动恢复

---

## 工具说明

### `get_stock_data(ticker)`
- 数据源：yfinance，返回实时价格、涨跌幅、52周高低、PE、成交量

### `search_web(query)`
- 数据源：Tavily API，返回 3 条结果
- news_node 自动追加当前年份，确保返回最新新闻

### `get_stock_history(ticker, period)`
- period：`1mo` / `3mo` / `6mo` / `1y` / `2y`
- 走势图保存到 `charts/`，dpi=100（平衡质量与传输速度）

### `search_documents(query)`
- ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2`（本地，支持中文，约 120MB）
- 向量库持久化，重启不丢失；首次上传 PDF 自动下载模型

### `send_email_report(to, subject, body)`
- Gmail API，首次需 OAuth 授权生成 `token.pickle`

---

## 性能优化记录

- `@st.cache_resource` 包装 graph 加载，避免每次 rerun 重新编译
- 走势图 dpi=100（原150），文件体积减少约55%，渲染更快
- news_node 自动注入年份，避免 Tavily 返回旧数据

---

## System Prompt 经验

对 LLM 下指令要用强制句式，而不是建议：

```
# 无效（LLM 容易忽略）
如果用户询问财务数据，优先调用 search_documents

# 有效
用户询问财报时，【必须】首先调用 search_documents，禁止直接调用 search_web
```

| 模型 | 指令遵循能力 |
|------|------------|
| Claude | 最强，软语气即可 |
| GPT-4o / gpt-oss-120b | 强 |
| Gemini 2.5 Flash | 中等 |
| LLaMA（Groq） | 较弱，需强制句式 |

---

## 注意事项

- `tools.py` 不要修改，工具签名变更会影响 LangGraph 节点绑定
- `graph.py` 是核心，修改后需重启 Streamlit（cache_resource 会失效）
- `skills/*.md` 修改后立即生效（注入 system prompt，非 graph 节点）
