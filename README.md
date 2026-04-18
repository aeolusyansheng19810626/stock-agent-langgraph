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
parse_node（QUALITY_CASCADE）
  └─ 分析问题，生成调度计划（JSON），含 need_scoring / need_risk / need_comparison / need_reflection 判断
   ↓ 条件路由
   ├─ [条件] financial_report_node → pdfplumber Map + QUALITY_CASCADE Reduce
   ↓ 条件路由（并行）
   ├─ [条件] data_node   → yfinance 获取数据   → LLaMA 技术面分析
   ├─ [条件] news_node   → Tavily 搜索新闻    → LLaMA 新闻摘要+情绪判断
   └─ [条件] rag_node    → ChromaDB 检索财报  → QUALITY_CASCADE 财务指标提取
   ↓ fan-in
[条件并行] deep_read_node（QUALITY_CASCADE）
  └─ need_deep_read=true 时触发，双阶段：1.摘要提取；2.质疑分析
[条件并行] scoring_node（QUALITY_CASCADE）
  └─ need_scoring=true 时才触发，Chain-of-Thought 多维度评分
[条件并行] risk_node（gpt-oss-120b）
  └─ need_risk=true 时才触发，输出结构化风险矩阵
[条件并行] comparison_node（gpt-oss-120b）
  └─ need_comparison=true 时才触发，输出对比排名、胜出方和逐维度对比表
[条件并行] hypothesis_node（gpt-oss-120b）
  └─ need_hypothesis=true 时触发（如果/假设/what if），输出传导路径和情景分析表
   ↓
report_node
  ├─ 运行模式：Gemini 2.5 Flash（失败时 fallback → Groq QUALITY_CASCADE）
  ├─ 开发模式：Groq QUALITY_CASCADE
  └─ 如有 comparison_result / risk_result，在报告中追加对比表格和风险矩阵章节
   ↓ 条件路由
[条件] reflection_node（gpt-oss-120b）
  └─ need_reflection=true 时审核报告，输出问题列表、修订建议和修订版报告
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
    need_scoring: bool          # 是否需要多维度评分（综合分析=True，简单查价=False）
    need_deep_read: bool        # 是否需要精读批判（精读/深度分析/质疑/论文/研究报告=True）
    need_risk: bool             # 是否需要风险矩阵（风险/隐患/担忧/risk=True）
    need_comparison: bool       # 是否需要多股票对比
    need_hypothesis: bool       # 是否需要假设推演（如果/假设/what if=True）
    need_reflection: bool       # 是否需要报告自我反思（深度/详细/严谨/全面或need_scoring=True）
    comparison_dimensions: List[str]  # 指定对比维度，空列表代表全维度
    use_financial_report: bool  # 是否触发 financial_report_node
    pdf_path: Optional[str]
    financial_metrics: Optional[dict]; risk_signals: Optional[list]; report_citations: Optional[list]
    stock_data: str; news: str; rag_result: str
    deep_read_result: Optional[dict]  # deep_read_node 输出的摘要与批判 JSON
    scoring_result: dict  # scoring_node 输出的多维度评分 JSON
    risk_result: Optional[dict]  # risk_node 输出的结构化风险矩阵 JSON
    comparison_result: Optional[dict]  # comparison_node 输出的排名和对比表 JSON
    hypothesis_result: Optional[dict]  # hypothesis_node 输出的推演结论 JSON
    reflection_result: Optional[str]  # reflection_node 输出的问题列表/严重程度/修订建议 JSON 字符串
    report: str; final_report: str; email_status: str; final_model: str
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

模型采用 **5-tier 级联**，限速时自动降级到下一档，无需手动切换：

```python
TIER_TOP       = "openai/gpt-oss-120b"    # 上
TIER_UPPER_MID = "openai/gpt-oss-20b"     # 上中
TIER_MID       = "qwen/qwen3-32b"         # 中
TIER_LOW       = "meta-llama/llama-4-scout-17b-16e-instruct"  # 下（Fast 节点固定用此档）
TIER_DEBUG     = "llama-3.1-8b-instant"   # 调试

QUALITY_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW, TIER_DEBUG]
```

| 节点 | 策略 |
|------|------|
| parse / rag / scoring / report(Groq) / financial_report Reduce | QUALITY_CASCADE（上→下依次降级） |
| risk | TIER_TOP（openai/gpt-oss-120b，结构化风险矩阵） |
| comparison | TIER_TOP（openai/gpt-oss-120b，结构化对比排名和表格） |
| hypothesis | TIER_TOP（openai/gpt-oss-120b，假设推演结论） |
| reflection | TIER_TOP（openai/gpt-oss-120b，报告审核与修订） |
| data / news / financial_report Map | TIER_LOW 直接调用，不级联 |

## 双模式运行

| 模式 | parse/rag/scoring | risk/comparison/reflection | data/news | report_node |
|------|-------------------|--------------------------|-----------|-------------|
| 开发模式（dev_mode=True） | QUALITY_CASCADE | gpt-oss-120b | LLaMA | Groq QUALITY_CASCADE |
| 运行模式（dev_mode=False） | QUALITY_CASCADE | gpt-oss-120b | LLaMA | Gemini 2.5 Flash → Groq fallback |

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

## UI 展示

- 主报告默认展示 `final_report`，当 `reflection_node` 触发时展示修订版报告。
- 报告下方会显示「自我反思」折叠区，包含发现的问题、严重程度和修订建议。

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
