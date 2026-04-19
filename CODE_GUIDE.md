# 代码结构说明

---

## 核心文件

### `app.py` — 界面层（Streamlit Web UI）
用户直接看到的页面。负责：
- 渲染聊天界面、侧边栏、走势图卡片
- 接收用户输入，调用 `graph.py` 里的 LangGraph 图执行分析
- 把分析结果（报告、工具调用步骤、图表）展示出来
- 管理 session_state（对话历史、Gemini 是否耗尽、已上传文档列表等）

**不包含任何 AI 逻辑，只管显示。**

---

### `graph.py` — AI 大脑（LangGraph 多 Agent 图）
整个项目最核心的文件。定义了 10 个节点：

```
parse_node            → 必须触发：理解问题，输出 JSON 调度计划（决定启用哪些节点）
financial_report_node → 条件触发（use_financial_report=True）：下载/读取财报 PDF，Map-Reduce 提取财务指标
data_node             → 条件触发（need_data）：yfinance 获取行情 → LLM 技术面分析
news_node             → 条件触发（need_news）：Tavily 搜索新闻 → LLM 新闻摘要+情绪
rag_node              → 条件触发（need_rag）：ChromaDB 检索财报 → LLM 财务指标提取
scoring_node          → 条件触发（need_scoring）：Chain-of-Thought 三维度评分（财务/情绪/技术）
risk_node             → 条件触发（need_risk）：识别关键风险，输出风险矩阵
comparison_node       → 条件触发（need_comparison）：多股票逐维度对比，输出排名
hypothesis_node       → 条件触发（need_hypothesis）：假设情景推演（Step 1~4）
deep_read_node       → 条件触发（need_deep_read）：双阶段精读（摘要+质疑）
reflection_node       → 条件触发（need_reflection）：严苛审核报告，输出修订版
report_node           → 必须触发：整合所有节点结论，生成最终分析报告
```

#### 执行顺序（完整路径）

```
parse_node
  ↓ [条件] financial_report_node
  ↓ [条件] deep_read_node
  ↓ 并行分发（fan-out）
  ├─ data_node
  ├─ news_node
  └─ rag_node
  ↓ 并行分析（fan-out，等待所有采集节点完成）
  ├─ scoring_node
  ├─ risk_node
  ├─ comparison_node
  ├─ hypothesis_node
  └─ reflection_node
  ↓ fan-in
report_node → END
```

- `data/news/rag` 三节点并行采集；完成后 `scoring/risk/comparison/hypothesis/reflection` 并行分析
- 各分析节点内部有条件判断，flag=False 时直接空返回，不消耗 token
- `reflection_node` 在 `report_node` 之前运行，可修订报告正文

#### 模型分工（5-tier 级联）

文件顶部集中配置，改常量即可：

```python
TIER_TOP       = "openai/gpt-oss-120b"
TIER_UPPER_MID = "openai/gpt-oss-20b"
TIER_MID       = "qwen/qwen3-32b"
TIER_LOW       = "meta-llama/llama-4-scout-17b-16e-instruct"
TIER_DEBUG     = "llama-3.1-8b-instant"
QUALITY_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW, TIER_DEBUG]
```

| 节点 | 策略 |
|------|------|
| parse / rag / scoring / risk / comparison / hypothesis / reflection / report(Groq路径) | QUALITY_CASCADE（遇限速自动降级） |
| data / news | TIER_LOW 直接调用，不降级 |
| financial_report Map 阶段 | TIER_LOW 直接调用，不降级 |
| financial_report Reduce 阶段 | QUALITY_CASCADE |

限速关键词：`429 / rate_limit / rate limit / 503 / over_capacity / model_overloaded`  
触发后立即尝试下一档，无等待。

#### 双模式运行

| 模式 | report_node |
|------|-------------|
| 开发模式（dev_mode=True） | Groq QUALITY_CASCADE |
| 运行模式（dev_mode=False） | Gemini 2.5 Flash → Groq fallback |

Gemini 配额耗尽（RESOURCE_EXHAUSTED）时标记 `gemini_exhausted=True`，后续自动走 Groq。

#### report_node prompt 构建规则（Token 优化）

各节点结果传入 report_node 前会做裁剪，避免重复 token：

| 字段 | 传入内容 | 截掉的部分 |
|------|---------|-----------|
| `stock_data` | `[技术面分析 by data_agent]` 部分 | `[原始数据]` 以下（已由 LLM 提炼） |
| `hypothesis_result` | `conclusion` + `scenarios` 一行摘要 | `transmission_chain` / `key_assumptions` |
| `chat_history_text` | 最近 12 行（约 3 轮） | 更早的历史轮次 |

---

### `tools.py` — 工具层（不要改这个文件）
定义了 AI 可以调用的 5 个工具函数：

| 工具 | 作用 |
|------|------|
| `get_stock_data` | 获取实时股价、PE、52周高低等 |
| `get_stock_history` | 生成历史走势图（保存为 PNG） |
| `search_web` | 用 Tavily 搜索网络新闻 |
| `search_documents` | 在已上传的 PDF 里检索内容（ChromaDB） |
| `send_email_report` | 用 Gmail 发送分析报告邮件 |

---

### `nodes/financial_report_node.py` — 财报精读节点

独立模块，实现 Map-Reduce PDF 解析：

- **`_read_pdf()`**: 用 pdfplumber 提取文字+表格，按 ~12000 字符切块
- **`_map_chunk()`**: 每个 chunk 用 TIER_LOW 提取财务指标 JSON（快速，不降级）
- **`_reduce()`**: 汇总所有 chunk 结果，用 QUALITY_CASCADE 生成最终摘要，返回 `(dict, model_used)`
- **`_vision_fallback()`**: 文字提取失败时将页面转图片，用 QUALITY_CASCADE vision 解析

输出字段：`financial_metrics / risk_signals / report_citations`

---

### `tools/` — 财报下载工具

| 文件 | 作用 |
|------|------|
| `cn_report_fetcher.py` | A股/港股：从东方财富 API 下载最新年报 PDF |
| `sec_fetcher.py` | 美股：从 SEC EDGAR 下载 10-K/20-F |

ticker 市场判断规则：
- `.SS` / `.SZ` → A股（东方财富）
- `.HK` → 港股（东方财富）
- 其他 → 美股（SEC EDGAR）

---

## 辅助文件

### `components/stock_ticker.py` — 侧边栏实时股价
侧边栏里那个「自选股票」输入框 + 实时价格卡片。
用 `@st.fragment(run_every=30)` 每 30 秒自动刷新，不影响主界面。

---

### `skills/*.md` — 工具使用说明（注入 prompt）
4 个 markdown 文件，描述每个工具的使用场景和限制。
被 `app.py` 读取后注入到 system prompt 里，让 AI 知道"什么情况下该用哪个工具"。
改这里立即生效，不需要重启。

---

## 配置文件

### `.env` — API Keys（本地，不上传 git）
```
GROQ_API_KEY=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...
```

### `.env.example` — Key 模板（上传 git，占位用）

### `requirements.txt` — Python 依赖列表

---

## AgentState 关键字段

```python
# 路由控制（parse_node 输出）
need_data: bool            # 是否调用 data_node
need_news: bool            # 是否调用 news_node
need_rag: bool             # 是否调用 rag_node
need_scoring: bool         # 是否调用 scoring_node（综合分析=True，简单查价=False）
need_risk: bool            # 是否调用 risk_node
need_comparison: bool      # 是否调用 comparison_node（多股对比）
need_hypothesis: bool      # 是否调用 hypothesis_node（假设推演）
need_reflection: bool      # 是否调用 reflection_node（need_scoring=True 时同步开启）
use_financial_report: bool # 是否触发 financial_report_node

# 财报精读输出
financial_metrics: Optional[dict]
risk_signals: Optional[list]
report_citations: Optional[list]

# 各节点分析结论
stock_data: str            # data_node 输出（原始数据 + 技术面分析）
news: str                  # news_node 输出（新闻摘要 + 情绪判断）
rag_result: str            # rag_node 输出（财务指标提取）
deep_read_result: Optional[dict] # deep_read_node 输出的摘要与批判 JSON
scoring_result: dict       # scoring_node 输出的多维度评分 JSON
risk_result: Optional[dict]
comparison_result: Optional[dict]
hypothesis_result: Optional[dict]

# 最终输出
report: str                # report_node 生成的报告（可被 reflection_node 修订）
final_report: str          # 最终输出（reflection 修订后）

# 并联器
tool_calls: Annotated[List[dict], operator.add]  # 各节点工具调用记录，自动合并
errors: Annotated[List[dict], operator.add]
```

---

## 运行时自动生成

| 目录/文件 | 内容 |
|-----------|------|
| `charts/` | 走势图 PNG，每次生成新图存在这里 |
| `vectorstore/` | ChromaDB 向量库，上传 PDF 后自动建立 |
| `tmp/filings/` | 财报 PDF 下载缓存目录 |
| `token.pickle` | Gmail OAuth 凭证，首次邮件授权后生成 |

---

## 数据流总览

```
用户输入
  ↓
app.py（接收输入）
  ↓
graph.py → parse_node（规划，输出 JSON 调度计划）
  ↓ [条件] financial_report_node（pdfplumber Map + QUALITY_CASCADE Reduce）
  ↓ [条件] deep_read_node（双阶段精读：摘要提取 + 质疑批判）
  ↓ 并行采集
         data_node（yfinance → LLM 技术面分析）
         news_node（Tavily   → LLM 新闻摘要+情绪）
         rag_node （ChromaDB → LLM 财务指标提取）
  ↓ 并行分析（各节点内部条件判断，flag=False 时空返回）
         scoring_node    （CoT 三维评分 → JSON）
         risk_node       （风险矩阵 → JSON）
         comparison_node （多股对比排名 → JSON）
         hypothesis_node （假设情景推演 → JSON）
         reflection_node （报告审核修订）
  ↓ fan-in
         report_node（整合所有结论 → 最终报告）
           ├─ 运行模式：Gemini 2.5 Flash（失败时 fallback → Groq QUALITY_CASCADE）
           └─ 开发模式：Groq QUALITY_CASCADE
  ↓
app.py（显示报告 + 图表 + 工具调用步骤）
```
