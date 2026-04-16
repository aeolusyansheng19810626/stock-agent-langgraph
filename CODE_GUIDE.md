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
整个项目最核心的文件。定义了 5 个节点，每个节点都是真正的 Agent：

```
parse_node    → 理解用户问题，决定要调用哪些 agent（输出 JSON 调度计划）
data_node     → 调用 yfinance 获取股价数据，再用 LLM 做技术面分析
news_node     → 调用 Tavily 搜索新闻，再用 LLM 做摘要和情绪判断
rag_node      → 调用 ChromaDB 检索财报，再用 LLM 提取关键财务指标
scoring_node  → 汇总三路预分析结论，Chain-of-Thought 推理输出多维度评分 JSON
report_node   → 整合评分结论，生成带评级摘要的最终投资分析报告
```

`parse_node` 执行完后，`data/news/rag` 三个节点**并行执行**，
全部完成后进入 `scoring_node` 评分，再由 `report_node` 生成最终报告。

`scoring_node` 输出结构化 JSON（财务/情绪/技术三维评分 + 最终评级 + 置信度），
`report_node` 在报告顶部展示评级摘要，正文中引用推理链。

模型分工（在文件顶部集中配置，改常量即可）：
- `data_node / news_node` → LLaMA（快速模型，via Groq）
- `parse_node / rag_node / scoring_node` → openai/gpt-oss-120b（需要推理能力）
- `report_node` 主力 → Gemini 2.5 Flash（运行模式）
- `report_node` fallback → openai/gpt-oss-120b（via Groq）

---

### `tools.py` — 工具层（不要改这个文件）
定义了 AI 可以调用的 5 个工具函数：

| 工具 | 作用 |
|------|------|
| `get_stock_data` | 获取实时股价、PE、52周高低等 |
| `get_stock_history` | 生成历史走势图（保存为 PNG） |
| `search_web` | 用 Tavily 搜索网络新闻 |
| `search_documents` | 在已上传的 PDF 里检索内容 |
| `send_email_report` | 用 Gmail 发送分析报告邮件 |

这些函数被 `graph.py` 的各节点直接调用。

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

## 运行时自动生成

| 目录/文件 | 内容 |
|-----------|------|
| `charts/` | 走势图 PNG，每次生成新图存在这里 |
| `vectorstore/` | ChromaDB 向量库，上传 PDF 后自动建立 |
| `token.pickle` | Gmail OAuth 凭证，首次邮件授权后生成 |

---

## 数据流总览

```
用户输入
  ↓
app.py（接收输入）
  ↓
graph.py → parse_node（规划，输出 JSON 调度计划）
              ↓ 并行
         data_node（yfinance → LLM 技术面分析）
         news_node（Tavily   → LLM 新闻摘要）
         rag_node （ChromaDB → LLM 财务提取）
              ↓ 汇总预分析结论
         scoring_node（CoT 多维度评分 → JSON）
              ↓
         report_node（生成带评级摘要的最终报告）
  ↓
app.py（显示报告 + 图表）
```
