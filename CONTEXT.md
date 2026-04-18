# 项目上下文（给 AI 助手用，复制此文件内容粘贴到对话开头）

## 项目概述
`stock-agent-langgraph`：基于 LangGraph 的股票分析多 Agent 系统。
Streamlit Web UI，用户提问 → 多节点并行分析 → 生成报告。

---

## 技术栈
- LangGraph + Streamlit
- LLM：Groq API（主）+ Google Gemini 2.5 Flash（report_node 优先）
- 向量库：ChromaDB（vectorstore/）
- PDF 解析：pdfplumber + Vision fallback

---

## 关键文件

| 文件 | 职责 | 能改吗 |
|------|------|--------|
| `graph.py` | 所有节点 + 图构建，核心文件 | ✅ |
| `nodes/financial_report_node.py` | PDF 财报精读节点 | ✅ |
| `app.py` | Streamlit UI + 路由入口 | ✅ |
| `tools.py` | LangChain Tool 定义 | ❌ 不要改 |
| `components/stock_ticker.py` | 侧边栏实时股价组件 | ✅ |

---

## 图结构

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

---

## 已实现的节点（不要重复实现）

| 节点 | 触发条件 | 模型 |
|------|---------|------|
| parse_node | 每次必跑 | QUALITY_CASCADE |
| financial_report_node | use_financial_report=True | Map:TIER_LOW / Reduce:QUALITY_CASCADE |
| data_node | need_data=True | TIER_LOW |
| news_node | need_news=True | TIER_LOW |
| rag_node | need_rag=True | QUALITY_CASCADE |
| scoring_node | need_scoring=True | QUALITY_CASCADE |
| risk_node | need_risk=True | TIER_TOP |
| comparison_node | need_comparison=True | TIER_TOP |
| hypothesis_node | need_hypothesis=True | TIER_TOP |
| reflection_node | need_reflection=True | TIER_TOP |
| deep_read_node | need_deep_read=True | S1:QUALITY_CASCADE / S2:[TOP,UPPER_MID,MID] |
| report_node | 每次必跑 | Gemini → Groq fallback |

---

## 模型分级

```python
TIER_TOP       = "openai/gpt-oss-120b"
TIER_UPPER_MID = "openai/gpt-oss-20b"
TIER_MID       = "qwen/qwen3-32b"
TIER_LOW       = "meta-llama/llama-4-scout-17b-16e-instruct"
TIER_DEBUG     = "llama-3.1-8b-instant"
QUALITY_CASCADE = [TIER_TOP, TIER_UPPER_MID, TIER_MID, TIER_LOW, TIER_DEBUG]
```

---

## 开发约定（必须遵守）

### 新增节点的完整步骤（不能漏）
1. `graph.py` 定义节点函数
2. `AgentState` 添加 `need_xxx: bool` + `xxx_result: Optional[...]`
3. `PLAN_SYSTEM` prompt 添加路由规则
4. `_normalize_plan` 的 **return dict** 添加 `need_xxx`（**历史上漏过，必查**）
5. `parse_node` 的 result dict 添加 `need_xxx`
6. `build_graph()` 注册节点和边
7. `report_node` 读取结果追加到报告

### 节点函数模板
```python
def xxx_node(state: AgentState) -> dict:
    if not state.get("need_xxx"):
        return {"xxx_result": None, "tool_calls": [], "errors": []}
    # 逻辑...
    return {"xxx_result": result, "tool_calls": [...], "errors": [...]}
```

### 禁止事项
- ❌ **禁止重写整个文件**：只允许精确修改涉及变更的函数或段落，不得整体替换 `graph.py` / `app.py` 等核心文件
- ❌ 不要修改 `tools.py`
- ❌ 不要改动 `QUALITY_CASCADE` 顺序
- ❌ 节点内读 API Key 统一用 `state.get("groq_api_key") or os.getenv(...)`
- ❌ LLM 输出 JSON 的 prompt 必须加「严格输出纯 JSON，禁止任何其他文字」
- ❌ 不要自己写 JSON 解析，统一用 `_parse_json_from_text()`

### 调试 / 自动测试规范
- 执行自动测试（`test_node.py` / `test_parse.py`）或调试时，**必须先将模型切换为 `TIER_DEBUG`**（`llama-3.1-8b-instant`），避免消耗付费模型额度
- 测试通过后再将模型改回原来的值，提交时确保代码中不含 `TIER_DEBUG` 的临时赋值
- 切换方式：在被测节点顶部临时替换模型常量，或直接修改调用处，测试完毕立即还原

### Commit 规范
commit message 必须包含：
1. **修改者名字**：Claude Code / Gemini / Codex / Copilot 等
2. **修改内容摘要**：具体改了什么

格式示例：
```
Claude Code: add deep_read_node and update AgentState fields
Gemini: fix scoring_node cascade model assignment
Codex: refactor report_node to support hypothesis_result
```

---

## 环境变量（.env）
```
GROQ_API_KEY=
GEMINI_API_KEY=
TAVILY_API_KEY=
```
Gmail 发送用 OAuth，首次运行生成 `token.pickle`，无需配置密码。

---

## 当前进度
- ✅ 多维度综合评分系统（scoring_node）
- ✅ 风险因子深度挖掘（risk_node）
- ✅ 对比分析（comparison_node）
- ✅ 自我反思 Agent（reflection_node）
- ✅ 假设推演（hypothesis_node）
- ✅ 论文/财报精读（deep_read_node）
- ✅ app.py 路由修复（财报请求强制走 LangGraph pipeline）
