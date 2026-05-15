import React from "react";
import type { ToolCallData } from "../types/sse";

const TOOL_LABEL: Record<string, string> = {
  get_stock_data:    "获取股票实时数据",
  search_web:        "搜索网络新闻",
  get_stock_history: "获取历史走势图",
  search_documents:  "检索财报文档",
  send_email_report: "发送邮件报告",
};

export const StepRow: React.FC<{ step: ToolCallData }> = ({ step }) => {
  const isLLM = step.tool_name === "llm";
  const label = TOOL_LABEL[step.tool_name] ?? step.tool_name;
  const argsStr = (() => {
    try {
      const s = JSON.stringify(step.tool_args);
      return s.length > 80 ? s.slice(0, 80) + "…" : s;
    } catch { return ""; }
  })();
  return (
    <div className="sx-step-row">
      <span className="sx-step-no">STEP {step.step}</span>
      <span className={`sx-step-name ${isLLM ? "llm" : ""}`}>{label}</span>
      <span className="sx-step-args">{argsStr}</span>
      {step.retries > 0 && <span className="sx-step-retry">⟳ {step.retries}/3</span>}
      <span className="sx-step-status">✓ 完成</span>
    </div>
  );
};
