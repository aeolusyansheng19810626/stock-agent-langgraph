import React from "react";
import { Markdown } from "./Markdown";
import { StepRow } from "./StepRow";
import { ReportCard } from "../report/ReportCard";
import type { AssistantMessage as AsstMsg } from "../store/chat";

export const AssistantMessage: React.FC<{ msg: AsstMsg }> = ({ msg }) => {
  const time = msg.elapsed ? `${msg.elapsed.toFixed(1)}s` : "…";
  // Detect "structured" report — when financial_metrics or scoring_result is present,
  // render the rich ReportCard; otherwise fall back to plain markdown body.
  const hasStructured =
    msg.payloads.financial_report_node ||
    msg.payloads.scoring_node ||
    msg.payloads.report_node;

  return (
    <div className="sx-msg">
      <div className="sx-msg-head">
        <span className="sx-role">● StockAI</span>
        <span className="sx-mono">{msg.finalModel || "…"}</span>
        <span className="sx-mono">{time}</span>
      </div>

      {msg.steps.map((s) => (
        <StepRow key={`${s.step}-${s.tool_name}`} step={s} />
      ))}

      {msg.text || hasStructured ? (
        hasStructured ? (
          <ReportCard message={msg} />
        ) : (
          <div className="sx-msg-body">
            <Markdown source={msg.text} />
            {msg.pending && <span style={{ opacity: 0.4 }}>▌</span>}
          </div>
        )
      ) : null}

      {msg.charts.map((p) => (
        <div className="sx-chart-block" key={p}>
          <img src={`/${p.replace(/^\.?\//, "")}`} alt="走势图" style={{ width: "100%" }} />
        </div>
      ))}

      {msg.errors.length > 0 && (
        <details style={{ marginTop: 8, color: "var(--down)" }}>
          <summary style={{ cursor: "pointer" }}>⚠️ 部分节点出现异常 ({msg.errors.length})</summary>
          <div style={{ padding: "8px 0", fontSize: 11.5 }}>
            {msg.errors.map((e, i) => (
              <div key={i} style={{ fontFamily: "var(--font-mono)" }}>
                {e.node || "?"} / {e.tool || "?"}: {e.message}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
};
