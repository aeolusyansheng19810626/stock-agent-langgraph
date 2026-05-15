import React from "react";
import { Icon } from "../icons/Icon";
import { Markdown } from "../chat/Markdown";
import { useSettings } from "../store/settings";
import { useWatchlist } from "../store/watchlist";
import { sendEmail } from "../api/client";
import { CandleChart } from "./CandleChart";
import { KpiGrid, fmToKpis } from "./KpiGrid";
import { VerdictCards, type Scenario } from "./VerdictCards";
import type { AssistantMessage } from "../store/chat";

export const ReportCard: React.FC<{ message: AssistantMessage }> = ({ message }) => {
  const recipient = useSettings((s) => s.emailRecipient);
  const setOpen   = useSettings((s) => s.setOpen);
  const activeSym = useWatchlist((s) => s.active);
  const quote     = useWatchlist((s) => (activeSym ? s.quotes[activeSym] : undefined));
  const [range, setRange] = React.useState<"日" | "周" | "1月" | "3月" | "1年" | "5年">("3月");

  const onCopy     = () => navigator.clipboard?.writeText(message.text);
  const onDownload = () => {
    const blob = new Blob([message.text], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `stockai_report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const onEmail = async () => {
    if (!recipient || !recipient.includes("@")) { setOpen(true); return; }
    try {
      const r = await sendEmail(recipient, "AI 股票分析报告", message.text);
      alert(r.ok ? "已发送" : `发送失败：${r.message}`);
    } catch (e) { alert(`发送失败：${(e as Error).message}`); }
  };

  // Pull structured payloads (filled via SSE node.complete events).
  const fm  = (message.payloads.financial_report_node?.financial_metrics
            || message.payloads.report_node?.financial_metrics) as Record<string, unknown> | undefined;
  const hyp = (message.payloads.hypothesis_node?.hypothesis_result
            || message.payloads.report_node?.hypothesis_result) as { scenarios?: Scenario[]; conclusion?: string } | undefined;
  const sco = (message.payloads.scoring_node?.scoring_result
            || message.payloads.report_node?.scoring_result) as Record<string, unknown> | undefined;

  const kpis = fm ? fmToKpis(fm) : [];

  const reportTitle = quote?.name ? `${quote.name} · 深度分析` : "AI 股票分析报告";

  // Format 0.0188 → "+0.94%"
  const pctStr = quote?.pct !== undefined
    ? `${quote.pct >= 0 ? "+" : ""}${quote.pct.toFixed(2)}%`
    : "";

  return (
    <div className="sx-report">
      <div className="sx-report-head">
        <div className="title">
          {reportTitle}
          <span className="tag">AI 研报</span>
        </div>
        <div className="sx-report-actions">
          <button className="sx-report-act" onClick={onCopy}>
            <Icon name="copy" size={12} /> 复制
          </button>
          <button className="sx-report-act" onClick={onDownload}>
            <Icon name="download" size={12} /> 导出
          </button>
          <button className="sx-report-act primary" onClick={onEmail}>
            <Icon name="mail" size={12} /> 邮件发送
          </button>
        </div>
      </div>

      <div className="sx-report-body">
        {kpis.length > 0 && <KpiGrid items={kpis} />}

        {sco && sco.final_rating !== undefined && (
          <div className="sx-kpi-grid" style={{ marginTop: kpis.length ? -16 : 0, marginBottom: 16 }}>
            <div className="sx-kpi">
              <div className="sx-kpi-label">综合评级</div>
              <div className="sx-kpi-val">{String(sco.final_rating)}</div>
              <div className="sx-kpi-sub">置信度 <span className="num">{String(sco.confidence ?? "—")}%</span></div>
            </div>
            <div className="sx-kpi">
              <div className="sx-kpi-label">财务评分</div>
              <div className="sx-kpi-val">{String(sco.financial_score ?? "—")}</div>
              <div className="sx-kpi-sub">/ 10</div>
            </div>
            <div className="sx-kpi">
              <div className="sx-kpi-label">情绪评分</div>
              <div className="sx-kpi-val">{String(sco.sentiment_score ?? "—")}</div>
              <div className="sx-kpi-sub">/ 10</div>
            </div>
            <div className="sx-kpi">
              <div className="sx-kpi-label">技术评分</div>
              <div className="sx-kpi-val">{String(sco.technical_score ?? "—")}</div>
              <div className="sx-kpi-sub">/ 10</div>
            </div>
          </div>
        )}

        {message.charts.length > 0 ? (
          <div className="sx-chart-block">
            <div className="sx-chart-head">
              <div>
                <span className="sym">{activeSym ?? "—"}</span>
                <span className="price" style={{ marginLeft: 10 }}>{quote?.price?.toFixed(2) ?? "—"}</span>
                <span className={`chg ${quote && quote.pct! >= 0 ? "up" : "down"}`}>{pctStr}</span>
              </div>
            </div>
            <img src={`/${message.charts[0].replace(/^\.?\//, "")}`} alt="走势图" style={{ width: "100%", borderRadius: 4 }} />
          </div>
        ) : activeSym ? (
          <div className="sx-chart-block">
            <div className="sx-chart-head">
              <div>
                <span className="sym">{activeSym}</span>
                <span className="price" style={{ marginLeft: 10 }}>{quote?.price?.toFixed(2) ?? "—"}</span>
                <span className={`chg ${quote && quote.pct! >= 0 ? "up" : "down"}`}>{pctStr}</span>
              </div>
              <div className="ranges">
                {(["日", "周", "1月", "3月", "1年", "5年"] as const).map((r) => (
                  <button key={r} className={`sx-range ${range === r ? "active" : ""}`} onClick={() => setRange(r)}>
                    {r}
                  </button>
                ))}
              </div>
            </div>
            <CandleChart width={620} height={160} />
          </div>
        ) : null}

        <div className="sx-sec">
          <Markdown source={message.text} />
          {message.pending && <span style={{ opacity: 0.4 }}>▌</span>}
        </div>

        {hyp?.scenarios && hyp.scenarios.length > 0 && (
          <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--line)" }}>
            <div className="sx-sec-title">
              <span className="num">→</span> 三档情景估值
            </div>
            <VerdictCards scenarios={hyp.scenarios} />
          </div>
        )}
      </div>
    </div>
  );
};
