import React from "react";

export interface KpiItem {
  label: string;
  value: string;
  sub?:  string;
  dir?:  "up" | "down";
}

export const KpiGrid: React.FC<{ items: KpiItem[] }> = ({ items }) => {
  if (!items.length) return null;
  // Pad to multiples of 4 for the 4-col grid
  const padded = [...items];
  while (padded.length % 4 !== 0) padded.push({ label: "", value: "" });
  return (
    <div className="sx-kpi-grid">
      {padded.map((it, i) => (
        <div className="sx-kpi" key={i}>
          <div className="sx-kpi-label">{it.label}</div>
          <div className="sx-kpi-val">{it.value || "—"}</div>
          {it.sub && <div className={`sx-kpi-sub ${it.dir ?? ""}`}>{it.sub}</div>}
        </div>
      ))}
    </div>
  );
};

const FM_LABELS: Record<string, string> = {
  revenue:             "营业收入",
  revenue_yoy:         "营收同比",
  net_profit:          "归母净利润",
  net_profit_yoy:      "净利同比",
  gross_margin:        "毛利率",
  debt_ratio:          "资产负债率",
  eps:                 "每股收益",
  operating_cash_flow: "经营活动现金流",
};

const KPI_PRIORITY: Array<keyof typeof FM_LABELS> = [
  "revenue", "net_profit", "gross_margin", "eps",
  "revenue_yoy", "net_profit_yoy", "debt_ratio", "operating_cash_flow",
];

const PCT_FIELDS = new Set(["revenue_yoy", "net_profit_yoy"]);

/* Convert AgentState.financial_metrics → KpiItem[] in priority order. */
export function fmToKpis(metrics: Record<string, unknown>): KpiItem[] {
  const items: KpiItem[] = [];
  for (const k of KPI_PRIORITY) {
    const v = metrics[k];
    if (v === null || v === undefined || v === "") continue;
    const value = String(v);
    let dir: "up" | "down" | undefined;
    if (PCT_FIELDS.has(k)) {
      const m = value.match(/([+\-]?)([\d.]+)/);
      if (m) dir = m[1] === "-" || (m[1] === "" && parseFloat(m[2]) < 0) ? "down" : "up";
    }
    items.push({ label: FM_LABELS[k], value, dir });
  }
  return items;
}
