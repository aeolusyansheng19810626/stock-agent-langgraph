import React from "react";

export interface Scenario {
  name?:         string;
  probability?:  number | string;
  impact?:       string;
  price_impact?: string;
  target?:       string;
}

function classify(name: string): "bull" | "base" | "bear" {
  if (/乐观|bull|upside/i.test(name)) return "bull";
  if (/悲观|bear|downside/i.test(name)) return "bear";
  return "base";
}

export const VerdictCards: React.FC<{ scenarios: Scenario[] }> = ({ scenarios }) => {
  if (!scenarios.length) return null;

  const sorted: Record<"bull" | "base" | "bear", Scenario | undefined> = {
    bull: undefined, base: undefined, bear: undefined,
  };
  for (const s of scenarios) sorted[classify(s.name ?? "")] = s;

  const order: ("bull" | "base" | "bear")[] = ["bull", "base", "bear"];
  const labels: Record<string, string> = { bull: "乐观情景", base: "基准情景", bear: "悲观情景" };

  return (
    <div className="sx-verdict">
      {order.map((k) => {
        const s = sorted[k];
        return (
          <div className={`sx-verdict-card ${k}`} key={k}>
            <div className="lbl">{labels[k]}{s?.probability !== undefined ? ` · ${s.probability}%` : ""}</div>
            <div className="val">{s?.price_impact || s?.target || "—"}</div>
            {s?.impact && <div className="rng">{s.impact.length > 40 ? s.impact.slice(0, 40) + "…" : s.impact}</div>}
          </div>
        );
      })}
    </div>
  );
};
