import React from "react";
import { Sparkline } from "../report/Sparkline";
import type { Quote } from "../types/sse";

function fmtPrice(p?: number): string {
  if (p === undefined || p === null || Number.isNaN(p)) return "—";
  return Math.abs(p) >= 1000 ? p.toFixed(2) : p.toFixed(2);
}

export const WatchlistItem: React.FC<{
  symbol: string;
  quote?: Quote;
  active: boolean;
  onClick: () => void;
  onRemove: () => void;
}> = ({ symbol, quote, active, onClick, onRemove }) => {
  const pct  = quote?.pct ?? 0;
  const up   = pct >= 0;
  const seed = symbol.split("").reduce((a, c) => a + c.charCodeAt(0), 0);

  return (
    <div className={`sx-watch-item ${active ? "active" : ""}`} onClick={onClick}>
      <span className="sx-watch-sym">{symbol}</span>
      <span className="sx-watch-price">{fmtPrice(quote?.price)}</span>
      <span className="sx-watch-name" title={quote?.name}>
        {quote?.name ?? "—"}
      </span>
      <span className={`sx-watch-chg ${up ? "up" : "down"}`}>
        {up ? "+" : ""}{pct.toFixed(2)}%
      </span>
      {active && <Sparkline seed={seed} up={up} />}
      <button
        className="sx-watch-del"
        onClick={(e) => { e.stopPropagation(); onRemove(); }}
        title="移除"
      >×</button>
    </div>
  );
};
