import React from "react";
import type { Quote } from "../types/sse";

function fmtNum(v: unknown, decimals = 2): string {
  if (v === null || v === undefined || v === "") return "—";
  const n = typeof v === "number" ? v : parseFloat(String(v));
  if (Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1e8) return `${(n / 1e8).toFixed(2)}亿`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e4).toFixed(2)}万`;
  return n.toFixed(decimals);
}

function fmtPct(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  const n = typeof v === "number" ? v : parseFloat(String(v));
  if (Number.isNaN(n)) return "—";
  return `${n.toFixed(2)}%`;
}

export const QuoteCard: React.FC<{ quote?: Quote }> = ({ quote }) => {
  if (!quote || quote.error) {
    return (
      <div className="sx-quote">
        <div className="sx-quote-name">未选择股票或数据获取失败</div>
      </div>
    );
  }
  const up = (quote.pct ?? 0) >= 0;
  return (
    <div className="sx-quote">
      <div className="sx-quote-head">
        <span className="sx-quote-sym">{quote.symbol}</span>
        <span className="sx-quote-exch">{quote.exchange ?? ""}</span>
      </div>
      <div className="sx-quote-name">{quote.name ?? quote.symbol}</div>
      <div>
        <span className="sx-quote-price">{fmtNum(quote.price)}</span>
        <span className={`sx-quote-chg ${up ? "up" : "down"}`} style={{ marginLeft: 10 }}>
          {up ? "+" : ""}{fmtNum(quote.change)} ({up ? "+" : ""}{fmtNum(quote.pct)}%) {up ? "▲" : "▼"}
        </span>
      </div>

      <div className="sx-stats">
        <div className="sx-stat"><div className="sx-stat-lbl">今开</div><div className="sx-stat-val">{fmtNum(quote.open)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">昨收</div><div className="sx-stat-val">{fmtNum(quote.prevClose)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">最高</div><div className="sx-stat-val">{fmtNum(quote.high)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">最低</div><div className="sx-stat-val">{fmtNum(quote.low)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">成交量</div><div className="sx-stat-val">{fmtNum(quote.volume)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">总市值</div><div className="sx-stat-val">{fmtNum(quote.marketCap)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">52周高</div><div className="sx-stat-val">{fmtNum(quote.high52w)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">52周低</div><div className="sx-stat-val">{fmtNum(quote.low52w)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">市盈率TTM</div><div className="sx-stat-val">{fmtNum(quote.pe)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">市净率</div><div className="sx-stat-val">{fmtNum(quote.pb)}</div></div>
        <div className="sx-stat"><div className="sx-stat-lbl">股息率</div><div className="sx-stat-val">{fmtPct((quote.divYield ?? 0) * 100)}</div></div>
      </div>
    </div>
  );
};
