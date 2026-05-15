/* 62-bar pseudo-random candlestick — copied from
 * design_handoff_stockai/reference/stockai-app.jsx:803-848 */
import React from "react";

type Candle = { o: number; c: number; h: number; l: number };

export const CandleChart: React.FC<{
  width?: number;
  height?: number;
  upColor?: string;
  downColor?: string;
}> = ({ width = 560, height = 160, upColor = "var(--up)", downColor = "var(--down)" }) => {
  const N = 62;
  const raw: Candle[] = [];
  let p = 1620;
  let hi = -Infinity, lo = Infinity;
  for (let i = 0; i < N; i++) {
    const trend = Math.sin(i * 0.18) * 14 + Math.cos(i * 0.06) * 10;
    const noise = (Math.sin(i * 7.13) * Math.cos(i * 2.7)) * 18;
    const o = p;
    p = p + trend * 0.3 + noise * 0.5;
    const c = p;
    const h = Math.max(o, c) + Math.abs(Math.sin(i * 3.3)) * 10;
    const l = Math.min(o, c) - Math.abs(Math.cos(i * 2.1)) * 10;
    raw.push({ o, c, h, l });
    hi = Math.max(hi, h); lo = Math.min(lo, l);
  }
  const pad = 2;
  const maxH = Math.max(2, height - 20);
  const range = hi - lo;
  const barW = (width - pad * 2) / N;
  const yOf = (v: number) => 10 + (1 - (v - lo) / range) * (maxH - 10);

  const grids = [0.25, 0.5, 0.75].map((g, i) => (
    <line
      key={i}
      x1={0} x2={width}
      y1={10 + g * (maxH - 10)} y2={10 + g * (maxH - 10)}
      stroke="var(--line)" strokeDasharray="2 4" strokeWidth={0.5}
    />
  ));
  const candles = raw.map((c, i) => {
    const x = pad + i * barW + barW / 2;
    const isUp = c.c >= c.o;
    const color = isUp ? upColor : downColor;
    return (
      <g key={i}>
        <line x1={x} x2={x} y1={yOf(c.h)} y2={yOf(c.l)} stroke={color} strokeWidth={1} opacity={0.7} />
        <rect
          x={x - barW * 0.32}
          y={yOf(Math.max(c.o, c.c))}
          width={barW * 0.64}
          height={Math.max(1, Math.abs(yOf(c.o) - yOf(c.c)))}
          fill={color}
          opacity={isUp ? 0.85 : 0.9}
        />
      </g>
    );
  });

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ display: "block" }}>
      {grids}
      {candles}
    </svg>
  );
};
