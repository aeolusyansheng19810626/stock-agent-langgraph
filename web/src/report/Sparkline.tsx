/* 24-point pseudo-random sparkline — copied from
 * design_handoff_stockai/reference/stockai-app.jsx:792-857 */
import React from "react";

function sparkPath(seed: number, w = 60, h = 16): string {
  const pts: [number, number][] = [];
  let v = 0.5;
  for (let i = 0; i < 24; i++) {
    const r = Math.sin(seed * 12.9 + i * 0.7) * 0.5 + Math.sin(seed + i * 1.3) * 0.3;
    v = Math.max(0.1, Math.min(0.9, v + r * 0.15));
    pts.push([(i / 23) * w, h - v * h]);
  }
  return "M " + pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L ");
}

export const Sparkline: React.FC<{ seed: number; up: boolean }> = ({ seed, up }) => {
  const col = up ? "var(--up)" : "var(--down)";
  return (
    <svg className="sx-spark" viewBox="0 0 60 16" preserveAspectRatio="none" style={{ width: "100%" }}>
      <path d={sparkPath(seed)} fill="none" stroke={col} strokeWidth={1} opacity={0.85} />
    </svg>
  );
};
