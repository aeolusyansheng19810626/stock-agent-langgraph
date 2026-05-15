/* Inline 16px stroke icons — copied from
 * design_handoff_stockai/reference/stockai-app.jsx:860-880 */
import React from "react";

export type IconName =
  | "search" | "plus" | "upload" | "bell" | "settings" | "send"
  | "attach" | "chart" | "ai" | "copy" | "download" | "mail" | "x" | "dev";

export const Icon: React.FC<{ name: IconName; size?: number }> = ({ name, size = 14 }) => {
  const s = { width: size, height: size } as const;
  const stroke = {
    fill: "none", stroke: "currentColor", strokeWidth: 1.5,
    strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
  };
  switch (name) {
    case "search":   return <svg {...s} viewBox="0 0 16 16"><circle cx="7" cy="7" r="5" {...stroke}/><path d="M11 11l3 3" {...stroke}/></svg>;
    case "plus":     return <svg {...s} viewBox="0 0 16 16"><path d="M8 3v10M3 8h10" {...stroke}/></svg>;
    case "upload":   return <svg {...s} viewBox="0 0 16 16"><path d="M8 2v9M4 6l4-4 4 4M2 14h12" {...stroke}/></svg>;
    case "bell":     return <svg {...s} viewBox="0 0 16 16"><path d="M4 12V7a4 4 0 118 0v5l1 2H3zM6 14a2 2 0 004 0" {...stroke}/></svg>;
    case "settings": return <svg {...s} viewBox="0 0 16 16"><circle cx="8" cy="8" r="2" {...stroke}/><path d="M8 2v1M8 13v1M2 8h1M13 8h1M3.5 3.5l.7.7M11.8 11.8l.7.7M3.5 12.5l.7-.7M11.8 4.2l.7-.7" {...stroke}/></svg>;
    case "send":     return <svg {...s} viewBox="0 0 16 16"><path d="M14 2L2 7l5 2 2 5z" {...stroke}/><path d="M14 2L7 9" {...stroke}/></svg>;
    case "attach":   return <svg {...s} viewBox="0 0 16 16"><path d="M12 7l-5 5a2.5 2.5 0 01-3.5-3.5l6-6a3.5 3.5 0 115 5L8 14" {...stroke}/></svg>;
    case "chart":    return <svg {...s} viewBox="0 0 16 16"><path d="M2 14V2M2 14h12M5 11V7M8 11V4M11 11V9" {...stroke}/></svg>;
    case "ai":       return <svg {...s} viewBox="0 0 16 16"><path d="M8 2l1.5 3L13 6.5 9.5 8 8 11 6.5 8 3 6.5 6.5 5z" {...stroke}/><path d="M13 11l.5 1 1 .5-1 .5-.5 1-.5-1-1-.5 1-.5z" {...stroke}/></svg>;
    case "copy":     return <svg {...s} viewBox="0 0 16 16"><rect x="5" y="5" width="8" height="8" rx="1" {...stroke}/><path d="M3 10V4a1 1 0 011-1h6" {...stroke}/></svg>;
    case "download": return <svg {...s} viewBox="0 0 16 16"><path d="M8 2v9M4 8l4 4 4-4M2 14h12" {...stroke}/></svg>;
    case "mail":     return <svg {...s} viewBox="0 0 16 16"><rect x="2" y="4" width="12" height="9" rx="1" {...stroke}/><path d="M2 5l6 4 6-4" {...stroke}/></svg>;
    case "x":        return <svg {...s} viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" {...stroke}/></svg>;
    case "dev":      return <svg {...s} viewBox="0 0 16 16"><path d="M6 5L3 8l3 3M10 5l3 3-3 3" {...stroke}/></svg>;
    default: return null;
  }
};
