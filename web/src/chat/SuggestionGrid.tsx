import React from "react";

const SUGGESTIONS: { lbl: string; q: string; hint: string }[] = [
  { lbl: "→ 个股分析",  q: "分析一下英伟达 NVDA 的基本面和技术面",     hint: "⌘1" },
  { lbl: "→ 横向对比",  q: "帮我对比苹果 AAPL 和微软 MSFT 的走势",      hint: "⌘2" },
  { lbl: "→ 新闻资讯",  q: "特斯拉最近有什么重要新闻和催化剂？",        hint: "⌘3" },
  { lbl: "→ 邮件报告",  q: "分析完英伟达后发报告到我的邮箱",            hint: "⌘4" },
];

export const SuggestionGrid: React.FC<{ onPick: (q: string) => void }> = ({ onPick }) => (
  <div className="sx-suggest-grid">
    {SUGGESTIONS.map((s) => (
      <div className="sx-suggest" key={s.lbl} onClick={() => onPick(s.q)}>
        <div className="lbl">{s.lbl}</div>
        <div className="q">{s.q}</div>
        <div className="hint">{s.hint}</div>
      </div>
    ))}
  </div>
);
