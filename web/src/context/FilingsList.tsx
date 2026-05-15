import React from "react";
import { useWatchlist } from "../store/watchlist";

/* TODO: 接入巨潮资讯（A股）/ SEC EDGAR（美股）公告抓取。
 * 当前 mock 数据展示样式，与已上传 PDF 文档列表呼应。 */
const FIXTURES = [
  { type: "季报", desc: "2026 年第一季度报告",  date: "2026-04-25", current: true  },
  { type: "公告", desc: "关于现金分红派息的公告", date: "2026-04-12", current: false },
  { type: "年报", desc: "2025 年年度报告",        date: "2026-03-28", current: false },
  { type: "8-K",  desc: "重大事项披露",           date: "2026-03-10", current: false },
];

export const FilingsList: React.FC = () => {
  const sym = useWatchlist((s) => s.active);
  return (
    <div>
      <div className="sx-ctx-block-head">
        <span>近期公告 · {sym ?? "—"}</span>
        <span style={{ fontSize: 10 }}>TODO · 接入巨潮 / SEC</span>
      </div>
      {FIXTURES.map((f, i) => (
        <div className="sx-news-item" key={i}>
          <div className="sx-news-meta">
            <span className={`sx-news-dot ${f.current ? "current" : ""}`} />
            <span className="sx-news-src">{f.type}</span>
            <span className="mono">{f.date}</span>
            {f.current && <span style={{ color: "var(--accent)", fontSize: 10 }}>● 当前上下文</span>}
          </div>
          <div className="sx-news-title">{f.desc}</div>
        </div>
      ))}
    </div>
  );
};
