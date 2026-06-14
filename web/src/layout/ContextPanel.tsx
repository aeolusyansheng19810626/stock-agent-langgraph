import React from "react";
import { QuoteCard } from "../context/QuoteCard";
import { NewsList } from "../context/NewsList";
import { FilingsList } from "../context/FilingsList";
import { useWatchlist } from "../store/watchlist";
import { CopilotChat } from "@copilotkit/react-ui";
import { useUI } from "../store/ui";

export const ContextPanel: React.FC = () => {
  const tab = useUI((s) => s.ctxTab);
  const setTab = useUI((s) => s.setCtxTab);
  const sym   = useWatchlist((s) => s.active);
  const quote = useWatchlist((s) => (sym ? s.quotes[sym] : undefined));

  return (
    <aside className="sx-context">
      <div className="sx-ctx-tabs">
        {([["quote", "行情"], ["news", "资讯"], ["filings", "公告"], ["copilot", "副驾驶"]] as const).map(([k, label]) => (
          <button
            key={k}
            className={`sx-ctx-tab ${tab === k ? "active" : ""}`}
            onClick={() => setTab(k)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="sx-ctx-body">
        {tab === "quote" && (
          <>
            <div className="sx-ctx-block-head">
              <span>当前报价</span>
              <span style={{ color: "var(--up)", display: "inline-flex", alignItems: "center", gap: 4 }}>
                <span className="sx-pulse" style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--up)", display: "inline-block", boxShadow: "0 0 6px var(--up)" }} />
                实时
              </span>
            </div>
            <QuoteCard quote={quote} />
          </>
        )}
        {tab === "news"    && <NewsList />}
        {tab === "filings" && <FilingsList />}
        {tab === "copilot" && (
          <div className="sx-ctx-copilot-wrap">
            <CopilotChat />
          </div>
        )}
      </div>
    </aside>
  );
};
