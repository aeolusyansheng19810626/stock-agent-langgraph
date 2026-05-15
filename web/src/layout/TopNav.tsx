import React from "react";
import { Icon } from "../icons/Icon";
import { useSettings } from "../store/settings";

type Tab = { key: string; label: string; live?: boolean };
const TABS: Tab[] = [
  { key: "ai",     label: "AI 分析", live: true },
  { key: "market", label: "行情" },
  { key: "watch",  label: "自选" },
  { key: "news",   label: "资讯" },
  { key: "docs",   label: "文档" },
  { key: "reports",label: "报告" },
];

export const TopNav: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState<string>("ai");
  const setOpen = useSettings((s) => s.setOpen);

  return (
    <header className="sx-topnav">
      <div className="sx-brand">
        <div className="sx-logo">S</div>
        <div className="sx-brand-name">StockAI</div>
        <div className="sx-brand-tag">TERMINAL</div>
      </div>

      <nav className="sx-nav">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`sx-nav-item ${activeTab === t.key ? "active" : ""}`}
            onClick={() => setActiveTab(t.key)}
          >
            {t.label}
            {t.live && <span className="sx-dot" />}
          </button>
        ))}
      </nav>

      <div className="sx-topnav-right">
        <div className="sx-top-search">
          <Icon name="search" size={13} />
          <input placeholder="搜索代码、公司名、文档…" />
          <span className="sx-kbd">⌘K</span>
        </div>
        <button className="sx-icon-btn" aria-label="通知">
          <Icon name="bell" />
          <span className="sx-badge" />
        </button>
        <button className="sx-icon-btn" aria-label="设置" onClick={() => setOpen(true)}>
          <Icon name="settings" />
        </button>
        <div className="sx-avatar">陈</div>
      </div>
    </header>
  );
};
