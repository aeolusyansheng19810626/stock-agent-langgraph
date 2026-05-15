import React from "react";
import { Icon } from "../icons/Icon";
import { useWatchlist } from "../store/watchlist";
import { WatchlistItem } from "./WatchlistItem";

export const Watchlist: React.FC = () => {
  const symbols   = useWatchlist((s) => s.symbols);
  const quotes    = useWatchlist((s) => s.quotes);
  const active    = useWatchlist((s) => s.active);
  const setActive = useWatchlist((s) => s.setActive);
  const add       = useWatchlist((s) => s.add);
  const refresh   = useWatchlist((s) => s.refresh);

  React.useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => clearInterval(t);
  }, [refresh]);

  const onAdd = () => {
    const sym = window.prompt("代码（NVDA / 600519.SS / 00700.HK）：");
    if (sym) add(sym);
  };

  return (
    <div className="sx-sb-section grow">
      <div className="sx-sb-header">
        <span>自选股<span className="sx-count"> · {symbols.length}</span></span>
        <button className="sx-add" onClick={onAdd} title="添加自选">
          <Icon name="plus" size={12} />
        </button>
      </div>
      <div className="sx-watchlist">
        {symbols.map((sym) => (
          <WatchlistItem
            key={sym}
            symbol={sym}
            quote={quotes[sym]}
            active={sym === active}
            onClick={() => setActive(sym)}
          />
        ))}
      </div>
    </div>
  );
};
