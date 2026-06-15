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

  const [adding, setAdding] = React.useState(false);
  const [input, setInput]   = React.useState("");
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => clearInterval(t);
  }, [refresh]);

  React.useEffect(() => {
    if (adding) inputRef.current?.focus();
  }, [adding]);

  const commit = () => {
    const sym = input.trim().toUpperCase();
    if (sym) add(sym);
    setInput("");
    setAdding(false);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    if (e.key === "Escape") { setInput(""); setAdding(false); }
  };

  return (
    <div className="sx-sb-section grow">
      <div className="sx-sb-header">
        <span>自选股<span className="sx-count"> · {symbols.length}</span></span>
        {!adding && (
          <button className="sx-add" onClick={() => setAdding(true)} title="添加自选">
            <Icon name="plus" size={12} />
          </button>
        )}
      </div>

      {adding && (
        <div className="sx-watchlist-add">
          <input
            ref={inputRef}
            className="sx-watchlist-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            onBlur={commit}
            placeholder="NVDA / ^N225 / 600519.SS"
            maxLength={20}
          />
        </div>
      )}

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
