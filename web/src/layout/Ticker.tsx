import React from "react";

const SYMBOLS = "000001.SS,399001.SZ,399006.SZ,^HSI,^GSPC,^NDX,^N225,1306.T,CNY=X,GC=F,BZ=F";

const LABELS: Record<string, string> = {
  "000001.SS": "上证指数",
  "399001.SZ": "深证成指",
  "399006.SZ": "创业板指",
  "^HSI":      "恒生指数",
  "^GSPC":     "标普500",
  "^NDX":      "纳斯达克100",
  "^N225":     "日经225",
  "1306.T":    "TOPIX",
  "CNY=X":     "美元/人民币",
  "GC=F":      "COMEX金",
  "BZ=F":      "布伦特原油",
};

interface TickerItem {
  sym: string;
  price: string;
  chg: string;
  dir: "up" | "down" | "flat";
}

function fmt(symbol: string, price: number): string {
  if (symbol === "CNY=X") return price.toFixed(4);
  if (price >= 1000) return price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return price.toFixed(2);
}

async function fetchTicker(): Promise<TickerItem[]> {
  const res = await fetch(`/api/quote?symbols=${SYMBOLS}`);
  if (!res.ok) return [];
  const data = await res.json();
  const quotes: any[] = data.quotes ?? [];
  return quotes
    .filter((q) => q.price != null)
    .map((q) => {
      const pct: number = q.pct ?? 0;
      return {
        sym:   LABELS[q.symbol] ?? q.symbol,
        price: fmt(q.symbol, q.price),
        chg:   (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%",
        dir:   pct > 0 ? "up" : pct < 0 ? "down" : "flat",
      };
    });
}

export const Ticker: React.FC = () => {
  const [items, setItems] = React.useState<TickerItem[]>([]);

  React.useEffect(() => {
    fetchTicker().then(setItems);
    const id = setInterval(() => fetchTicker().then(setItems), 90_000);
    return () => clearInterval(id);
  }, []);

  if (items.length === 0) return <div className="sx-ticker" />;

  const doubled = [...items, ...items];
  return (
    <div className="sx-ticker">
      <div className="sx-ticker-label">
        <span className="sx-pulse" />
        实时行情
      </div>
      <div className="sx-ticker-track">
        {doubled.map((it, i) => (
          <div className="sx-ticker-item" key={i}>
            <span className="sx-ticker-sym">{it.sym}</span>
            <span className="sx-ticker-price">{it.price}</span>
            <span className={`sx-ticker-chg ${it.dir}`}>{it.chg}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
