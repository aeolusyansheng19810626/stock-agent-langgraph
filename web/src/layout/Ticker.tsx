import React from "react";

/* Static index/macro list — yfinance can serve some of these (^SSEC ^HSI etc),
 * but for visual correctness we keep the design-fixture values until M10
 * wires real-time poll. North-bound flow has no free data source.
 *  TODO: 接入北向资金数据源 */
const ITEMS: { sym: string; price: string; chg: string; dir: "up" | "down" }[] = [
  { sym: "上证指数",     price: "3,284.62",  chg: "+0.62%",   dir: "up"   },
  { sym: "深证成指",     price: "10,418.30", chg: "+0.94%",   dir: "up"   },
  { sym: "创业板指",     price: "2,106.78",  chg: "+1.34%",   dir: "up"   },
  { sym: "恒生指数",     price: "19,842.18", chg: "-0.54%",   dir: "down" },
  { sym: "科创50",       price: "942.18",    chg: "+1.82%",   dir: "up"   },
  { sym: "美元/人民币",  price: "7.214",     chg: "-0.08%",   dir: "down" },
  { sym: "北向资金",     price: "+38.4亿",   chg: "净流入",   dir: "up"   },
  { sym: "COMEX金",      price: "2,712.30",  chg: "+0.18%",   dir: "up"   },
  { sym: "布伦特原油",   price: "71.26",     chg: "-0.84%",   dir: "down" },
  { sym: "10年期国债",   price: "2.118%",    chg: "-1bp",     dir: "down" },
];

export const Ticker: React.FC = () => {
  // Duplicate the array to make translateX(-50%) loop seamless.
  const doubled = [...ITEMS, ...ITEMS];
  return (
    <div className="sx-ticker">
      <div className="sx-ticker-label">
        <span className="sx-pulse" />
        实时 · A 股盘中
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
