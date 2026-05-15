import React from "react";

/* TODO: 接入实时资讯源（财联社 / 新华财经 RSS / Tavily 实时新闻）。
 * 当前使用设计稿固定 mock，确保上下文栏视觉完整。 */
const FIXTURES = [
  { src: "新华财经",   time: "14分钟前", title: "茅台一季报营收同比 +18.4%，超出市场一致预期", dir: "bull" },
  { src: "财联社",     time: "1小时前",  title: "央行 MLF 净投放 2000 亿元，呵护流动性意图明显", dir: "bull" },
  { src: "21财经",     time: "2小时前",  title: "机构下调白酒板块 2027 年盈利预测，关注库存周期", dir: "bear" },
  { src: "彭博",       time: "4小时前",  title: "北向资金连续 6 日净流入，单日规模创年内新高",  dir: "bull" },
  { src: "华尔街见闻", time: "6小时前",  title: "美联储会议纪要释放鸽派信号，9 月降息概率升至 78%", dir: "neutral" },
];

export const NewsList: React.FC = () => (
  <div>
    <div className="sx-ctx-block-head">
      <span>市场资讯</span>
      <span style={{ fontSize: 10 }}>TODO · 接入实时源</span>
    </div>
    {FIXTURES.map((n, i) => (
      <div className="sx-news-item" key={i}>
        <div className="sx-news-meta">
          <span className={`sx-news-dot ${n.dir}`} />
          <span className="sx-news-src">{n.src}</span>
          <span className="mono">{n.time}</span>
        </div>
        <div className="sx-news-title">{n.title}</div>
      </div>
    ))}
  </div>
);
