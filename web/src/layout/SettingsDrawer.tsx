import React from "react";
import { Icon } from "../icons/Icon";
import { useSettings, type Theme } from "../store/settings";
import { useChat } from "../store/chat";
import { sendEmail } from "../api/client";

const Toggle: React.FC<{ on: boolean; onClick: () => void }> = ({ on, onClick }) => (
  <button className={`sx-toggle ${on ? "on" : ""}`} onClick={onClick} aria-pressed={on} />
);

const SettingRow: React.FC<{
  label: string;
  sub?: string;
  on: boolean;
  onChange: () => void;
}> = ({ label, sub, on, onChange }) => (
  <div className="sx-setting-row">
    <div className="lbl">
      {label}
      {sub && <span className="sub">{sub}</span>}
    </div>
    <Toggle on={on} onClick={onChange} />
  </div>
);

export const SettingsDrawer: React.FC = () => {
  const s        = useSettings();
  const messages = useChat((c) => c.messages);

  // Send the most recent assistant report
  const lastReport = React.useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.text) return m.text;
    }
    return "";
  }, [messages]);

  const onSendNow = async () => {
    if (!s.emailRecipient || !s.emailRecipient.includes("@")) {
      alert("请填入有效的邮箱"); return;
    }
    if (!lastReport) {
      alert("当前没有可发送的报告"); return;
    }
    try {
      const r = await sendEmail(s.emailRecipient, "AI 股票分析报告", lastReport);
      alert(r.ok ? "已发送" : `发送失败：${r.message}`);
    } catch (e) {
      alert(`发送失败：${(e as Error).message}`);
    }
  };

  const themes: { key: Theme; label: string }[] = [
    { key: "amber", label: "墨黑琥珀" },
    { key: "cyan",  label: "电光青"   },
    { key: "ink",   label: "账簿墨绿" },
  ];

  return (
    <>
      <div className={`sx-drawer-bg ${s.settingsOpen ? "open" : ""}`} onClick={() => s.setOpen(false)} />
      <div className={`sx-drawer ${s.settingsOpen ? "open" : ""}`}>
        <div className="sx-drawer-head">
          <h3>工作区设置</h3>
          <button className="sx-icon-btn" onClick={() => s.setOpen(false)} aria-label="关闭">
            <Icon name="x" size={14} />
          </button>
        </div>

        <div className="sx-drawer-body">
          {s.devMode && (
            <div className="sx-dev-banner">
              <Icon name="dev" size={13} />
              开发模式已开启 · Gemini 与思维链关闭，仅使用 Groq
            </div>
          )}

          <div className="sx-setting-group">
            <h4>主题</h4>
            <div style={{ display: "flex", gap: 6 }}>
              {themes.map((t) => (
                <button
                  key={t.key}
                  className="sx-report-act"
                  style={s.theme === t.key
                    ? { color: "var(--accent-fg)", background: "var(--accent)", borderColor: "var(--accent)" }
                    : {}}
                  onClick={() => s.setTheme(t.key)}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <div className="sx-setting-group">
            <h4>分析师行为</h4>
            <SettingRow
              label="开发模式"
              sub="跳过 Gemini，仅用 Groq；省额度，不影响功能"
              on={s.devMode}
              onChange={() => s.toggle("devMode")}
            />
            <SettingRow
              label="引用必标注来源"
              sub="所有数据点必须附上 [chunk] 引用"
              on={s.citeSources}
              onChange={() => s.toggle("citeSources")}
            />
            <SettingRow
              label="不给出明确买卖建议"
              sub="只给情景与风险，禁用 BUY/SELL 标签"
              on={s.noBuySell}
              onChange={() => s.toggle("noBuySell")}
            />
            <SettingRow
              label="红涨绿跌（A 股惯例）"
              sub="切换涨跌色配置"
              on={s.riseRedFallGreen}
              onChange={() => s.toggle("riseRedFallGreen")}
            />
          </div>

          <div className="sx-setting-group">
            <h4>报告投递</h4>
            <SettingRow
              label="自动邮件发送"
              sub="生成报告后自动发到下方邮箱"
              on={s.autoEmail}
              onChange={() => s.toggle("autoEmail")}
            />

            <div className="sx-email-block">
              <div className="sx-email-block-title">收件邮箱</div>
              <input
                className="sx-email-input"
                type="email"
                placeholder="you@example.com"
                value={s.emailRecipient}
                onChange={(e) => s.setEmail(e.target.value)}
              />
              <button className="sx-send-btn" onClick={onSendNow} disabled={!lastReport}>
                <Icon name="mail" size={12} /> 立即发送当前报告
              </button>
            </div>
          </div>

          <div className="sx-setting-group">
            <h4>数据源</h4>
            <SettingRow
              label="实时行情（Level-2 · Wind）"
              sub="盘中实时价格 + 五档行情"
              on={s.livel2}
              onChange={() => s.toggle("livel2")}
            />
            <SettingRow
              label="盘前 / 盘后数据"
              on={s.premarket}
              onChange={() => s.toggle("premarket")}
            />
            <SettingRow
              label="巨潮资讯 · 自动同步公告"
              on={s.cninfoSync}
              onChange={() => s.toggle("cninfoSync")}
            />
          </div>
        </div>
      </div>
    </>
  );
};
