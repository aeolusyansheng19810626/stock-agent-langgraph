import React from "react";
import { useCopilotAction } from "@copilotkit/react-core";
import { useSettings } from "../store/settings";
import { useWatchlist } from "../store/watchlist";
import { useUI } from "../store/ui";
import type { Theme } from "../store/settings";
import type { CtxTab } from "../store/ui";

export const CopilotActions: React.FC = () => {
  useCopilotAction({
    name: "setTheme",
    description: "切换界面主题。amber=墨黑琥珀暗色, cyan=电光青色, ink=账簿墨绿色",
    parameters: [{ name: "theme", type: "string", enum: ["amber", "cyan", "ink"], required: true, description: "主题名称" }],
    handler: ({ theme }) => { useSettings.getState().setTheme(theme as Theme); },
  });

  useCopilotAction({
    name: "addToWatchlist",
    description: "把股票代码加入自选股列表，如 NVDA、AAPL、600519.SS",
    parameters: [{ name: "symbol", type: "string", required: true, description: "股票代码（大写）" }],
    handler: ({ symbol }) => { useWatchlist.getState().add(symbol); },
  });

  useCopilotAction({
    name: "removeFromWatchlist",
    description: "从自选股列表移除股票代码",
    parameters: [{ name: "symbol", type: "string", required: true, description: "股票代码" }],
    handler: ({ symbol }) => { useWatchlist.getState().remove(symbol); },
  });

  useCopilotAction({
    name: "switchContextTab",
    description: "切换右侧上下文面板标签页：quote=行情, news=资讯, filings=公告, copilot=副驾驶",
    parameters: [{ name: "tab", type: "string", enum: ["quote", "news", "filings", "copilot"], required: true, description: "tab 名称" }],
    handler: ({ tab }) => { useUI.getState().setCtxTab(tab as CtxTab); },
  });

  useCopilotAction({
    name: "setDevMode",
    description: "开启或关闭开发模式（devMode=true 时跳过 Gemini 仅用 Groq）",
    parameters: [{ name: "enabled", type: "boolean", required: true, description: "是否开启开发模式" }],
    handler: ({ enabled }) => {
      const s = useSettings.getState();
      if (s.devMode !== enabled) s.toggle("devMode");
    },
  });

  return null;
};
