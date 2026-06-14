import React from "react";
import { useCopilotReadable } from "@copilotkit/react-core";
import { useSettings } from "../store/settings";
import { useWatchlist } from "../store/watchlist";
import { useDocs } from "../store/docs";
import { useChat } from "../store/chat";

export const CopilotReadables: React.FC = () => {
  const { theme, riseRedFallGreen, devMode } = useSettings();
  const { symbols, active } = useWatchlist();
  const docs = useDocs((s) => s.docs);
  const messages = useChat((s) => s.messages);

  const lastReport = React.useMemo(() => {
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    if (!last || last.role !== "assistant") return "";
    return last.text.slice(0, 500);
  }, [messages]);

  useCopilotReadable({ description: "当前界面主题与配置", value: { theme, riseRedFallGreen, devMode } });
  useCopilotReadable({ description: "用户自选股列表与当前激活标的", value: { symbols, active } });
  useCopilotReadable({ description: "已上传的财报PDF文件名列表", value: docs.map((d) => d.name) });
  useCopilotReadable({ description: "最近一条AI分析报告摘要（前500字）", value: lastReport });

  return null;
};
