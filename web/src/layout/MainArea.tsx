import React from "react";
import { ChatStream } from "../chat/ChatStream";
import { Composer } from "../chat/Composer";
import { SuggestionGrid } from "../chat/SuggestionGrid";
import { useChat } from "../store/chat";
import { useDocs } from "../store/docs";
import { useWatchlist } from "../store/watchlist";
import { useUI } from "../store/ui";

export const MainArea: React.FC = () => {
  const status     = useChat((s) => s.status);
  const tokenCount = useChat((s) => s.tokenCount);
  const messages   = useChat((s) => s.messages);
  const activeSym  = useWatchlist((s) => s.active);
  const docs       = useDocs((s) => s.docs);
  const isStreaming = useChat((s) => !!s.activeAssistant);

  const onPick = (q: string) => {
    useUI.getState().setSuggestion(q);
  };

  const ctxLabel = activeSym
    ? `上下文 · ${activeSym}${docs.length ? ` · ${docs[0].name.replace(/\.pdf$/i, "")}` : ""}`
    : "上下文 · —";

  return (
    <main className="sx-main">
      <div className="sx-main-head">
        <div className="sx-main-title">
          <h1>AI 分析师</h1>
          <span className="sx-ctx">{ctxLabel}</span>
        </div>
        <div className="sx-main-meta">
          {isStreaming ? (
            <span className="sx-meta-chip live">
              <span className="sx-pulse" /> {status || "流式输出中"}
            </span>
          ) : (
            <span className="sx-meta-chip">就绪</span>
          )}
          <span>会话 · #<span className="num">{messages.length}</span></span>
          <span>Token · <span className="num">{tokenCount.toLocaleString()}</span> / 64K</span>
        </div>
      </div>

      <ChatStream />

      <div style={{ padding: "0 32px" }}>
        {!messages.length && <SuggestionGrid onPick={onPick} />}
      </div>

      <Composer />
    </main>
  );
};
