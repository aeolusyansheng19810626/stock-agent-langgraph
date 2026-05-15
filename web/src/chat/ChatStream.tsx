import React from "react";
import { AssistantMessage } from "./AssistantMessage";
import { UserMessage } from "./UserMessage";
import { useChat } from "../store/chat";

export const ChatStream: React.FC = () => {
  const messages = useChat((s) => s.messages);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [messages]);

  if (!messages.length) {
    return (
      <div className="sx-chat" ref={ref}>
        <div style={{ margin: "auto", textAlign: "center", color: "var(--fg-dimmer)" }}>
          <div style={{ fontFamily: "var(--font-serif)", fontSize: 24, color: "var(--fg-dim)", marginBottom: 6 }}>
            你好，我是 AI 股票分析师
          </div>
          <div style={{ fontSize: 12 }}>点击下方卡片或直接输入问题开始分析</div>
        </div>
      </div>
    );
  }

  return (
    <div className="sx-chat" ref={ref}>
      {messages.map((m) =>
        m.role === "user"
          ? <UserMessage key={m.id} msg={m} />
          : <AssistantMessage key={m.id} msg={m} />,
      )}
    </div>
  );
};
