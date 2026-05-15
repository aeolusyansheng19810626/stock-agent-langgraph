import React from "react";
import type { UserMessage as UserMsg } from "../store/chat";

function nowTime(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export const UserMessage: React.FC<{ msg: UserMsg }> = ({ msg }) => (
  <div className="sx-msg user">
    <div className="sx-msg-head">
      <span className="sx-mono">{nowTime()}</span>
      <span className="sx-role">陈先生</span>
    </div>
    <div className="sx-msg-body">
      {msg.imageB64 && (
        <img src={`data:image/png;base64,${msg.imageB64}`} alt="附图" style={{ maxWidth: 360, marginBottom: 8 }} />
      )}
      <div style={{ whiteSpace: "pre-wrap" }}>{msg.text}</div>
    </div>
  </div>
);
