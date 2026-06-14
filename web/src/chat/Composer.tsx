import React from "react";
import { Icon } from "../icons/Icon";
import { streamAnalyze } from "../api/sse";
import { useChat } from "../store/chat";
import { useDocs } from "../store/docs";
import { useSettings } from "../store/settings";
import { useWatchlist } from "../store/watchlist";
import { CopilotTextarea } from "@copilotkit/react-textarea";
import { useUI } from "../store/ui";

async function fileToB64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const result = r.result as string;
      resolve(result.split(",")[1] ?? "");
    };
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

export const Composer: React.FC = () => {
  const [text, setText] = React.useState("");
  const [imageB64, setImageB64] = React.useState<string | null>(null);
  const [imagePreview, setImagePreview] = React.useState<string | null>(null);
  const [deepRead, setDeepRead] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const ch          = useChat();
  const messages    = useChat((s) => s.messages);
  const activeSym   = useWatchlist((s) => s.active);
  const activeDoc   = useDocs((s) => s.docs[0]);
  const setStatus   = useChat((s) => s.setStatus);
  const settings    = useSettings();

  const pending = !!ch.activeAssistant;

  const suggestion = useUI((s) => s.suggestion);
  const clearSuggestion = useUI((s) => s.clearSuggestion);
  React.useEffect(() => {
    if (suggestion) {
      setText(suggestion);
      clearSuggestion();
    }
  }, [suggestion, clearSuggestion]);

  const submit = async () => {
    const userText = text.trim();
    if (!userText && !imageB64) return;
    if (pending) return;

    const finalText = userText || (imageB64 ? "请分析这张图片" : "");
    ch.pushUser(finalText, imageB64 ?? undefined);
    setText(""); setImageB64(null); setImagePreview(null);

    const id = ch.startAssistant();
    setStatus("正在分析…");

    const chat_history = messages.map((m) => ({
      role:    m.role,
      content: m.role === "user" ? m.text : m.text,
    }));

    try {
      await streamAnalyze({
        user_input:        finalText,
        chat_history,
        dev_mode:          settings.devMode,
        gemini_exhausted:  settings.geminiExhausted,
        image_b64:         imageB64 ?? undefined,
      }, {
        onEvent: (e) => {
          switch (e.event) {
            case "node.start":
              setStatus(e.data.label);
              break;
            case "node.complete":
              if (e.data.payload && Object.keys(e.data.payload).length) {
                ch.putPayload(id, e.data.node, e.data.payload as Record<string, unknown>);
              }
              break;
            case "tool.call":
              ch.appendStep(id, e.data);
              break;
            case "report.token":
              ch.appendToken(id, e.data.delta);
              break;
            case "report.section":
              ch.appendToken(id, "\n\n" + e.data.markdown);
              break;
            case "chart":
              ch.pushChart(id, e.data.path);
              break;
            case "error":
              ch.pushError(id, e.data);
              break;
            case "done":
              if (e.data.gemini_exhausted) settings.setExhausted(true);
              ch.finishAssistant(id, {
                finalModel:  e.data.final_model,
                emailStatus: e.data.email_status,
                elapsed:     e.data.elapsed,
                text:        e.data.final_report || undefined,
                tickers:     e.data.tickers ?? [],
              });
              setStatus("");
              break;
          }
        },
        onError: (err) => {
          ch.pushError(id, { node: "stream", tool: "", message: String(err) });
          ch.finishAssistant(id, { finalModel: "error" });
          setStatus("");
        },
      });
    } catch (err) {
      ch.pushError(id, { node: "stream", tool: "", message: String(err) });
      ch.finishAssistant(id, { finalModel: "error" });
      setStatus("");
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const onPickImage = async (file: File | null) => {
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) {
      alert("图片超过 2MB，已忽略");
      return;
    }
    const b64 = await fileToB64(file);
    setImageB64(b64);
    setImagePreview(URL.createObjectURL(file));
  };

  return (
    <div className="sx-composer-wrap">
      <div className="sx-composer">
        <div className="sx-composer-chips">
          {activeSym && (
            <span className="sx-chip accent">
              <span className="mono">{activeSym}</span>
            </span>
          )}
          {activeDoc && (
            <span className="sx-chip" title={activeDoc.name}>
              <Icon name="attach" size={11} />
              {activeDoc.name.length > 22 ? activeDoc.name.slice(0, 22) + "…" : activeDoc.name}
            </span>
          )}
          {imagePreview && (
            <span className="sx-chip">
              <img src={imagePreview} alt="附图" style={{ height: 18, borderRadius: 2 }} />
              <span className="x" onClick={() => { setImageB64(null); setImagePreview(null); }}>×</span>
            </span>
          )}
        </div>

        <CopilotTextarea
          placeholder="请输入你的问题，例如：分析一下英伟达的股票…"
          value={text}
          onValueChange={(v: string) => setText(v)}
          onKeyDown={onKey}
          autosuggestionsConfig={{
            textareaPurpose: "股票分析提问，例如分析特定股票的基本面、技术面、风险等",
            chatApiConfigs: {
              suggestionsApiConfig: { maxTokens: 20, stop: ["。", "？", "！", "\n"] },
            },
          }}
        />

        <div className="sx-composer-foot">
          <div className="sx-comp-tools">
            <button className="sx-comp-btn" onClick={() => fileInputRef.current?.click()}>
              <Icon name="attach" size={13} /> 附件
            </button>
            <button className={`sx-comp-btn ${deepRead ? "active" : ""}`} onClick={() => setDeepRead((v) => !v)}>
              <Icon name="ai" size={13} /> 深度分析
            </button>
          </div>
          <button className="sx-send" onClick={submit} disabled={pending || (!text.trim() && !imageB64)}>
            <Icon name="send" size={12} /> 发送 <span className="sx-kbd">⏎</span>
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => onPickImage(e.target.files?.[0] ?? null)}
        />
      </div>
    </div>
  );
};
