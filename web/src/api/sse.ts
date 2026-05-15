import { fetchEventSource } from "@microsoft/fetch-event-source";
import type { AnalyzeRequest, SSEEvent } from "../types/sse";

export interface AnalyzeHandlers {
  onEvent: (e: SSEEvent) => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
  signal?: AbortSignal;
}

/** POST /api/analyze and dispatch each SSE frame to handlers.onEvent. */
export async function streamAnalyze(req: AnalyzeRequest, handlers: AnalyzeHandlers): Promise<void> {
  await fetchEventSource("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body:    JSON.stringify(req),
    signal:  handlers.signal,
    openWhenHidden: true,
    onmessage(msg) {
      if (!msg.event) return;
      let data: unknown = {};
      try { data = JSON.parse(msg.data); } catch { /* noop */ }
      handlers.onEvent({ event: msg.event as SSEEvent["event"], data: data as never });
    },
    onerror(err) {
      handlers.onError?.(err);
      throw err;  // stop reconnect attempts
    },
    onclose() { handlers.onClose?.(); },
  });
}
