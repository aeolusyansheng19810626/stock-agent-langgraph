import { create } from "zustand";
import type { ToolCallData } from "../types/sse";

export interface AssistantMessage {
  id:         string;
  role:       "assistant";
  text:       string;                              // streamed report markdown
  steps:      ToolCallData[];                      // tool-call cards
  payloads:   Record<string, Record<string, unknown>>;  // node → structured data
  charts:     string[];                            // chart paths
  errors:     { node: string; tool: string; message: string }[];
  finalModel: string;
  emailStatus:string;
  elapsed?:   number;
  pending:    boolean;                             // true while streaming
}

export interface UserMessage {
  id:    string;
  role:  "user";
  text:  string;
  imageB64?: string;
}

export type ChatMessage = UserMessage | AssistantMessage;

export interface ChatState {
  messages:        ChatMessage[];
  activeAssistant: string | null;     // id of the assistant message currently streaming
  status:          string;            // last node label, displayed in header
  tokenCount:      number;
  reset:           () => void;
  pushUser:        (text: string, imageB64?: string) => string;
  startAssistant:  () => string;
  appendToken:     (id: string, delta: string) => void;
  appendStep:      (id: string, step: ToolCallData) => void;
  setStatus:       (label: string) => void;
  putPayload:      (id: string, node: string, payload: Record<string, unknown>) => void;
  pushChart:       (id: string, path: string) => void;
  pushError:       (id: string, err: { node: string; tool: string; message: string }) => void;
  finishAssistant: (id: string, args: Partial<AssistantMessage>) => void;
}

let _idSeq = 0;
const newId = () => `m-${Date.now()}-${++_idSeq}`;

export const useChat = create<ChatState>((set) => ({
  messages:        [],
  activeAssistant: null,
  status:          "",
  tokenCount:      0,

  reset: () => set({ messages: [], activeAssistant: null, status: "", tokenCount: 0 }),

  pushUser: (text, imageB64) => {
    const id = newId();
    set((s) => ({ messages: [...s.messages, { id, role: "user", text, imageB64 }] }));
    return id;
  },

  startAssistant: () => {
    const id = newId();
    set((s) => ({
      messages: [...s.messages, {
        id, role: "assistant", text: "", steps: [], payloads: {},
        charts: [], errors: [], finalModel: "", emailStatus: "", pending: true,
      }],
      activeAssistant: id,
    }));
    return id;
  },

  appendToken: (id, delta) => set((s) => ({
    messages: s.messages.map((m) =>
      m.id === id && m.role === "assistant" ? { ...m, text: m.text + delta } : m,
    ),
    tokenCount: s.tokenCount + delta.length,
  })),

  appendStep: (id, step) => set((s) => ({
    messages: s.messages.map((m) =>
      m.id === id && m.role === "assistant" ? { ...m, steps: [...m.steps, step] } : m,
    ),
  })),

  setStatus: (label) => set({ status: label }),

  putPayload: (id, node, payload) => set((s) => ({
    messages: s.messages.map((m) =>
      m.id === id && m.role === "assistant"
        ? { ...m, payloads: { ...m.payloads, [node]: payload } }
        : m,
    ),
  })),

  pushChart: (id, path) => set((s) => ({
    messages: s.messages.map((m) =>
      m.id === id && m.role === "assistant" ? { ...m, charts: [...m.charts, path] } : m,
    ),
  })),

  pushError: (id, err) => set((s) => ({
    messages: s.messages.map((m) =>
      m.id === id && m.role === "assistant" ? { ...m, errors: [...m.errors, err] } : m,
    ),
  })),

  finishAssistant: (id, args) => set((s) => ({
    messages: s.messages.map((m) =>
      m.id === id && m.role === "assistant" ? { ...m, ...args, pending: false } : m,
    ),
    activeAssistant: null,
  })),
}));
