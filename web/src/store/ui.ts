import { create } from "zustand";

export type CtxTab = "quote" | "news" | "filings" | "copilot";

interface UIState {
  ctxTab: CtxTab;
  setCtxTab: (t: CtxTab) => void;
  suggestion: string;
  setSuggestion: (s: string) => void;
  clearSuggestion: () => void;
}

export const useUI = create<UIState>((set) => ({
  ctxTab: "quote",
  setCtxTab: (t) => set({ ctxTab: t }),
  suggestion: "",
  setSuggestion: (s) => set({ suggestion: s }),
  clearSuggestion: () => set({ suggestion: "" }),
}));
