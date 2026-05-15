import { create } from "zustand";
import { fetchQuotes } from "../api/client";
import type { Quote } from "../types/sse";

const STORAGE_KEY = "stockai.watchlist.v1";
const DEFAULT_SYMBOLS = ["NVDA", "AAPL", "MSFT", "TSLA", "600519.SS", "00700.HK"];

function loadStored(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SYMBOLS;
    const arr = JSON.parse(raw);
    return Array.isArray(arr) && arr.length ? arr : DEFAULT_SYMBOLS;
  } catch {
    return DEFAULT_SYMBOLS;
  }
}

function persist(symbols: string[]): void {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(symbols)); } catch { /* noop */ }
}

interface WatchlistState {
  symbols:    string[];
  quotes:     Record<string, Quote>;
  active:     string;
  loading:    boolean;
  setActive:  (sym: string) => void;
  add:        (sym: string) => void;
  remove:     (sym: string) => void;
  refresh:    () => Promise<void>;
}

export const useWatchlist = create<WatchlistState>((set, get) => ({
  symbols: loadStored(),
  quotes:  {},
  active:  loadStored()[0],
  loading: false,

  setActive: (sym) => set({ active: sym }),

  add: (sym) => {
    const norm = sym.trim().toUpperCase();
    if (!norm) return;
    set((s) => {
      if (s.symbols.includes(norm)) return s;
      const next = [...s.symbols, norm];
      persist(next);
      return { symbols: next };
    });
    get().refresh();
  },

  remove: (sym) => set((s) => {
    const next = s.symbols.filter((x) => x !== sym);
    persist(next);
    const active = s.active === sym ? (next[0] ?? "") : s.active;
    return { symbols: next, active };
  }),

  refresh: async () => {
    const symbols = get().symbols;
    if (!symbols.length) return;
    set({ loading: true });
    try {
      const quotes = await fetchQuotes(symbols);
      const map: Record<string, Quote> = {};
      for (const q of quotes) map[q.symbol] = q;
      set({ quotes: map });
    } catch {
      /* swallow — stale quotes are still useful */
    } finally {
      set({ loading: false });
    }
  },
}));
