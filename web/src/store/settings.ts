import { create } from "zustand";

export type Theme = "amber" | "cyan" | "ink";

type ToggleKey =
  | "devMode" | "autoEmail" | "citeSources" | "noBuySell"
  | "livel2"  | "premarket" | "cninfoSync"  | "riseRedFallGreen";

export interface SettingsState {
  theme:            Theme;
  devMode:          boolean;
  geminiExhausted:  boolean;
  autoEmail:        boolean;
  citeSources:      boolean;
  noBuySell:        boolean;
  livel2:           boolean;
  premarket:        boolean;
  cninfoSync:       boolean;
  riseRedFallGreen: boolean;       // A 股惯例：红涨绿跌
  emailRecipient:   string;
  settingsOpen:     boolean;
  setTheme:         (t: Theme) => void;
  toggle:           (key: ToggleKey) => void;
  setOpen:          (v: boolean) => void;
  setEmail:         (v: string) => void;
  setExhausted:     (v: boolean) => void;
}

export const useSettings = create<SettingsState>((set) => ({
  theme:            "amber",
  devMode:          false,
  geminiExhausted:  false,
  autoEmail:        true,
  citeSources:      true,
  noBuySell:        false,
  livel2:           true,
  premarket:        true,
  cninfoSync:       true,
  riseRedFallGreen: false,
  emailRecipient:   "",
  settingsOpen:     false,
  setTheme:         (t) => set({ theme: t }),
  toggle:           (k) => set((s) => ({ [k]: !s[k] }) as Partial<SettingsState>),
  setOpen:          (v) => set({ settingsOpen: v }),
  setEmail:         (v) => set({ emailRecipient: v }),
  setExhausted:     (v) => set({ geminiExhausted: v }),
}));
