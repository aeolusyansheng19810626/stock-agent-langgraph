import { create } from "zustand";
import { deleteDoc as apiDelete, fetchDocs, uploadDoc } from "../api/client";
import type { DocItem } from "../types/sse";

interface DocsState {
  docs:    DocItem[];
  loading: boolean;
  error:   string | null;
  refresh: () => Promise<void>;
  add:     (file: File) => Promise<void>;
  remove:  (name: string) => Promise<void>;
}

export const useDocs = create<DocsState>((set) => ({
  docs:    [],
  loading: false,
  error:   null,

  refresh: async () => {
    set({ loading: true, error: null });
    try {
      const docs = await fetchDocs();
      set({ docs });
    } catch (e) {
      set({ error: (e as Error).message });
    } finally {
      set({ loading: false });
    }
  },

  add: async (file) => {
    set({ loading: true, error: null });
    try {
      await uploadDoc(file);
      const docs = await fetchDocs();
      set({ docs });
    } catch (e) {
      set({ error: (e as Error).message });
    } finally {
      set({ loading: false });
    }
  },

  remove: async (name) => {
    try {
      await apiDelete(name);
      const docs = await fetchDocs();
      set({ docs });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },
}));
