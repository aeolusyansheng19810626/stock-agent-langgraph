import type { DocItem, Quote } from "../types/sse";

const BASE = "";  // same-origin in prod; vite proxies /api → :8000 in dev

async function jsonOk<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail ?? msg; } catch { /* noop */ }
    throw new Error(msg);
  }
  return res.json();
}

export async function fetchQuotes(symbols: string[]): Promise<Quote[]> {
  if (!symbols.length) return [];
  const res = await fetch(`${BASE}/api/quote?symbols=${encodeURIComponent(symbols.join(","))}`);
  const body = await jsonOk<{ quotes: Quote[] }>(res);
  return body.quotes;
}

export async function fetchDocs(): Promise<DocItem[]> {
  const res = await fetch(`${BASE}/api/docs`);
  const body = await jsonOk<{ docs: DocItem[] }>(res);
  return body.docs;
}

export async function uploadDoc(file: File): Promise<DocItem> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/api/docs`, { method: "POST", body: fd });
  const body = await jsonOk<{ doc: DocItem }>(res);
  return body.doc;
}

export async function deleteDoc(name: string): Promise<void> {
  const res = await fetch(`${BASE}/api/docs/${encodeURIComponent(name)}`, { method: "DELETE" });
  await jsonOk(res);
}

export async function fetchHistory(): Promise<unknown[]> {
  const res = await fetch(`${BASE}/api/history`);
  const body = await jsonOk<{ records: unknown[] }>(res);
  return body.records;
}

export async function clearHistory(): Promise<void> {
  const res = await fetch(`${BASE}/api/history`, { method: "DELETE" });
  await jsonOk(res);
}

export async function sendEmail(to: string, subject: string, body: string): Promise<{ ok: boolean; message: string }> {
  const res = await fetch(`${BASE}/api/email`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ to, subject, body }),
  });
  return jsonOk(res);
}
