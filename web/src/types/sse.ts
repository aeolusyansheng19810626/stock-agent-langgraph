/* TS mirror of server/events.py — keep in sync. */

export type ChatRole = "user" | "assistant" | "system";

export interface ChatHistoryEntry {
  role: ChatRole;
  content: string;
}

export interface AnalyzeRequest {
  user_input: string;
  chat_history?: ChatHistoryEntry[];
  dev_mode?: boolean;
  gemini_exhausted?: boolean;
  image_b64?: string | null;
}

export type SSEEventName =
  | "node.start"
  | "node.complete"
  | "tool.call"
  | "report.token"
  | "report.section"
  | "error"
  | "chart"
  | "done";

export interface NodeStartData     { node: string; label: string; }
export interface NodeCompleteData  { node: string; payload: Record<string, unknown> | null; }
export interface ToolCallData      { step: number; tool_name: string; tool_args: Record<string, unknown>; retries: number; }
export interface ReportTokenData   { delta: string; }
export interface ReportSectionData { type: string; markdown: string; }
export interface ErrorData         { node: string; tool: string; message: string; }
export interface ChartData         { path: string; }
export interface DoneData {
  final_model:      string;
  final_report:     string;
  email_status:     string;
  gemini_exhausted: boolean;
  elapsed:          number;
}

export type SSEEvent =
  | { event: "node.start";     data: NodeStartData     }
  | { event: "node.complete";  data: NodeCompleteData  }
  | { event: "tool.call";      data: ToolCallData      }
  | { event: "report.token";   data: ReportTokenData   }
  | { event: "report.section"; data: ReportSectionData }
  | { event: "error";          data: ErrorData         }
  | { event: "chart";          data: ChartData         }
  | { event: "done";           data: DoneData          };

export interface Quote {
  symbol:    string;
  name?:     string;
  exchange?: string;
  price?:    number;
  change?:   number;
  pct?:      number;
  open?:     number;
  prevClose?:number;
  high?:     number;
  low?:      number;
  volume?:   number;
  high52w?:  number;
  low52w?:   number;
  marketCap?:number;
  pe?:       number;
  pb?:       number;
  divYield?: number;
  error?:    string;
}

export interface DocItem {
  id:         string;
  name:       string;
  size:       string;
  chunks:     number;
  uploadedAt: string;
  kind:       string;
}
