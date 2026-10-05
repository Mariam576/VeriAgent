// Every call to the VeriAgent API goes through this file.
//
// Sign-in, in short (the server side is app/api/auth.py):
//   * The ACCESS TOKEN proves who you are for about half an hour. It is kept in a
//     variable here - in memory only. It is never written to localStorage, where
//     any script that got onto the page could read it.
//   * The REFRESH TOKEN lives in a cookie that page scripts cannot read at all
//     (httpOnly). When the access token has run out, or the page was reloaded,
//     POST /auth/refresh exchanges that cookie for a new access token.
//   * In development mode (AUTH_MODE=dev) there are no tokens: a name is sent in
//     the X-User-Id header instead.

import { readEvents } from "./sse";
import type {
  ActionConfig,
  ApprovalDecision,
  AuthConfig,
  Comparison,
  DocumentInfo,
  ForensicsOut,
  Health,
  InvestigationDetail,
  InvestigationSummary,
  MemoryNote,
  MemoryOverview,
  Preference,
  ProfileField,
  LineItem,
  Report,
  StreamEvent,
  TokenResponse,
  User,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string = "error",
  ) {
    super(message);
  }
}

let accessToken: string | null = null;
let devUser: string | null = null;
let onSignedOut: (() => void) | null = null;
let refreshing: Promise<boolean> | null = null;

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export function setDevUser(name: string | null): void {
  devUser = name;
}

/** Called when the server no longer accepts our sign-in, so the page can show the sign-in form. */
export function onSessionEnded(handler: (() => void) | null): void {
  onSignedOut = handler;
}

function authHeaders(): Record<string, string> {
  if (accessToken) return { Authorization: `Bearer ${accessToken}` };
  if (devUser) return { "X-User-Id": devUser };
  return {};
}

async function errorFrom(response: Response): Promise<ApiError> {
  let message = `The server answered with status ${response.status}.`;
  let code = "error";
  try {
    const body = await response.json();
    if (body?.error?.message) {
      message = String(body.error.message);
      code = String(body.error.code ?? code);
    } else if (Array.isArray(body?.detail)) {
      // FastAPI's own validation errors: [{loc: [...], msg: "..."}]
      message = body.detail
        .map((item: { loc?: unknown[]; msg?: string }) => `${(item.loc ?? []).slice(1).join(".")}: ${item.msg}`)
        .join("; ");
      code = "validation_failed";
    } else if (typeof body?.detail === "string") {
      message = body.detail;
    }
  } catch {
    // not JSON: keep the generic message
  }
  return new ApiError(message, response.status, code);
}

/** Exchange the refresh cookie for a new access token. Several callers share one attempt. */
export function refreshSession(): Promise<boolean> {
  refreshing ??= (async () => {
    try {
      const response = await fetch("/auth/refresh", { method: "POST", credentials: "same-origin" });
      if (!response.ok) return false;
      const body = (await response.json()) as TokenResponse;
      accessToken = body.access_token;
      return true;
    } catch {
      return false;
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
}

async function send(path: string, init: RequestInit = {}, retry = true): Promise<Response> {
  const response = await fetch(path, {
    ...init,
    credentials: "same-origin",
    headers: { ...authHeaders(), ...(init.headers as Record<string, string> | undefined) },
  });
  // An access token that ran out: get a new one once, then repeat the request.
  if (response.status === 401 && retry && accessToken && !path.startsWith("/auth/")) {
    if (await refreshSession()) return send(path, init, false);
    accessToken = null;
    onSignedOut?.();
  }
  if (!response.ok) throw await errorFrom(response);
  return response;
}

async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await send(path, init);
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

function body(payload: unknown): RequestInit {
  return { headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
}

export interface DocumentProfile {
  document_id: string;
  filename: string;
  document_type: string;
  profile: {
    built_by: string;
    number: ProfileField | null;
    contract_reference: ProfileField | null;
    purchase_order_reference: ProfileField | null;
    parties: ProfileField[];
    date: ProfileField | null;
    term_start: ProfileField | null;
    term_end: ProfileField | null;
    term_months: ProfileField | null;
    total: ProfileField | null;
    subtotal: ProfileField | null;
    tax: ProfileField | null;
    payment_terms_days: ProfileField | null;
    line_items: LineItem[];
  } | null;
  note: string | null;
}

export const api = {
  // ------------------------------------------------------------------ sign-in
  health: () => json<Health>("/health"),
  authConfig: () => json<AuthConfig>("/auth/config"),
  register: (email: string, password: string, name: string, signupCode: string) =>
    json<TokenResponse>("/auth/register", {
      method: "POST",
      ...body({ email, password, name: name || null, signup_code: signupCode || null }),
    }),
  login: (email: string, password: string) =>
    json<TokenResponse>("/auth/login", { method: "POST", ...body({ email, password }) }),
  google: (credential: string, signupCode: string) =>
    json<TokenResponse>("/auth/google", { method: "POST", ...body({ credential, signup_code: signupCode || null }) }),
  restore: async (): Promise<TokenResponse | null> => {
    const response = await fetch("/auth/refresh", { method: "POST", credentials: "same-origin" });
    return response.ok ? ((await response.json()) as TokenResponse) : null;
  },
  me: () => json<User>("/auth/me"),
  logout: () => json<void>("/auth/logout", { method: "POST" }),
  changePassword: (currentPassword: string | null, newPassword: string) =>
    json<TokenResponse>("/auth/password", {
      method: "POST",
      ...body({ current_password: currentPassword, new_password: newPassword }),
    }),
  deleteAccount: () => json<void>("/auth/me", { method: "DELETE", ...body({ confirm: "DELETE" }) }),

  // ---------------------------------------------------------------- documents
  documents: async () => (await json<{ documents: DocumentInfo[] }>("/documents")).documents,
  document: (id: string) => json<DocumentInfo>(`/documents/${id}`),
  upload: (file: File, documentType: string) => {
    const form = new FormData();
    form.append("file", file);
    if (documentType) form.append("document_type", documentType);
    // background=true: the server answers at once and reads the file afterwards.
    return json<DocumentInfo>("/documents/upload?background=true", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => json<void>(`/documents/${id}`, { method: "DELETE" }),
  profile: (id: string) => json<DocumentProfile>(`/documents/${id}/profile`),
  forensics: (id: string) => json<ForensicsOut>(`/documents/${id}/forensics`),
  compare: (documentIds: string[]) =>
    json<Comparison>("/comparisons", { method: "POST", ...body({ document_ids: documentIds }) }),

  // ----------------------------------------------------------- investigations
  investigations: async () =>
    (await json<{ investigations: InvestigationSummary[] }>("/investigations")).investigations,
  investigation: (id: string) => json<InvestigationDetail>(`/investigations/${id}`),
  decide: (id: string, decision: ApprovalDecision) =>
    json<Report>(`/investigations/${id}/approve`, { method: "POST", ...body(decision) }),
  resume: (id: string) => json<Report>(`/investigations/${id}/resume`, { method: "POST" }),
  actionConfig: () => json<ActionConfig>("/actions/config"),

  /** Ask a question and receive progress while the agent works. */
  chat: async (
    message: string,
    conversationId: string | null,
    onEvent: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const response = await send("/chat/stream", {
      method: "POST",
      signal,
      ...body({ message, conversation_id: conversationId }),
    });
    await readEvents(response, (raw) => {
      try {
        onEvent({ event: raw.event, data: JSON.parse(raw.data) } as StreamEvent);
      } catch {
        // an event that is not valid JSON is skipped
      }
    });
  },

  // ------------------------------------------------------------------- memory
  memory: () => json<MemoryOverview>("/memory"),
  setPreference: (key: string, value: string) =>
    json<Preference>(`/memory/preferences/${encodeURIComponent(key)}`, { method: "PUT", ...body({ value }) }),
  addNote: (text: string) => json<MemoryNote>("/memory/notes", { method: "POST", ...body({ text }) }),
  deleteNote: (id: string) => json<void>(`/memory/notes/${id}`, { method: "DELETE" }),
  forgetEverything: () => json<Record<string, number>>("/memory", { method: "DELETE" }),
};
