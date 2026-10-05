// The main screen (specification section 50):
//
//   left     your documents, upload, investigation history
//   middle   the chat, with live progress while the agent works
//   right    the result: risk, findings as evidence cards, the human decision
//
// This component owns the data all three panels share and passes it down.

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ActionConfig, AuthConfig, DocumentInfo, Health, InvestigationSummary, Report, StreamEvent, User } from "../api/types";
import { errorText } from "../util/format";
import { AccountMenu } from "./AccountMenu";
import { Chat, type ChatMessage } from "./Chat";
import { CompareView } from "./CompareView";
import { DocumentDetails } from "./DocumentDetails";
import { DocumentsPanel } from "./DocumentsPanel";
import { HistoryPanel } from "./HistoryPanel";
import { MemoryView } from "./MemoryView";
import { ResultPanel } from "./ResultPanel";

interface Props {
  user: User;
  config: AuthConfig;
  onUserChanged: (user: User) => void;
  onSignOut: () => void;
  onAccountDeleted: () => void;
}

const POLL_MS = 1500;
const MAX_POLL_MS = 30_000;

export function Workspace({ user, config, onUserChanged, onSignOut, onAccountDeleted }: Props) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [history, setHistory] = useState<InvestigationSummary[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [actionConfig, setActionConfig] = useState<ActionConfig | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [progress, setProgress] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const [selected, setSelected] = useState<InvestigationSummary | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [comparing, setComparing] = useState(false);
  const [detailsFor, setDetailsFor] = useState<DocumentInfo | null>(null);
  const [showMemory, setShowMemory] = useState(false);
  const [notice, setNotice] = useState("");
  const stop = useRef<AbortController | null>(null);

  /** Returns false when the list could not be loaded. `quiet`: a background check that shows no message. */
  const loadDocuments = useCallback(async (quiet = false): Promise<boolean> => {
    try {
      setDocuments(await api.documents());
      return true;
    } catch (error) {
      if (!quiet) setNotice(errorText(error));
      return false;
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      setHistory(await api.investigations());
    } catch (error) {
      setNotice(errorText(error));
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
    void loadHistory();
    api.health().then(setHealth).catch(() => setHealth(null));
    api.actionConfig().then(setActionConfig).catch(() => setActionConfig(null));
  }, [loadDocuments, loadHistory]);

  // Documents uploaded in the background: ask again until none is "processing". A check that
  // fails (the server is restarting) does not end the asking; it only becomes less frequent.
  const processing = documents.some((document) => document.status === "processing");
  useEffect(() => {
    if (!processing) return;
    let stopped = false;
    let delay = POLL_MS;
    let timer = 0;
    const check = async () => {
      const ok = await loadDocuments(true);
      if (stopped) return;
      delay = ok ? POLL_MS : Math.min(delay * 2, MAX_POLL_MS);
      timer = window.setTimeout(() => void check(), delay);
    };
    timer = window.setTimeout(() => void check(), delay);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [processing, loadDocuments]);

  // A document that was deleted, or is no longer ready, cannot stay ticked for comparison.
  useEffect(() => {
    const ready = new Set(documents.filter((document) => document.status === "ready").map((document) => document.id));
    setCompareIds((current) => (current.every((id) => ready.has(id)) ? current : current.filter((id) => ready.has(id))));
  }, [documents]);

  const ask = async (question: string) => {
    setBusy(true);
    setNotice("");
    setProgress([]);
    setMessages((current) => [...current, { role: "user", text: question }]);
    const controller = new AbortController();
    stop.current = controller;
    let answered = false;
    const onEvent = (event: StreamEvent) => {
      if (event.event === "started") {
        setConversationId(event.data.conversation_id);
      } else if (event.event === "progress") {
        setProgress((lines) => (lines[lines.length - 1] === event.data.message ? lines : [...lines, event.data.message]));
      } else if (event.event === "report") {
        setReport(event.data);
        setSelected(null);
      } else if (event.event === "answer") {
        answered = true;
        setMessages((current) => [
          ...current,
          {
            role: "assistant", text: event.data.answer, risk: event.data.risk_level, status: event.data.status,
            pending: event.data.approval_status === "pending", investigationId: event.data.investigation_id,
          },
        ]);
      } else if (event.event === "error") {
        answered = true;
        setMessages((current) => [...current, { role: "error", text: event.data.message }]);
      }
    };
    try {
      await api.chat(question, conversationId, onEvent, controller.signal);
      if (!answered) setMessages((current) => [...current, { role: "error", text: "The connection ended before an answer arrived. Open the investigation in the history to continue it." }]);
    } catch (error) {
      const stopped = controller.signal.aborted;
      setMessages((current) => [...current, { role: "error", text: stopped ? "Stopped. The investigation can be continued from the history." : errorText(error) }]);
    } finally {
      stop.current = null;
      setBusy(false);
      setProgress([]);
      void loadHistory();
    }
  };

  const openInvestigation = async (item: InvestigationSummary) => {
    setNotice("");
    try {
      const detail = await api.investigation(item.id);
      setSelected(detail);
      setReport(detail.report);
    } catch (error) {
      setNotice(errorText(error));
    }
  };

  const newConversation = () => {
    setMessages([]);
    setConversationId(null);
    setReport(null);
    setSelected(null);
  };

  const reportChanged = (updated: Report) => {
    setReport(updated);
    setSelected(null);
    // The answer in the chat was "waiting for your decision": it no longer is.
    const waiting = updated.human_review.status === "pending";
    setMessages((current) => current.map((message) => (message.investigationId === updated.investigation_id ? { ...message, pending: waiting } : message)));
    void loadHistory();
  };

  const toggleCompare = (id: string) =>
    setCompareIds((current) => (current.includes(id) ? current.filter((other) => other !== id) : [...current, id].slice(-6)));

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            ✓
          </span>
          <strong>VeriAgent</strong>
          {health && (
            <span className="muted small" title="The language model that reads the documents">
              {health.llm_provider === "mock" ? "offline demo model" : health.llm_model}
            </span>
          )}
        </div>
        <div className="topbar-actions">
          <button className="ghost" onClick={() => setShowMemory(true)}>
            Memory
          </button>
          <AccountMenu user={user} devMode={config.auth_mode === "dev"} minPassword={config.min_password_chars} onUserChanged={onUserChanged} onSignOut={onSignOut} onAccountDeleted={onAccountDeleted} />
        </div>
      </header>

      {notice && (
        <div className="notice" role="alert">
          {notice}
          <button className="ghost" onClick={() => setNotice("")} aria-label="Dismiss">
            ✕
          </button>
        </div>
      )}

      <main className="columns">
        <aside className="panel left">
          <DocumentsPanel
            documents={documents}
            compareIds={compareIds}
            onToggleCompare={toggleCompare}
            onCompare={() => setComparing(true)}
            onChanged={() => void loadDocuments()}
            onDetails={setDetailsFor}
            onError={setNotice}
          />
          <HistoryPanel items={history} activeId={report?.investigation_id ?? selected?.id ?? null} onOpen={(item) => void openInvestigation(item)} />
        </aside>

        <section className="panel middle">
          <Chat
            messages={messages}
            busy={busy}
            progress={progress}
            hasDocuments={documents.some((document) => document.status === "ready")}
            onAsk={(question) => void ask(question)}
            onStop={() => stop.current?.abort()}
            onNew={newConversation}
          />
        </section>

        <aside className="panel right">
          {/* key: nothing typed or shown for one investigation (an error, a half-made decision) stays for the next */}
          <ResultPanel
            key={report?.investigation_id ?? selected?.id ?? "none"}
            report={report}
            summary={selected}
            actionConfig={actionConfig}
            onReport={reportChanged}
            onError={setNotice}
          />
        </aside>
      </main>

      {comparing && <CompareView documentIds={compareIds} onClose={() => setComparing(false)} />}
      {detailsFor && <DocumentDetails document={detailsFor} onClose={() => setDetailsFor(null)} />}
      {showMemory && <MemoryView onClose={() => setShowMemory(false)} />}
    </div>
  );
}
