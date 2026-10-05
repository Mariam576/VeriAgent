// Middle panel: the conversation, with live progress while the agent works.

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { RiskLevel } from "../api/types";
import { RiskBadge } from "./common";

export interface ChatMessage {
  role: "user" | "assistant" | "error";
  text: string;
  risk?: RiskLevel;
  status?: string;
  pending?: boolean;
  investigationId?: string;
}

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  progress: string[];
  hasDocuments: boolean;
  onAsk: (question: string) => void;
  onStop: () => void;
  onNew: () => void;
}

const EXAMPLES = [
  "Review this invoice against the contract and tell me whether it should be approved.",
  "By how much does the invoice exceed the contract value?",
  "What payment terms does the contract set?",
];

export function Chat({ messages, busy, progress, hasDocuments, onAsk, onStop, onNew }: Props) {
  const [draft, setDraft] = useState("");
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages, progress]);

  const send = (text: string) => {
    const question = text.trim();
    if (!question || busy) return;
    setDraft("");
    onAsk(question);
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    send(draft);
  };

  const keyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(draft);
    }
  };

  return (
    <div className="chat">
      <div className="block-head">
        <h2>Chat</h2>
        {messages.length > 0 && (
          <button className="small-button" onClick={onNew} disabled={busy}>
            New conversation
          </button>
        )}
      </div>

      <div className="messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="welcome">
            <h3>Ask about your documents</h3>
            <p className="muted">
              VeriAgent searches the documents you uploaded, compares them, calculates differences with tools and tells you where each
              statement comes from. It never decides for you: findings that matter wait for your approval.
            </p>
            {!hasDocuments && <p className="muted">Start by uploading documents on the left.</p>}
            <div className="examples">
              {EXAMPLES.map((example) => (
                <button key={example} className="example" onClick={() => send(example)} disabled={busy || !hasDocuments}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`message message-${message.role}`}>
            {message.role === "assistant" && (
              <div className="message-meta">
                <RiskBadge level={message.risk} />
                {message.pending && <span className="chip chip-warn">Waiting for your decision</span>}
              </div>
            )}
            <p>{message.text}</p>
          </div>
        ))}

        {busy && (
          <div className="message message-assistant working">
            <ol className="progress">
              {progress.length === 0 && <li className="current">Starting...</li>}
              {progress.map((line, index) => (
                <li key={index} className={index === progress.length - 1 ? "current" : "done"}>
                  {line}
                </li>
              ))}
            </ol>
          </div>
        )}
        <div ref={end} />
      </div>

      <form className="composer" onSubmit={submit}>
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={keyDown}
          placeholder={hasDocuments ? "Ask a question about your documents..." : "Upload documents first, then ask a question..."}
          rows={2}
          maxLength={2000}
          aria-label="Your question"
        />
        {busy ? (
          <button type="button" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="primary" type="submit" disabled={!draft.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}
