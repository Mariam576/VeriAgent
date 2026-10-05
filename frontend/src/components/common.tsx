// Small building blocks used all over the interface.

import { useEffect, type ReactNode } from "react";
import type { Citation } from "../api/types";

export function RiskBadge({ level }: { level: string | null | undefined }) {
  if (!level) return null;
  return <span className={`badge risk-${level}`}>{level.charAt(0).toUpperCase() + level.slice(1)}</span>;
}

export function Chip({ tone = "neutral", children }: { tone?: "neutral" | "good" | "warn" | "bad" | "info"; children: ReactNode }) {
  return <span className={`chip chip-${tone}`}>{children}</span>;
}

/** A section that opens and closes (the "expandable evidence card" of the specification). */
export function Expandable({ title, badge, defaultOpen = false, children }: { title: ReactNode; badge?: ReactNode; defaultOpen?: boolean; children: ReactNode }) {
  return (
    <details className="expandable" open={defaultOpen}>
      <summary>
        <span className="expandable-title">{title}</span>
        {badge}
      </summary>
      <div className="expandable-body">{children}</div>
    </details>
  );
}

/** Where a statement comes from: document, page, and the exact words. */
export function Evidence({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <ul className="evidence">
      {citations.map((citation, index) => (
        <li key={`${citation.document_id}-${citation.page}-${index}`}>
          <span className="evidence-source">
            {citation.document} — Page {citation.page}
          </span>
          {citation.quote && <blockquote>{citation.quote}</blockquote>}
        </li>
      ))}
    </ul>
  );
}

export function Section({ title, count, children }: { title: string; count?: number; children: ReactNode }) {
  return (
    <section className="section">
      <h3>
        {title}
        {count !== undefined && <span className="count">{count}</span>}
      </h3>
      {children}
    </section>
  );
}

export function Modal({ title, onClose, wide = false, children }: { title: string; onClose: () => void; wide?: boolean; children: ReactNode }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className={`modal ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true" aria-label={title}>
        <header>
          <h2>{title}</h2>
          <button className="ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return message ? (
    <p className="error" role="alert">
      {message}
    </p>
  ) : null;
}
