// Small building blocks used all over the interface.

import { useEffect, useRef, type ReactNode } from "react";
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

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

/**
 * A dialog over the page. For people who use the keyboard or a screen reader it has to
 * behave like one: focus moves into it when it opens, Tab stays inside it, Escape closes
 * it, and afterwards focus returns to the button that opened it.
 */
export function Modal({ title, onClose, wide = false, children }: { title: string; onClose: () => void; wide?: boolean; children: ReactNode }) {
  const dialog = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close.current();
        return;
      }
      if (event.key !== "Tab" || !dialog.current) return;
      const items = Array.from(dialog.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((item) => item.offsetParent !== null);
      if (!items.length) {
        event.preventDefault();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const inside = dialog.current.contains(document.activeElement);
      if (event.shiftKey && (!inside || document.activeElement === first || document.activeElement === dialog.current)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (!inside || document.activeElement === last)) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      opener?.focus();
    };
  }, []);

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div ref={dialog} tabIndex={-1} className={`modal ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true" aria-label={title}>
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
