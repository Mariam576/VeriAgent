// Left panel, bottom: earlier investigations.

import type { InvestigationSummary } from "../api/types";
import { formatDateTime } from "../util/format";
import { Chip, RiskBadge } from "./common";

interface Props {
  items: InvestigationSummary[];
  activeId: string | null;
  onOpen: (item: InvestigationSummary) => void;
}

function state(item: InvestigationSummary) {
  if (item.approval_status === "pending") return <Chip tone="warn">Needs decision</Chip>;
  if (item.status === "failed") return <Chip tone="bad">Failed</Chip>;
  if (item.status === "interrupted") return <Chip tone="warn">Interrupted</Chip>;
  if (item.status === "running") return <Chip tone="info">Running</Chip>;
  return <RiskBadge level={item.risk_level} />;
}

export function HistoryPanel({ items, activeId, onOpen }: Props) {
  return (
    <div className="block grow">
      <div className="block-head">
        <h2>History</h2>
      </div>
      {items.length === 0 ? (
        <p className="muted small empty">Your investigations will appear here.</p>
      ) : (
        <ul className="list">
          {items.map((item) => (
            <li key={item.id}>
              <button className={`history-item ${item.id === activeId ? "active" : ""}`} onClick={() => onOpen(item)}>
                <span className="history-question">{item.question}</span>
                <span className="history-meta">
                  {state(item)}
                  <span className="muted small">{formatDateTime(item.created_at)}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
