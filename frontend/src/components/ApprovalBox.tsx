// The human decision: approve, reject or replace the agent's recommendation,
// and choose which proposed follow-up actions may be carried out.
//
// Nothing leaves VeriAgent unless its box is ticked here and "Approve" (or
// "Modify") is sent. The exact text that would be sent can be read first.

import { useState } from "react";
import { api } from "../api/client";
import type { ActionConfig, Report } from "../api/types";
import { errorText } from "../util/format";
import { ErrorNote, Expandable } from "./common";

interface Props {
  report: Report;
  actionConfig: ActionConfig | null;
  onDecided: (report: Report) => void;
  onError: (message: string) => void;
}

type Choice = "approve" | "reject" | "modify";

export function ApprovalBox({ report, actionConfig, onDecided }: Props) {
  const [choice, setChoice] = useState<Choice>("approve");
  const [note, setNote] = useState("");
  const [replacement, setReplacement] = useState("");
  const [ticked, setTicked] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const proposals = report.actions.filter((action) => action.status === "proposed");

  const send = async () => {
    if (choice === "modify" && !replacement.trim()) {
      setProblem("Write the recommendation that should replace the agent's.");
      return;
    }
    setBusy(true);
    setProblem("");
    try {
      onDecided(
        await api.decide(report.investigation_id, {
          decision: choice,
          note: note.trim() || undefined,
          modified_action: choice === "modify" ? replacement.trim() : undefined,
          approved_actions: choice === "reject" ? [] : ticked,
        }),
      );
    } catch (error) {
      setProblem(errorText(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="approval" role="group" aria-label="Human decision">
      <h3>Your decision is needed</h3>
      <p className="muted small">The investigation is paused. The report is a draft until you decide.</p>

      <div className="choices">
        {(["approve", "reject", "modify"] as const).map((option) => (
          <label key={option} className={`choice ${choice === option ? "active" : ""}`}>
            <input type="radio" name="decision" value={option} checked={choice === option} onChange={() => setChoice(option)} />
            {option === "approve" ? "Approve" : option === "reject" ? "Reject" : "Modify"}
          </label>
        ))}
      </div>
      <p className="muted small">
        {choice === "approve" && "Accept the agent's recommendation as it stands."}
        {choice === "reject" && "Do not accept the recommendation. Nothing proposed below is carried out."}
        {choice === "modify" && "Replace the recommendation with your own. The agent's wording is kept for the record."}
      </p>

      {choice === "modify" && (
        <label>
          Your recommendation
          <textarea value={replacement} onChange={(event) => setReplacement(event.target.value)} rows={3} maxLength={1000} />
        </label>
      )}
      <label>
        Note (optional, shown in the report)
        <textarea value={note} onChange={(event) => setNote(event.target.value)} rows={2} maxLength={1000} />
      </label>

      {proposals.length > 0 && choice !== "reject" && (
        <div className="proposals">
          <h4>Follow-up outside VeriAgent</h4>
          {proposals.map((action) => (
            <div key={action.id} className="proposal">
              <label className="tick">
                <input
                  type="checkbox"
                  checked={ticked.includes(action.id)}
                  onChange={() => setTicked((current) => (current.includes(action.id) ? current.filter((id) => id !== action.id) : [...current, action.id]))}
                />
                Create an issue in {action.target}
              </label>
              <Expandable title="Read exactly what would be sent">
                <p>
                  <strong>{action.title}</strong>
                </p>
                <pre className="issue-body">{action.body}</pre>
              </Expandable>
            </div>
          ))}
        </div>
      )}
      {proposals.length === 0 && actionConfig?.enabled === false && report.risk_level !== "low" && report.risk_level !== "medium" && (
        <p className="muted small">No follow-up action is configured on this server (ACTION_SERVER).</p>
      )}

      <ErrorNote message={problem} />
      <button className="primary" onClick={() => void send()} disabled={busy}>
        {busy ? "Sending..." : "Send decision"}
      </button>
    </div>
  );
}
