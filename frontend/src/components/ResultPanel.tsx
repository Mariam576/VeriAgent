// Right panel: the structured result of one investigation.
//
// Everything shown here is plain text from the report. React writes text into the
// page as text, never as HTML, so nothing a document says can turn into a link, an
// image or a script.

import { useState } from "react";
import { api } from "../api/client";
import type { ActionConfig, Contradiction, FollowUpAction, InvestigationSummary, Report } from "../api/types";
import { errorText, formatValue, percent, safeLink, titleCase } from "../util/format";
import { ApprovalBox } from "./ApprovalBox";
import { Chip, ErrorNote, Evidence, Expandable, RiskBadge, Section } from "./common";

interface Props {
  report: Report | null;
  summary: InvestigationSummary | null;
  actionConfig: ActionConfig | null;
  onReport: (report: Report) => void;
  onError: (message: string) => void;
}

function sides(item: Contradiction): { a: string; b: string } | null {
  const a = item.value_a !== null ? formatValue(item.value_a, item.unit_a) : item.text_a;
  const b = item.value_b !== null ? formatValue(item.value_b, item.unit_b) : item.text_b;
  return a && b ? { a, b } : null;
}

function reviewLine(report: Report): string {
  const review = report.human_review;
  if (review.status === "pending") return "Waiting for your decision";
  if (review.status === "approved" || review.status === "rejected" || review.status === "modified") {
    return `${titleCase(review.status)} by ${review.reviewer ?? "reviewer"}${review.note ? ` — ${review.note}` : ""}`;
  }
  return review.required ? "Recommended (pausing is switched off)" : "Not required";
}

function ActionLine({ action }: { action: FollowUpAction }) {
  const link = safeLink(action.url);
  const tone = action.status === "done" ? "good" : action.status === "proposed" ? "info" : action.status === "not_approved" ? "neutral" : "bad";
  const label = { proposed: "Proposed", not_approved: "Not approved", done: "Created", failed: "Failed", unknown: "Result unknown" }[action.status];
  return (
    <Expandable title={`Issue in ${action.target}`} badge={<Chip tone={tone}>{label}</Chip>}>
      <p>
        <strong>{action.title}</strong>
      </p>
      {action.status === "done" && (
        <p>
          Reference: {link ? <a href={link} target="_blank" rel="noopener noreferrer">{action.reference ?? link}</a> : (action.reference ?? "created")}
        </p>
      )}
      {action.detail && <p className="muted">{action.detail}</p>}
      {action.body && <pre className="issue-body">{action.body}</pre>}
    </Expandable>
  );
}

export function ResultPanel({ report, summary, actionConfig, onReport, onError }: Props) {
  const [resuming, setResuming] = useState(false);
  const [problem, setProblem] = useState("");

  const resume = async (id: string) => {
    setResuming(true);
    setProblem("");
    try {
      onReport(await api.resume(id));
    } catch (error) {
      setProblem(errorText(error));
    } finally {
      setResuming(false);
    }
  };

  if (!report) {
    if (summary) {
      // An investigation without a report: it failed, or its run was cut off.
      const canResume = summary.status === "interrupted" || summary.status === "running";
      return (
        <div className="result">
          <div className="block-head">
            <h2>Result</h2>
          </div>
          <p className="result-question">{summary.question}</p>
          <p>
            <Chip tone={summary.status === "failed" ? "bad" : "warn"}>{titleCase(summary.status)}</Chip>
          </p>
          {summary.error && <p className="muted">{summary.error}</p>}
          {canResume && (
            <>
              <p className="muted">The run stopped before it finished. It can continue from the last completed step.</p>
              <button className="primary" onClick={() => void resume(summary.id)} disabled={resuming}>
                {resuming ? "Continuing..." : "Continue"}
              </button>
            </>
          )}
          <ErrorNote message={problem} />
        </div>
      );
    }
    return (
      <div className="result">
        <div className="block-head">
          <h2>Result</h2>
        </div>
        <p className="muted empty">Ask a question, or open an investigation from the history. The findings and their evidence appear here.</p>
      </div>
    );
  }

  if (report.status === "not_applicable") {
    return (
      <div className="result">
        <div className="block-head">
          <h2>Result</h2>
        </div>
        <p>{report.summary}</p>
      </div>
    );
  }

  const facts = report.key_findings.filter((finding) => finding.type === "fact");
  const calculations = report.key_findings.filter((finding) => finding.type === "calculation");

  return (
    <div className="result">
      <div className="block-head">
        <h2>Result</h2>
        <Chip tone={report.status === "no_issues_found" ? "good" : report.status === "review_required" ? "warn" : "neutral"}>{report.status_label}</Chip>
      </div>
      <p className="result-question">{report.question}</p>

      <div className="metrics">
        <div className={`metric metric-risk risk-${report.risk_level}`}>
          <span className="metric-label">Risk level</span>
          <span className="metric-value">{titleCase(report.risk_level)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Confidence</span>
          <span className="metric-value">{percent(report.confidence)}</span>
          <span className="meter" aria-hidden="true">
            <span style={{ width: percent(report.confidence) }} />
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Evidence</span>
          <span className="metric-value">{titleCase(report.evidence_quality)}</span>
        </div>
      </div>

      <p className="summary">{report.summary}</p>

      {report.human_review.status === "pending" ? (
        // key: a fresh form for every investigation. Without it the chosen decision, the note and
        // a ticked "create an issue" box would carry over to the next investigation that is opened.
        <ApprovalBox key={report.investigation_id} report={report} actionConfig={actionConfig} onDecided={onReport} onError={onError} />
      ) : (
        <p className="review-line">
          <strong>Human review:</strong> {reviewLine(report)}
        </p>
      )}

      <Section title="Recommendation">
        <p className="recommendation">{report.recommended_action || "No recommendation."}</p>
        {report.human_review.original_action && <p className="muted small">The agent had recommended: {report.human_review.original_action}</p>}
      </Section>

      {report.actions.length > 0 && report.human_review.status !== "pending" && (
        <Section title="Follow-up actions" count={report.actions.length}>
          {report.actions.map((action) => (
            <ActionLine key={action.id} action={action} />
          ))}
        </Section>
      )}

      {report.contradictions.length > 0 && (
        <Section title="Contradictions" count={report.contradictions.length}>
          {report.contradictions.map((item) => {
            const values = sides(item);
            return (
              <Expandable key={item.id} title={item.description} badge={<RiskBadge level={item.severity} />} defaultOpen={report.contradictions.length <= 2}>
                <dl className="facts">
                  <dt>Finding</dt>
                  <dd>{titleCase(item.type)}</dd>
                  {values && (
                    <>
                      <dt>{item.source_a}</dt>
                      <dd>{values.a}</dd>
                      <dt>{item.source_b}</dt>
                      <dd>{values.b}</dd>
                    </>
                  )}
                  {item.difference !== null && (
                    <>
                      <dt>Difference</dt>
                      <dd>
                        {formatValue(item.difference, item.unit_b ?? item.unit_a)}
                        {item.percentage_change !== null && ` (${formatValue(item.percentage_change, "%")})`}
                        <span className="muted small"> calculated by a tool</span>
                      </dd>
                    </>
                  )}
                </dl>
                <h4>Evidence</h4>
                <Evidence citations={item.citations} />
              </Expandable>
            );
          })}
        </Section>
      )}

      {report.document_integrity.length > 0 && (
        <Section title="Document integrity" count={report.document_integrity.length}>
          <p className="muted small">Signs that a file may have been altered. These are risk indicators, not proof.</p>
          {report.document_integrity.map((item) => (
            <Expandable key={item.id} title={item.document} badge={<RiskBadge level={item.level} />} defaultOpen>
              <ul className="plain">
                {item.indicators.map((indicator, index) => (
                  <li key={index}>
                    {indicator.reason}
                    {indicator.region && indicator.type !== "hidden_instruction" && <blockquote>{indicator.region}</blockquote>}
                  </li>
                ))}
              </ul>
            </Expandable>
          ))}
        </Section>
      )}

      {report.missing_evidence.length > 0 && (
        <Section title="Missing evidence" count={report.missing_evidence.length}>
          <ul className="plain">
            {report.missing_evidence.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
        </Section>
      )}

      {facts.length > 0 && (
        <Section title="Key findings" count={facts.length}>
          {facts.map((finding, index) => (
            <Expandable key={index} title={finding.statement}>
              <Evidence citations={finding.citations} />
            </Expandable>
          ))}
        </Section>
      )}

      {(calculations.length > 0 || report.calculations.length > 0) && (
        <Section title="Calculations" count={report.calculations.length}>
          <ul className="plain">
            {calculations.map((finding, index) => (
              <li key={index}>{finding.statement}</li>
            ))}
          </ul>
          <Expandable title="How they were calculated">
            <ul className="plain mono">
              {report.calculations.map((item) => (
                <li key={item.id}>
                  {item.description}: {item.tool_name}({Object.entries(item.arguments).map(([name, value]) => `${name}=${JSON.stringify(value)}`).join(", ")}) = {item.result}
                </li>
              ))}
            </ul>
          </Expandable>
        </Section>
      )}

      {report.risk_reasons.length > 0 && (
        <Section title="Why this risk level" count={report.risk_reasons.length}>
          {report.risk_reasons.map((reason, index) => (
            <Expandable key={index} title={reason.statement}>
              <Evidence citations={reason.citations} />
            </Expandable>
          ))}
        </Section>
      )}

      {report.sources.length > 0 && (
        <Section title="Sources" count={report.sources.length}>
          <ul className="plain">
            {report.sources.map((source, index) => (
              <li key={index}>
                {source.document} — Page {source.page}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {report.memory.recalled.length > 0 && (
        <Section title="From earlier investigations">
          <p className="muted small">Memory, not evidence from the current documents.</p>
          <ul className="plain">
            {report.memory.recalled.map((note, index) => (
              <li key={index}>{note.text}</li>
            ))}
          </ul>
        </Section>
      )}

      {report.limitations.length > 0 && (
        <Section title="Limitations">
          <ul className="plain muted">
            {report.limitations.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
        </Section>
      )}

      <p className="muted small stats">
        {report.stats.llm_calls} model call(s) · {report.stats.tool_calls} tool call(s) · {report.stats.retrieved_chunks} passages read ·{" "}
        {(report.stats.duration_ms / 1000).toFixed(1)} s
      </p>
    </div>
  );
}
