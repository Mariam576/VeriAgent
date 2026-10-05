// One document: its key facts (the "profile") and the result of the file inspection.

import { useEffect, useState } from "react";
import { api, type DocumentProfile } from "../api/client";
import type { DocumentInfo, ForensicsOut, ProfileField } from "../api/types";
import { errorText, formatValue, titleCase } from "../util/format";
import { Chip, ErrorNote, Modal } from "./common";

const FIELDS: [string, string][] = [
  ["number", "Document number"],
  ["contract_reference", "Contract reference"],
  ["purchase_order_reference", "Purchase order reference"],
  ["date", "Date"],
  ["term_start", "Term starts"],
  ["term_end", "Term ends"],
  ["term_months", "Term length"],
  ["total", "Total amount"],
  ["subtotal", "Subtotal"],
  ["tax", "Tax"],
  ["payment_terms_days", "Payment terms"],
];

function show(name: string, field: ProfileField): string {
  if (field.number !== null) return name === "term_months" ? `${field.number} months` : formatValue(field.number, field.unit);
  return field.text ?? "";
}

function Fact({ label, value, field }: { label: string; value: string; field: ProfileField }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>
        {value} <span className="muted small page">p. {field.page}</span>
        <blockquote>{field.quote}</blockquote>
      </dd>
    </>
  );
}

export function DocumentDetails({ document, onClose }: { document: DocumentInfo; onClose: () => void }) {
  const [profile, setProfile] = useState<DocumentProfile | null>(null);
  const [forensics, setForensics] = useState<ForensicsOut | null>(null);
  const [problem, setProblem] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .forensics(document.id)
      .then((result) => !cancelled && setForensics(result))
      .catch((error) => !cancelled && setProblem(errorText(error)));
    api
      .profile(document.id)
      .then((result) => !cancelled && setProfile(result))
      .catch((error) => !cancelled && setProblem(errorText(error)));
    return () => {
      cancelled = true;
    };
  }, [document.id]);

  const report = forensics?.report;
  const facts = profile?.profile;
  return (
    <Modal title={document.filename} onClose={onClose} wide>
      <ErrorNote message={problem} />

      <h3>File inspection</h3>
      {!forensics && !problem && <p className="muted">Loading...</p>}
      {forensics && !report && <p className="muted">{forensics.note}</p>}
      {report && (
        <>
          <p>
            <Chip tone={report.level === "high" ? "bad" : report.level === "medium" ? "warn" : "good"}>
              {report.level === "none" ? "Nothing found" : `${titleCase(report.level)} (score ${report.score.toFixed(2)})`}
            </Chip>{" "}
            <span className="muted small">A risk indicator, never proof that a document is forged.</span>
          </p>
          {report.note && <p className="muted small">{report.note}</p>}
          {report.indicators.length > 0 && (
            <ul className="plain">
              {report.indicators.map((indicator, index) => (
                <li key={index}>
                  <strong>{titleCase(indicator.type)}.</strong> {indicator.reason}
                  {indicator.region && indicator.type !== "hidden_instruction" && <blockquote>{indicator.region}</blockquote>}
                </li>
              ))}
            </ul>
          )}
          {Object.keys(report.metadata).length > 0 && (
            <dl className="facts compact">
              {Object.entries(report.metadata).map(([name, value]) => (
                <span key={name} className="pair">
                  <dt>{titleCase(name)}</dt>
                  <dd>{String(value)}</dd>
                </span>
              ))}
            </dl>
          )}
        </>
      )}

      <h3>Key facts</h3>
      {!profile && !problem && <p className="muted">Reading the document... This takes a moment the first time.</p>}
      {profile && !facts && <p className="muted">{profile.note}</p>}
      {facts && (
        <>
          {profile?.note && <p className="muted">{profile.note}</p>}
          <dl className="facts">
            {FIELDS.map(([name, label]) => {
              const field = (facts as unknown as Record<string, ProfileField | null>)[name];
              return field ? <Fact key={name} label={label} value={show(name, field)} field={field} /> : null;
            })}
            {facts.parties.map((party, index) => (
              <Fact key={index} label={party.role ? titleCase(party.role) : "Party"} value={party.text ?? ""} field={party} />
            ))}
          </dl>
          {facts.line_items.length > 0 && (
            <div className="table-wrap">
              <table className="compare">
                <thead>
                  <tr>
                    <th>Line</th>
                    <th>Quantity</th>
                    <th>Unit price</th>
                    <th>Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {facts.line_items.map((item, index) => (
                    <tr key={index} title={`Page ${item.page}: ${item.quote}`}>
                      <td>{item.description}</td>
                      <td>{item.quantity ?? "—"}</td>
                      <td>{item.unit_price !== null ? formatValue(item.unit_price, item.unit) : "—"}</td>
                      <td>{item.amount !== null ? formatValue(item.amount, item.unit) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="muted small">Read by {facts.built_by === "llm" ? "the language model, then checked against the page" : "rules"}. Every value shows the words it was taken from.</p>
        </>
      )}
    </Modal>
  );
}
