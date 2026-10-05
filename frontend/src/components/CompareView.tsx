// Documents side by side: one row per fact, one column per document.

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Comparison } from "../api/types";
import { errorText, formatValue, titleCase } from "../util/format";
import { ErrorNote, Evidence, Expandable, Modal, RiskBadge } from "./common";

export function CompareView({ documentIds, onClose }: { documentIds: string[]; onClose: () => void }) {
  const [result, setResult] = useState<Comparison | null>(null);
  const [problem, setProblem] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .compare(documentIds)
      .then((comparison) => !cancelled && setResult(comparison))
      .catch((error) => !cancelled && setProblem(errorText(error)));
    return () => {
      cancelled = true;
    };
  }, [documentIds]);

  return (
    <Modal title="Compare documents" onClose={onClose} wide>
      <ErrorNote message={problem} />
      {!result && !problem && <p className="muted">Reading the documents... The first comparison of a document takes a moment; after that its key facts are stored.</p>}
      {result && (
        <>
          {result.notes.map((note) => (
            <p key={note} className="muted small">
              {note}
            </p>
          ))}

          <h3>
            Differences found <span className="count">{result.findings.length}</span>
          </h3>
          {result.findings.length === 0 ? (
            <p className="muted">No differences between the facts these documents state. A person should still read them.</p>
          ) : (
            result.findings.map((finding, index) => (
              <Expandable key={index} title={finding.description} badge={<RiskBadge level={finding.severity} />} defaultOpen={index === 0}>
                <dl className="facts">
                  <dt>Finding</dt>
                  <dd>{titleCase(finding.type)}</dd>
                  {finding.difference !== null && (
                    <>
                      <dt>Difference</dt>
                      <dd>
                        {formatValue(finding.difference, finding.unit_b ?? finding.unit_a)}
                        {finding.percentage_change !== null && ` (${formatValue(finding.percentage_change, "%")})`}
                      </dd>
                    </>
                  )}
                </dl>
                <Evidence citations={finding.citations} />
              </Expandable>
            ))
          )}

          <h3>Key facts, side by side</h3>
          <div className="table-wrap">
            <table className="compare">
              <thead>
                <tr>
                  <th />
                  {result.documents.map((document) => (
                    <th key={document.document_id}>
                      {document.filename}
                      <span className="muted small"> {titleCase(document.document_type)}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => (
                  <tr key={row.field} className={row.differs ? "differs" : ""}>
                    <th scope="row">{row.label}</th>
                    {row.cells.map((cell, index) => (
                      <td key={index} title={cell ? `Page ${cell.page}: ${cell.quote}` : undefined}>
                        {cell ? (
                          <>
                            {cell.value} <span className="muted small page">p. {cell.page}</span>
                          </>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">Rows where the documents disagree are highlighted. Hover over a value to see the words it was read from.</p>

          {result.line_items.map((lines) => (
            <div key={lines.document_id}>
              <h3>Lines of {lines.filename}</h3>
              <div className="table-wrap">
                <table className="compare">
                  <thead>
                    <tr>
                      <th>Description</th>
                      <th>Quantity</th>
                      <th>Unit price</th>
                      <th>Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lines.items.map((item, index) => (
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
            </div>
          ))}
        </>
      )}
    </Modal>
  );
}
