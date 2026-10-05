// Left panel, top: your documents and the upload.

import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { api } from "../api/client";
import type { DocumentInfo } from "../api/types";
import { errorText, titleCase } from "../util/format";
import { Chip } from "./common";

interface Props {
  documents: DocumentInfo[];
  compareIds: string[];
  onToggleCompare: (id: string) => void;
  onCompare: () => void;
  onChanged: () => void;
  onDetails: (document: DocumentInfo) => void;
  onError: (message: string) => void;
}

const TYPES = ["", "contract", "invoice", "purchase_order", "policy", "quotation", "receipt", "report", "other"];

export function DocumentsPanel({ documents, compareIds, onToggleCompare, onCompare, onChanged, onDetails, onError }: Props) {
  const [documentType, setDocumentType] = useState("");
  const [uploading, setUploading] = useState(0);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setUploading((count) => count + files.length);
    onError("");
    for (const file of files) {
      try {
        await api.upload(file, documentType);
      } catch (error) {
        onError(`${file.name}: ${errorText(error)}`);
      } finally {
        setUploading((count) => count - 1);
        onChanged();
      }
    }
  };

  const picked = (event: ChangeEvent<HTMLInputElement>) => {
    void upload(Array.from(event.target.files ?? []));
    event.target.value = ""; // so the same file can be picked again
  };

  const dropped = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    void upload(Array.from(event.dataTransfer.files));
  };

  const remove = async (document: DocumentInfo) => {
    if (!window.confirm(`Delete ${document.filename}? Its text and search index are removed. Reports of earlier investigations keep their text.`)) return;
    try {
      await api.deleteDocument(document.id);
    } catch (error) {
      onError(errorText(error));
    }
    onChanged();
  };

  return (
    <div className="block">
      <div className="block-head">
        <h2>Documents</h2>
        {compareIds.length >= 1 && (
          <button className="small-button" onClick={onCompare} title="Put the ticked documents side by side">
            Compare {compareIds.length}
          </button>
        )}
      </div>

      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={dropped}
      >
        <input ref={input} type="file" accept=".pdf,.docx,.txt" multiple hidden onChange={picked} />
        <button className="primary" onClick={() => input.current?.click()} disabled={uploading > 0}>
          {uploading > 0 ? `Uploading ${uploading}...` : "Upload files"}
        </button>
        <span className="muted small">PDF, DOCX or TXT. Drop files here.</span>
        <label className="inline small">
          Type
          <select value={documentType} onChange={(event) => setDocumentType(event.target.value)} aria-label="Document type">
            {TYPES.map((type) => (
              <option key={type} value={type}>
                {type ? titleCase(type) : "Guess from the file"}
              </option>
            ))}
          </select>
        </label>
      </div>

      {documents.length === 0 ? (
        <p className="muted small empty">No documents yet. Upload a contract and an invoice to begin.</p>
      ) : (
        <ul className="list">
          {documents.map((document) => (
            <li key={document.id} className="document">
              <input
                type="checkbox"
                checked={compareIds.includes(document.id)}
                disabled={document.status !== "ready"}
                onChange={() => onToggleCompare(document.id)}
                aria-label={`Select ${document.filename} for comparison`}
              />
              <button className="document-main" onClick={() => onDetails(document)} disabled={document.status !== "ready"} title="Key facts and file inspection">
                <span className="document-name">{document.filename}</span>
                <span className="document-meta">
                  {document.status === "ready" && (
                    <>
                      {titleCase(document.document_type)} · {document.page_count} page{document.page_count === 1 ? "" : "s"}
                    </>
                  )}
                  {document.status === "processing" && <Chip tone="info">Reading...</Chip>}
                  {document.status === "failed" && <Chip tone="bad">Failed</Chip>}
                  {document.integrity_level === "high" && <Chip tone="bad">Signs of editing</Chip>}
                  {document.integrity_level === "medium" && <Chip tone="warn">Check file</Chip>}
                </span>
                {document.status === "failed" && document.error && <span className="document-error">{document.error}</span>}
              </button>
              <button className="ghost danger" onClick={() => void remove(document)} aria-label={`Delete ${document.filename}`} title="Delete">
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
