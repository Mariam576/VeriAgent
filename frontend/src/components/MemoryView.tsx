// Long-term memory: your preferences and the notes VeriAgent keeps between sessions.

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api } from "../api/client";
import type { MemoryOverview } from "../api/types";
import { errorText, formatDateTime, titleCase } from "../util/format";
import { ErrorNote, Modal } from "./common";

export function MemoryView({ onClose }: { onClose: () => void }) {
  const [memory, setMemory] = useState<MemoryOverview | null>(null);
  const [note, setNote] = useState("");
  const [problem, setProblem] = useState("");

  const load = useCallback(async () => {
    try {
      setMemory(await api.memory());
    } catch (error) {
      setProblem(errorText(error));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const run = async (action: () => Promise<unknown>) => {
    setProblem("");
    try {
      await action();
      await load();
    } catch (error) {
      setProblem(errorText(error));
    }
  };

  const add = (event: FormEvent) => {
    event.preventDefault();
    if (note.trim().length < 10) {
      setProblem("A note needs at least 10 characters.");
      return;
    }
    void run(async () => {
      await api.addNote(note.trim());
      setNote("");
    });
  };

  return (
    <Modal title="Memory" onClose={onClose}>
      <ErrorNote message={problem} />
      {!memory && !problem && <p className="muted">Loading...</p>}
      {memory && !memory.enabled && <p className="muted">Long-term memory is switched off on this server.</p>}
      {memory?.enabled && (
        <>
          <h3>Preferences</h3>
          {memory.preferences.map((preference) => (
            <label key={preference.key} className="preference">
              <span>
                {titleCase(preference.key)}
                <span className="hint">{preference.description}</span>
              </span>
              <select value={preference.value} onChange={(event) => void run(() => api.setPreference(preference.key, event.target.value))}>
                {preference.allowed_values.map((value) => (
                  <option key={value} value={value}>
                    {titleCase(value)}
                  </option>
                ))}
              </select>
            </label>
          ))}

          <h3>
            Notes <span className="count">{memory.notes.length}</span>
          </h3>
          <p className="muted small">Lasting facts and one-line summaries of finished investigations. They give context to later questions; they are never evidence.</p>
          <form className="row" onSubmit={add}>
            <input value={note} onChange={(event) => setNote(event.target.value)} placeholder="A lasting fact to remember..." maxLength={500} aria-label="New note" />
            <button type="submit">Add</button>
          </form>
          <ul className="list notes">
            {memory.notes.map((item) => (
              <li key={item.id}>
                <span>
                  {item.text}
                  <span className="muted small">
                    {" "}
                    {item.source === "user" ? "you" : "investigation"} · {formatDateTime(item.created_at)}
                  </span>
                </span>
                <button className="ghost danger" onClick={() => void run(() => api.deleteNote(item.id))} aria-label="Delete note">
                  ✕
                </button>
              </li>
            ))}
          </ul>
          {memory.notes.length > 0 && (
            <button className="danger-button" onClick={() => window.confirm("Delete all preferences and notes?") && void run(() => api.forgetEverything())}>
              Forget everything
            </button>
          )}
        </>
      )}
    </Modal>
  );
}
