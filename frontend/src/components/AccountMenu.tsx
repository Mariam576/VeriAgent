// Who is signed in, with password change, account deletion and sign-out.

import { useState, type FormEvent } from "react";
import { api, setAccessToken } from "../api/client";
import type { User } from "../api/types";
import { errorText } from "../util/format";
import { ErrorNote, Modal } from "./common";

interface Props {
  user: User;
  devMode: boolean;
  minPassword: number;
  onUserChanged: (user: User) => void;
  onSignOut: () => void;
  onAccountDeleted: () => void;
}

export function AccountMenu({ user, devMode, minPassword, onUserChanged, onSignOut, onAccountDeleted }: Props) {
  const [open, setOpen] = useState(false);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [problem, setProblem] = useState("");
  const [done, setDone] = useState("");

  const changePassword = async (event: FormEvent) => {
    event.preventDefault();
    setProblem("");
    setDone("");
    try {
      const session = await api.changePassword(user.has_password ? current : null, next);
      setAccessToken(session.access_token); // other sign-ins were ended; this one got a new token
      onUserChanged(session.user);
      setCurrent("");
      setNext("");
      setDone("Password changed. Other devices were signed out.");
    } catch (error) {
      setProblem(errorText(error));
    }
  };

  const deleteAccount = async () => {
    setProblem("");
    try {
      await api.deleteAccount();
      onAccountDeleted();
    } catch (error) {
      setProblem(errorText(error));
    }
  };

  return (
    <>
      <button className="ghost account" onClick={() => setOpen(true)} title="Account">
        <span className="avatar" aria-hidden="true">
          {user.name.charAt(0).toUpperCase()}
        </span>
        {user.name}
      </button>
      {open && (
        <Modal title="Account" onClose={() => setOpen(false)}>
          <p>
            <strong>{user.name}</strong>
            {user.email && <span className="muted"> · {user.email}</span>}
          </p>
          {devMode ? (
            <p className="muted">Development mode: there are no accounts on this server. The name only keeps your documents apart from others'.</p>
          ) : (
            <>
              <p className="muted small">
                Sign-in: {[user.has_password && "password", user.has_google && "Google"].filter(Boolean).join(" and ") || "none"}
              </p>
              <h3>{user.has_password ? "Change password" : "Set a password"}</h3>
              <form className="stack" onSubmit={(event) => void changePassword(event)}>
                {user.has_password && (
                  <label>
                    Current password
                    <input type="password" value={current} onChange={(event) => setCurrent(event.target.value)} autoComplete="current-password" required />
                  </label>
                )}
                <label>
                  New password
                  <input type="password" value={next} onChange={(event) => setNext(event.target.value)} autoComplete="new-password" minLength={minPassword} required />
                  <span className="hint">At least {minPassword} characters.</span>
                </label>
                <button type="submit">Save password</button>
              </form>
              {done && <p className="success">{done}</p>}

              <h3>Delete account</h3>
              <p className="muted small">Removes your account with all documents, investigations and memory. This cannot be undone.</p>
              <div className="row">
                <input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="Type DELETE" aria-label="Type DELETE to confirm" />
                <button className="danger-button" disabled={confirmation !== "DELETE"} onClick={() => void deleteAccount()}>
                  Delete my account
                </button>
              </div>
            </>
          )}
          <ErrorNote message={problem} />
          <hr />
          <button onClick={onSignOut}>Sign out</button>
        </Modal>
      )}
    </>
  );
}
