// The sign-in page: email and password, creating an account, and "Sign in with Google".

import { useEffect, useRef, useState, type FormEvent } from "react";
import { api } from "../api/client";
import type { AuthConfig, TokenResponse } from "../api/types";
import { errorText } from "../util/format";

interface Props {
  config: AuthConfig;
  onSignedIn: (session: TokenResponse) => void;
  onDevUser: (name: string) => void;
}

// The part of Google's sign-in script this page uses.
interface GoogleIdentity {
  accounts: {
    id: {
      initialize: (options: { client_id: string; callback: (response: { credential: string }) => void }) => void;
      renderButton: (element: HTMLElement, options: Record<string, string | number>) => void;
    };
  };
}

declare global {
  interface Window {
    google?: GoogleIdentity;
  }
}

function GoogleButton({ clientId, onCredential }: { clientId: string; onCredential: (credential: string) => void }) {
  const holder = useRef<HTMLDivElement>(null);
  const handler = useRef(onCredential);
  handler.current = onCredential;

  useEffect(() => {
    const show = () => {
      if (!window.google || !holder.current) return;
      window.google.accounts.id.initialize({ client_id: clientId, callback: (response) => handler.current(response.credential) });
      window.google.accounts.id.renderButton(holder.current, { theme: "outline", size: "large", width: 320, text: "continue_with" });
    };
    if (window.google) {
      show();
      return;
    }
    // Google's own script draws the button and talks to Google. It hands back an ID token,
    // which the server verifies (app/security/google.py). This page never sees a Google password.
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.onload = show;
    document.head.appendChild(script);
    return () => {
      script.onload = null;
    };
  }, [clientId]);

  return <div ref={holder} className="google-button" />;
}

export function SignIn({ config, onSignedIn, onDevUser }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [signupCode, setSignupCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const run = async (action: () => Promise<TokenResponse>) => {
    setBusy(true);
    setError("");
    try {
      onSignedIn(await action());
    } catch (problem) {
      setError(errorText(problem));
    } finally {
      setBusy(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (config.auth_mode === "dev") {
      if (name.trim()) onDevUser(name.trim());
      return;
    }
    void run(() => (mode === "login" ? api.login(email, password) : api.register(email, password, name, signupCode)));
  };

  if (config.auth_mode === "dev") {
    return (
      <div className="signin">
        <form className="card signin-card" onSubmit={submit}>
          <Brand />
          <p className="muted">
            This server runs in development mode: there are no accounts. Pick any name; your documents are kept under it.
          </p>
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={60} autoFocus required />
          </label>
          <button className="primary" type="submit">
            Continue
          </button>
        </form>
      </div>
    );
  }

  const registering = mode === "register";
  return (
    <div className="signin">
      <form className="card signin-card" onSubmit={submit}>
        <Brand />
        <div className="tabs" role="tablist">
          <button type="button" role="tab" aria-selected={!registering} className={registering ? "" : "active"} onClick={() => setMode("login")}>
            Sign in
          </button>
          {config.registration_open && (
            <button type="button" role="tab" aria-selected={registering} className={registering ? "active" : ""} onClick={() => setMode("register")}>
              Create account
            </button>
          )}
        </div>

        {registering && (
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={120} autoComplete="name" />
          </label>
        )}
        <label>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" required autoFocus />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={registering ? "new-password" : "current-password"}
            minLength={registering ? config.min_password_chars : 1}
            required
          />
          {registering && <span className="hint">At least {config.min_password_chars} characters.</span>}
        </label>
        {registering && config.signup_code_required && (
          <label>
            Sign-up code
            <input value={signupCode} onChange={(e) => setSignupCode(e.target.value)} required />
            <span className="hint">The person who runs this server gives it to you.</span>
          </label>
        )}

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Please wait..." : registering ? "Create account" : "Sign in"}
        </button>

        {config.google_client_id && (
          <>
            <div className="divider">or</div>
            <GoogleButton clientId={config.google_client_id} onCredential={(credential) => void run(() => api.google(credential, signupCode))} />
          </>
        )}
      </form>
    </div>
  );
}

function Brand() {
  return (
    <div className="brand brand-large">
      <span className="brand-mark" aria-hidden="true">
        ✓
      </span>
      <div>
        <h1>VeriAgent</h1>
        <p className="muted">Evidence verification for business documents</p>
      </div>
    </div>
  );
}
