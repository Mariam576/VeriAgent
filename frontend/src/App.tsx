// The top of the interface: who is signed in, and what to show.
//
//   loading -> (no session) -> sign-in page -> workspace
//
// On every page load the refresh cookie is tried first, so a signed-in person
// stays signed in after pressing F5 without the page ever storing a token.

import { useCallback, useEffect, useState } from "react";
import { api, onSessionEnded, setAccessToken, setDevUser } from "./api/client";
import type { AuthConfig, TokenResponse, User } from "./api/types";
import { SignIn } from "./auth/SignIn";
import { Workspace } from "./components/Workspace";
import { errorText } from "./util/format";

const DEV_NAME_KEY = "veriagent-dev-name"; // a display name, not a secret

export function App() {
  const [config, setConfig] = useState<AuthConfig | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [problem, setProblem] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const loaded = await api.authConfig();
        if (cancelled) return;
        setConfig(loaded);
        if (loaded.auth_mode === "dev") {
          const name = sessionStorage.getItem(DEV_NAME_KEY);
          if (name) {
            setDevUser(name);
            setUser(await api.me());
          }
        } else {
          const session = await api.restore();
          if (session && !cancelled) {
            setAccessToken(session.access_token);
            setUser(session.user);
          }
        }
      } catch (error) {
        if (!cancelled) setProblem(errorText(error));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const signedOut = useCallback(() => {
    setAccessToken(null);
    setDevUser(null);
    sessionStorage.removeItem(DEV_NAME_KEY);
    setUser(null);
  }, []);

  useEffect(() => {
    onSessionEnded(signedOut);
    return () => onSessionEnded(null);
  }, [signedOut]);

  const signedIn = (session: TokenResponse) => {
    setAccessToken(session.access_token);
    setUser(session.user);
  };

  const devUser = async (name: string) => {
    setDevUser(name);
    sessionStorage.setItem(DEV_NAME_KEY, name);
    try {
      setUser(await api.me());
    } catch (error) {
      setProblem(errorText(error));
    }
  };

  const signOut = async () => {
    try {
      if (config?.auth_mode !== "dev") await api.logout();
    } catch {
      // signing out locally is what matters
    }
    signedOut();
  };

  if (loading) return <div className="center muted">Loading...</div>;
  if (problem || !config) {
    return (
      <div className="center">
        <div className="card">
          <h2>The server cannot be reached</h2>
          <p className="muted">{problem || "No answer from the API."}</p>
          <button onClick={() => window.location.reload()}>Try again</button>
        </div>
      </div>
    );
  }
  if (!user) return <SignIn config={config} onSignedIn={signedIn} onDevUser={(name) => void devUser(name)} />;
  return <Workspace user={user} config={config} onUserChanged={setUser} onSignOut={() => void signOut()} onAccountDeleted={signedOut} />;
}
