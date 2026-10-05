// What the page does when its access token is no longer accepted.
//
// fetch() is replaced by a script of answers, so no server is needed.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, onSessionEnded, setAccessToken, setRefreshRetryMs } from "./client";

type Answer = { status: number; body?: unknown } | "network-error";

function json(status: number, body: unknown = {}): Response {
  if (status === 204) return new Response(null, { status }); // "no content" may not carry a body
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Answers per path, used in order; the last answer for a path repeats. */
function server(script: Record<string, Answer[]>) {
  const calls: { path: string; authorization: string | null }[] = [];
  const position: Record<string, number> = {};
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init: RequestInit = {}) => {
      const headers = (init.headers ?? {}) as Record<string, string>;
      calls.push({ path, authorization: headers.Authorization ?? null });
      const answers = script[path];
      if (!answers) throw new Error(`unexpected request to ${path}`);
      const index = Math.min(position[path] ?? 0, answers.length - 1);
      position[path] = index + 1;
      const answer = answers[index];
      if (answer === "network-error") throw new TypeError("Failed to fetch");
      return json(answer.status, answer.body);
    }),
  );
  return calls;
}

const session = (token: string) => ({ status: 200, body: { access_token: token, expires_in: 1800, user: { id: "u1" } } });
const expired = { status: 401, body: { error: { code: "unauthorized", message: "Your session has expired. Please sign in again." } } };

describe("an access token that is no longer accepted", () => {
  let signedOut: ReturnType<typeof vi.fn<() => void>>;

  beforeEach(() => {
    signedOut = vi.fn<() => void>();
    onSessionEnded(signedOut);
    setAccessToken("old-token");
    setRefreshRetryMs(0);
  });

  afterEach(() => {
    onSessionEnded(null);
    setAccessToken(null);
    vi.unstubAllGlobals();
  });

  it("is replaced, and the request is repeated with the new one", async () => {
    const calls = server({
      "/documents": [expired, { status: 200, body: { documents: [] } }],
      "/auth/refresh": [session("new-token")],
    });
    expect(await api.documents()).toEqual([]);
    expect(calls.map((call) => [call.path, call.authorization])).toEqual([
      ["/documents", "Bearer old-token"],
      ["/auth/refresh", null],
      ["/documents", "Bearer new-token"],
    ]);
    expect(signedOut).not.toHaveBeenCalled();
  });

  it("is replaced for the account requests too (found in review: change password failed after 30 idle minutes)", async () => {
    const calls = server({
      "/auth/password": [expired, session("after-change")],
      "/auth/me": [expired, { status: 204 }],
      "/auth/refresh": [session("new-token")],
    });
    await api.changePassword("old password", "a new password");
    await api.deleteAccount();
    expect(calls.filter((call) => call.path === "/auth/refresh")).toHaveLength(2);
    expect(signedOut).not.toHaveBeenCalled();
  });

  it("is not 'replaced' for a wrong password at sign-in", async () => {
    const calls = server({ "/auth/login": [{ status: 401, body: { error: { code: "unauthorized", message: "Email or password is incorrect." } } }] });
    await expect(api.login("a@b.c", "wrong")).rejects.toThrow("Email or password is incorrect.");
    expect(calls.map((call) => call.path)).toEqual(["/auth/login"]);
    expect(signedOut).not.toHaveBeenCalled();
  });

  it("ends the session when the server says the sign-in is over", async () => {
    const calls = server({ "/documents": [expired], "/auth/refresh": [{ status: 401 }] });
    await expect(api.documents()).rejects.toThrow("Your session has expired");
    expect(calls.filter((call) => call.path === "/auth/refresh")).toHaveLength(2); // asked twice, see the next test
    expect(signedOut).toHaveBeenCalledTimes(1);
  });

  it("asks once more when another tab used the same cookie a moment ago", async () => {
    server({
      "/documents": [expired, { status: 200, body: { documents: [] } }],
      "/auth/refresh": [{ status: 401 }, session("from-the-new-cookie")],
    });
    expect(await api.documents()).toEqual([]);
    expect(signedOut).not.toHaveBeenCalled();
  });

  it("does not sign anyone out because the server is restarting (found in review)", async () => {
    for (const answer of [{ status: 502 }, { status: 503 }, "network-error"] as Answer[]) {
      server({ "/documents": [expired], "/auth/refresh": [answer] });
      const failure = await api.documents().catch((error: unknown) => error);
      expect(failure).toBeInstanceOf(ApiError);
      expect((failure as ApiError).message).toMatch(/Try again in a moment/);
    }
    expect(signedOut).not.toHaveBeenCalled();
  });

  it("tells a sleeping server from 'not signed in' when the page loads", async () => {
    server({ "/auth/refresh": [{ status: 401 }] });
    expect(await api.restore()).toBeNull(); // nobody is signed in: show the sign-in page
    server({ "/auth/refresh": [{ status: 503 }] });
    await expect(api.restore()).rejects.toThrow(/not available right now/); // show "try again", keep the cookie
    server({ "/auth/refresh": [session("restored")] });
    expect((await api.restore())?.access_token).toBe("restored");
  });

  it("lets several requests share one refresh", async () => {
    const calls = server({
      "/documents": [expired, { status: 200, body: { documents: [] } }],
      "/investigations": [expired, { status: 200, body: { investigations: [] } }],
      "/auth/refresh": [session("new-token")],
    });
    await Promise.all([api.documents(), api.investigations()]);
    expect(calls.filter((call) => call.path === "/auth/refresh")).toHaveLength(1);
  });
});
