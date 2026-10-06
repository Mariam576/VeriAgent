"""Rate limiting: how often one client may do one thing.

Concept - sliding window:
    For each key (an IP address, an email address) the times of the recent
    attempts are remembered. An attempt is allowed while fewer than ``limit``
    of them fall inside the last ``window_seconds``.

The counters live in this process's memory. That is right for one API
process, which is how VeriAgent runs. With several processes each would count
separately; a shared store (Redis) would be the next step.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request

from app.config import get_settings
from app.errors import TooManyRequests

MAX_TRACKED_KEYS = 50_000


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit, self.window = limit, window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _recent(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        return hits

    def allowed(self, key: str) -> bool:
        """Would another attempt be allowed right now? Does not count as an attempt."""
        return len(self._recent(key, time.monotonic())) < self.limit

    def hit(self, key: str) -> bool:
        """Record an attempt. Returns False when it is over the limit."""
        now = time.monotonic()
        hits = self._recent(key, now)
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        if len(self._hits) > MAX_TRACKED_KEYS:  # do not let the table grow without bound
            self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v and v[-1] > now - self.window})
        return True

    def retry_after(self, key: str) -> int:
        hits = self._recent(key, time.monotonic())
        return max(1, int(hits[0] + self.window - time.monotonic()) + 1) if hits else 1

    def take_back(self, key: str) -> None:
        """Remove the latest attempt again (it was counted before its outcome was known, and it succeeded)."""
        hits = self._hits.get(key)
        if hits:
            hits.pop()

    def forget(self, key: str) -> None:
        self._hits.pop(key, None)

    def reset(self) -> None:
        self._hits.clear()


def client_address(request: Request) -> str:
    """Who is calling, for the per-address limits.

    Directly connected, that is the address of the connection. Behind a hosting
    platform's proxy every connection comes from the proxy, and the real address
    is in a header the proxy sets. Which header can be trusted depends on the
    proxy: one that the proxy always overwrites (``CLIENT_ADDRESS_HEADER``) is
    safe; ``X-Forwarded-For`` is a list that many proxies only add to, so its
    first entry may have been written by the client itself. That is why the
    limit that protects passwords counts per account, not per address
    (``failed_logins`` in app/api/auth.py).
    """
    settings = get_settings()
    if settings.TRUST_PROXY_HEADERS:
        for name in dict.fromkeys((settings.CLIENT_ADDRESS_HEADER.strip().lower(), "x-forwarded-for")):
            value = request.headers.get(name, "") if name else ""
            first = value.split(",")[0].strip()
            if first:
                return first[:64]
    return request.client.host if request.client else "unknown"


def check(limiter: RateLimiter, key: str, what: str) -> None:
    """Count an attempt; raise 429 with a waiting time when the limit is reached."""
    if not limiter.hit(key):
        raise TooManyRequests(f"Too many {what}. Try again in {limiter.retry_after(key)} seconds.")
