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

    def forget(self, key: str) -> None:
        self._hits.pop(key, None)

    def reset(self) -> None:
        self._hits.clear()


def client_address(request: Request) -> str:
    """Who is calling. Behind a trusted proxy the real address is in X-Forwarded-For."""
    if get_settings().TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


def check(limiter: RateLimiter, key: str, what: str) -> None:
    """Count an attempt; raise 429 with a waiting time when the limit is reached."""
    if not limiter.hit(key):
        raise TooManyRequests(f"Too many {what}. Try again in {limiter.retry_after(key)} seconds.")
