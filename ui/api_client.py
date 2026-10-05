"""A small HTTP client for the VeriAgent API.

The interface never imports the agent or touches the database. Everything it
shows comes from the same HTTP endpoints you can try at /docs, so the UI is
one client among many: a React frontend (Version 3) would call the same API.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class ApiError(Exception):
    """An error the user should see, already worded for them."""


def _message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"The API answered {response.status_code}."
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        detail = body.get("detail")
        if isinstance(detail, list) and detail:  # FastAPI validation errors
            return "; ".join(str(item.get("msg", item)) for item in detail)
        if detail:
            return str(detail)
    return f"The API answered {response.status_code}."


class ApiClient:
    def __init__(self, base_url: str, user: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-User-Id": user},
            # An investigation makes several model calls; connecting must still fail fast.
            timeout=httpx.Timeout(600.0, connect=5.0),
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url} ({type(exc).__name__}). Is it running?") from exc
        if response.status_code >= 400:
            raise ApiError(_message(response))
        return response.json() if response.content else None

    # ------------------------------------------------------------- basics
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    # ---------------------------------------------------------- documents
    def documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents")["documents"]

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        return self._request("POST", "/documents/upload", files={"file": (filename, data)})

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    # ----------------------------------------------------- investigations
    def investigations(self) -> list[dict[str, Any]]:
        return self._request("GET", "/investigations")["investigations"]

    def investigation(self, investigation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/investigations/{investigation_id}")

    def trace(self, investigation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/investigations/{investigation_id}/trace")

    def decide(self, investigation_id: str, decision: str, note: str | None, modified_action: str | None) -> dict[str, Any]:
        payload = {"decision": decision, "note": note or None, "modified_action": modified_action or None}
        return self._request("POST", f"/investigations/{investigation_id}/approve", json=payload)

    def resume(self, investigation_id: str) -> dict[str, Any]:
        """Continue an investigation that was cut off. Returns its report."""
        return self._request("POST", f"/investigations/{investigation_id}/resume")

    def chat_stream(self, message: str, conversation_id: str | None) -> Iterator[tuple[str, dict[str, Any]]]:
        """Send one chat turn and yield (event name, data) pairs as the server sends them.

        Server-Sent Events are lines of text: "event: <name>", then "data: <json>",
        then an empty line. Lines starting with ":" are keep-alive comments.
        """
        payload = {"message": message, "conversation_id": conversation_id}
        try:
            with self._client.stream("POST", "/chat/stream", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(_message(response))
                event = ""
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        event = line[len("event: "):]
                    elif line.startswith("data: "):
                        yield event, json.loads(line[len("data: "):])
        except httpx.HTTPError as exc:
            raise ApiError(f"The connection to the API was lost ({type(exc).__name__}).") from exc

    # -------------------------------------------------------------- memory
    def memory(self) -> dict[str, Any]:
        return self._request("GET", "/memory")

    def set_preference(self, key: str, value: str) -> None:
        self._request("PUT", f"/memory/preferences/{key}", json={"value": value})

    def add_note(self, text: str) -> None:
        self._request("POST", "/memory/notes", json={"text": text})

    def delete_note(self, note_id: str) -> None:
        self._request("DELETE", f"/memory/notes/{note_id}")
