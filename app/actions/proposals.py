"""What the agent proposes: the text of the follow-up issue.

The text is assembled by code from the structured report, the same way the
report itself is. No model writes it. Two reasons:

* the person who approves must see exactly what will be sent, and
* text taken from documents ends up in another system. It is escaped, so a
  document cannot plant a link, an image or an "@mention" in your tracker.
"""

from __future__ import annotations

from typing import Any

from app.formatting import escape_markdown
from app.rag.citations import format_citation
from app.schemas.response import InvestigationReport

MAX_TITLE_CHARS = 120
MAX_BODY_CHARS = 6000
MAX_LISTED = 8
# An action is proposed for findings that need a person anyway (spec section 48: "a high-risk issue").
PROPOSE_FROM_RISK = ("high", "critical")


def _safe(text: Any) -> str:
    # "@name" would notify a GitHub user: an invisible break after the @ prevents that.
    return escape_markdown(text).replace("@", "@​")


def _headline(report: InvestigationReport) -> str:
    if report.contradictions:
        return report.contradictions[0].description
    if report.document_integrity:
        first = report.document_integrity[0]
        return f"{first.document} shows signs that it may have been altered"
    if report.missing_evidence:
        return f"Missing evidence: {report.missing_evidence[0]}"
    return report.summary


def issue_title(report: InvestigationReport) -> str:
    title = " ".join(f"VeriAgent: {report.risk_level} risk - {_headline(report)}".split())
    return title if len(title) <= MAX_TITLE_CHARS else title[: MAX_TITLE_CHARS - 3].rstrip() + "..."


def issue_body(report: InvestigationReport) -> str:
    lines = [
        f"**Question:** {_safe(report.question)}",
        f"**Risk level:** {report.risk_level.capitalize()} | **Confidence:** {round(report.confidence * 100)}% | "
        f"**Status:** {_safe(report.status_label)}",
        "",
        "### Summary",
        _safe(report.summary),
    ]
    if report.contradictions:
        lines += ["", "### Contradictions"]
        for item in report.contradictions[:MAX_LISTED]:
            cites = "".join(" " + _safe(format_citation(c.document, c.page)) for c in item.citations)
            lines.append(f"- ({item.severity}) {_safe(item.description)}{cites}")
    if report.document_integrity:
        lines += ["", "### Document integrity (risk indicators, not proof)"]
        for item in report.document_integrity[:MAX_LISTED]:
            lines.append(f"- {_safe(item.document)}: {item.level}")
            lines += [f"  - {_safe(one.reason)}" for one in item.indicators[:3]]
    if report.missing_evidence:
        lines += ["", "### Missing evidence"]
        lines += [f"- {_safe(text)}" for text in report.missing_evidence[:MAX_LISTED]]
    lines += ["", "### Recommended action", _safe(report.recommended_action)]
    if report.sources:
        lines += ["", "### Sources"]
        lines += [f"- {_safe(f'{source.document}, page {source.page}')}" for source in report.sources[:MAX_LISTED]]
    lines += ["", f"_Created by VeriAgent after human approval. Investigation {_safe(report.investigation_id)}._"]
    body = "\n".join(lines)
    return body if len(body) <= MAX_BODY_CHARS else body[: MAX_BODY_CHARS - 20].rstrip() + "\n\n(shortened)"


def propose_actions(report: InvestigationReport, *, server: str, target: str) -> list[dict[str, Any]]:
    """The follow-up actions to offer with this report. At most one today: an issue for a high-risk result."""
    if report.status == "not_applicable" or report.risk_level not in PROPOSE_FROM_RISK:
        return []
    return [{
        "id": "A1",
        "kind": "create_issue",
        "server": server,
        "target": target,
        "title": issue_title(report),
        "body": issue_body(report),
        "severity": report.risk_level,
    }]
