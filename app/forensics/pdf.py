"""What a PDF shows to a reader, and what it hides.

Concept - a PDF page is a list of drawing commands:
    "write this text here", "fill this rectangle", "place this image". They
    are carried out one after the other, and later commands paint over
    earlier ones. Editing an amount with a PDF editor typically leaves traces
    in exactly this list: the old text is still there, a white rectangle is
    drawn over it, and the new text is written on top, often in another font.

    Text extraction reads the commands, not the picture. It returns the old
    amount AND the new one. So a program that only extracts text sees things
    a person never sees: covered text, white text on a white page, text in
    "invisible ink" (render mode 3), text one point high. That is also the
    easiest place to hide instructions for an AI system.

This module does two things with that knowledge:

1. ``visible_text`` returns only what a reader can see. The whole application
   reads documents through it, so hidden text never becomes evidence.
2. ``inspect_pdf`` reports what was hidden and other signs of editing.

How "hidden" is decided (PyMuPDF gives every text span its sequence number in
the command list, its colour and its render mode):

    invisible     render mode 3 or opacity 0 - unless it lies on an image, which is how
                  scanned pages carry their recognised text (an OCR layer) and is normal
    same colour   the text colour is (nearly) the colour behind it
    tiny          smaller than 2 points
    covered       a filled rectangle or an image drawn LATER covers it
    off page      outside the page's visible area
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.forensics.model import Indicator, indicator
from app.security.guardrails import detect_prompt_injection

logger = logging.getLogger(__name__)

MIN_VISIBLE_SIZE = 2.0  # points; smaller text cannot be read on screen or paper
COVER_SHARE = 0.8  # share of a character's box that must be covered to call it covered
MIN_CONTRAST = 0.08  # difference in brightness (0..1) below which text cannot be told from its background
SCAN_IMAGE_SHARE = 0.7  # an image this large on a page is treated as a scan of the page
MAX_TYPED_CHARS_ON_SCAN = 150
RARE_FONT_SHARE = 0.03
MIN_OVERLAPPING_CHARS = 3
OVERLAP_SHARE = 0.4  # share of the smaller character's box that another character must cover
MAX_PAGES_INSPECTED = 200

HIDDEN_WORDS = {
    "invisible": "written in invisible ink",
    "same_colour": "in the same colour as its background",
    "tiny": f"smaller than {MIN_VISIBLE_SIZE:g} points",
    "covered": "covered by a shape or image drawn over it",
    "off_page": "placed outside the visible page",
}

# Programs whose main purpose is to change existing PDFs. A file that names one was edited
# after it was made; that is common and innocent, so it is only a weak indicator.
_EDITING_TOOLS = (
    "ilovepdf", "sejda", "pdfescape", "smallpdf", "pdf-xchange", "pdfxchange", "phantompdf", "foxit pdf editor",
    "nitro pro", "pdfelement", "pdffiller", "dochub", "soda pdf", "pdf24", "pdfsam", "inkscape", "gimp",
    "photoshop", "illustrator", "master pdf editor", "xodo", "pdf expert", "pdf candy",
)
_STYLE_WORDS = re.compile(
    r"(?:bold|italic|oblique|regular|medium|light|semibold|demibold|black|heavy|book|roman|narrow|condensed|mt|ps|it|bd|bi)$"
)
_PDF_DATE = re.compile(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(?:([+\-Zz])(\d{2})?'?(\d{2})?)?")


# ------------------------------------------------------------------ tracing
@dataclass
class TracedChar:
    char: str
    bbox: tuple[float, float, float, float]
    origin: tuple[float, float]  # where the character starts on its baseline
    seqno: int  # position of its text command in the page's list of drawing commands
    font: str
    size: float
    hidden: str | None  # None = a reader can see it; otherwise one of the keys of HIDDEN_WORDS
    ocr: bool = False  # invisible ink on top of an image: the recognised text of a scanned page


@dataclass
class PageTrace:
    number: int
    chars: list[TracedChar] = field(default_factory=list)
    images: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)
    area: float = 0.0
    failed: bool = False

    @property
    def has_hidden(self) -> bool:
        return any(char.hidden for char in self.chars)


def _luminance(colour: Any, colourspace: int | None = None) -> float:
    """Brightness between 0 (black) and 1 (white) of a grey, RGB or CMYK colour."""
    if colour is None:
        return 0.0
    values = [float(v) for v in colour] if isinstance(colour, (tuple, list)) else [float(colour)]
    size = colourspace or len(values)
    if size == 1 or len(values) == 1:
        return values[0]
    if size == 4 and len(values) >= 4:
        c, m, y, k = values[:4]
        values = [(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)]
    r, g, b = values[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _area(box: tuple[float, float, float, float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _contains_point(box: tuple[float, float, float, float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def trace_page(page: Any, number: int) -> PageTrace:
    """Every character on the page with the verdict "visible" or the reason it is hidden."""
    trace = PageTrace(number=number, area=_area(tuple(page.rect)))
    try:
        log = page.get_bboxlog()
        spans = page.get_texttrace()
        drawings = page.get_drawings() if any(kind == "fill-path" for kind, _ in log) else []
    except Exception as exc:  # noqa: BLE001 - an unusual PDF must not stop the upload
        logger.warning("page %d could not be traced: %s: %s", number, type(exc).__name__, exc)
        trace.failed = True
        return trace

    visible_area = tuple(page.rect)
    trace.images = [(index, tuple(box)) for index, (kind, box) in enumerate(log) if kind == "fill-image"]
    # Opaque filled rectangles: (sequence number, box, brightness).
    fills = []
    for drawing in drawings:
        opacity = drawing.get("fill_opacity")
        if drawing.get("fill") is None or (opacity is not None and opacity < 0.9):
            continue
        if not drawing.get("items") or any(item[0] not in ("re", "qu") for item in drawing["items"]):
            continue  # only plain rectangles: the bounding box of a curve says little about what it covers
        fills.append((int(drawing.get("seqno", -1)), tuple(drawing["rect"]), _luminance(drawing["fill"])))

    for span in spans:
        seqno, size = int(span.get("seqno", 0)), float(span.get("size", 0.0))
        box = tuple(span["bbox"])
        font = str(span.get("font", ""))
        unseen = int(span.get("type", 0)) == 3 or float(span.get("opacity", 1.0)) < 0.05
        brightness = _luminance(span.get("color"), span.get("colorspace"))
        later = [(fill_box) for fill_seq, fill_box, _ in fills if fill_seq > seqno and _overlap(fill_box, box) > 0]
        later += [image_box for image_seq, image_box in trace.images if image_seq > seqno and _overlap(image_box, box) > 0]
        behind = [(fill_seq, fill_box, fill_light) for fill_seq, fill_box, fill_light in fills if fill_seq < seqno and _overlap(fill_box, box) > 0]
        images_behind = [image_box for image_seq, image_box in trace.images if image_seq < seqno and _overlap(image_box, box) > 0]

        for code, _glyph, origin, char_box in span["chars"]:
            char = chr(code)
            char_box = tuple(char_box)
            origin = (float(origin[0]), float(origin[1]))
            if char.isspace():
                trace.chars.append(TracedChar(char, char_box, origin, seqno, font, size, None))
                continue
            centre_x, centre_y = (char_box[0] + char_box[2]) / 2, (char_box[1] + char_box[3]) / 2
            on_image = any(_contains_point(image_box, centre_x, centre_y) for image_box in images_behind)
            hidden, ocr = None, False
            if _overlap(char_box, visible_area) <= 0:
                hidden = "off_page"
            elif size < MIN_VISIBLE_SIZE:
                hidden = "tiny"
            elif unseen:
                ocr = on_image or any(_contains_point(b, centre_x, centre_y) for _, b in trace.images)
                hidden = None if ocr else "invisible"
            elif any(_overlap(cover, char_box) >= COVER_SHARE * max(_area(char_box), 1e-6) for cover in later):
                hidden = "covered"
            elif not on_image:  # the colours of an image are unknown here, so text on one counts as visible
                under = [light for _, fill_box, light in sorted(behind, key=lambda item: item[0]) if _contains_point(fill_box, centre_x, centre_y)]
                background = under[-1] if under else 1.0  # the topmost rectangle beneath, else white paper
                if abs(brightness - background) < MIN_CONTRAST:
                    hidden = "same_colour"
            trace.chars.append(TracedChar(char, char_box, origin, seqno, font, size, hidden, ocr))
    return trace


def _key(char: str, origin: Any) -> tuple[str, float, float]:
    return char, round(float(origin[0]), 1), round(float(origin[1]), 1)


def _visible_lines(page: Any, trace: PageTrace) -> list[list[tuple[str, float, int, str]]] | None:
    """The page's lines, each as spans of (font family, size, flags, visible text). None if it cannot be read.

    PyMuPDF offers two views of a page: the text as lines and spans (what we
    want to return) and the trace of drawing commands (which knows what is
    hidden). They are joined through each character's identity: the character
    itself and where it starts. Two characters can share that identity when new
    text is printed over old text, so each identity holds its verdicts in
    drawing order.
    """
    import pymupdf

    verdicts: dict[tuple[str, float, float], list[bool]] = defaultdict(list)
    for char in trace.chars:
        verdicts[_key(char.char, char.origin)].append(char.hidden is not None)
    try:
        raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("page %d: text structure could not be read: %s", trace.number, exc)
        return None

    lines: list[tuple[tuple[float, float, float, float], list[tuple[str, float, int, str]]]] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            spans, total, seen = [], 0, []
            for span in line.get("spans", []):
                kept = []
                for char in span.get("chars", []):
                    total += 1
                    queue = verdicts.get(_key(char["c"], char["origin"]))
                    if not (queue.pop(0) if queue else False):
                        kept.append(char["c"])
                        if not char["c"].isspace():
                            seen.append(char["bbox"])
                spans.append((font_family(span.get("font", "")), float(span.get("size", 0.0)), int(span.get("flags", 0)), "".join(kept)))
            if total and not seen:
                continue  # the whole line is hidden
            # The part of the line a reader sees (a hidden amount at its end no longer counts).
            box = (min(b[0] for b in seen), min(b[1] for b in seen), max(b[2] for b in seen), max(b[3] for b in seen)) if seen else (0.0, 0.0, 0.0, 0.0)
            # Text added later is a line of its own in the file, even when it is printed on the
            # same line as older text ("Total amount due:" + a new amount). Join what a reader
            # sees as one line: same height on the page, side by side.
            partner = next(
                (index for index, (other, _) in enumerate(lines)
                 if abs((other[1] + other[3]) - (box[1] + box[3])) / 2 < 0.3 * min(other[3] - other[1], box[3] - box[1])
                 and box[0] >= other[2] - 2),
                None,
            )
            if partner is not None:
                other, other_spans = lines[partner]
                left = "".join(span[3] for span in other_spans)
                if left and not left[-1].isspace() and box[0] - other[2] > 1:
                    other_spans.append((other_spans[-1][0], other_spans[-1][1], other_spans[-1][2], " "))
                other_spans.extend(spans)
                lines[partner] = ((other[0], min(other[1], box[1]), box[2], max(other[3], box[3])), other_spans)
            else:
                lines.append((box, spans))
    return [spans for _, spans in lines]


def visible_text(page: Any, number: int = 1, trace: PageTrace | None = None) -> str:
    """The text of a page as a reader sees it: everything except hidden characters.

    When nothing is hidden (nearly every document) this is exactly PyMuPDF's
    normal text extraction.
    """
    import pymupdf

    plain = page.get_text("text", flags=pymupdf.TEXTFLAGS_TEXT)
    trace = trace or trace_page(page, number)
    if trace.failed or not trace.has_hidden:
        return plain
    lines = _visible_lines(page, trace)
    if lines is None:
        return plain
    return "".join("".join(text for _, _, _, text in spans) + "\n" for spans in lines)


# --------------------------------------------------------------- indicators
def _hidden_indicators(trace: PageTrace) -> list[Indicator]:
    found: list[Indicator] = []
    by_reason: dict[str, list[tuple[int, str]]] = defaultdict(list)
    # Runs of hidden characters, kept apart when visible text stands between them, and with their
    # spaces, so a snippet reads "Total amount due" and two covered amounts stay two amounts.
    runs: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    previous: dict[int, str | None] = {}
    run_number: dict[int, int] = defaultdict(int)
    for char in trace.chars:
        if char.char.isspace():
            last = previous.get(char.seqno)
            if last:
                runs[(last, char.seqno, run_number[char.seqno])].append(" ")
            continue
        if char.hidden is None and previous.get(char.seqno):
            run_number[char.seqno] += 1  # visible text interrupts the hidden run
        previous[char.seqno] = char.hidden
        if char.hidden:
            runs[(char.hidden, char.seqno, run_number[char.seqno])].append(char.char)
    for (reason, seqno, _), pieces in runs.items():
        text = " ".join("".join(pieces).split())
        if text:
            by_reason[reason].append((seqno, text))

    for reason, every in by_reason.items():
        words = HIDDEN_WORDS[reason]
        aimed_at_ai = [(seqno, text) for seqno, text in every if detect_prompt_injection(text)]
        if aimed_at_ai:
            found.append(indicator(
                "hidden_instruction",
                f"Page {trace.number} contains text that a reader cannot see ({words}) and that reads like instructions "
                "to an AI system. It was not read as part of the document.",
                page=trace.number, region=" ".join(text for _, text in aimed_at_ai),
            ))
        items = [item for item in every if item not in aimed_at_ai]
        if not items:
            continue
        text = (" | " if reason == "covered" else " ").join(text for _, text in items)
        if reason == "covered":
            # Is something else printed where the covered text was? Then the text was replaced, not just removed.
            covered_boxes = [char.bbox for char in trace.chars if char.hidden == "covered"]
            covered_from = min(seqno for seqno, _ in items)
            on_top = [
                char for char in trace.chars
                if char.hidden is None and not char.char.isspace() and char.seqno > covered_from
                and any(_overlap(char.bbox, box) > 0.3 * max(_area(char.bbox), 1e-6) for box in covered_boxes)
            ]
            if on_top:
                replacement = " | ".join(
                    " ".join("".join(char.char for char in trace.chars if char.hidden is None and char.seqno == seqno).split())
                    for seqno in dict.fromkeys(char.seqno for char in on_top)
                )
                found.append(indicator(
                    "replaced_text",
                    f"On page {trace.number} text is covered by a shape and other text is printed on top of it "
                    f"(now showing: {replacement[:60]}). The covered text was not read as part of the document.",
                    page=trace.number, region=text,
                ))
                continue
            found.append(indicator(
                "covered_text",
                f"On page {trace.number} text is {words}. It is still in the file but was not read as part of the document.",
                page=trace.number, region=text,
            ))
        elif reason == "off_page":
            found.append(indicator(
                "off_page_text", f"Page {trace.number} has text {words}.", page=trace.number, region=text,
            ))
        else:
            found.append(indicator(
                "hidden_text",
                f"Page {trace.number} contains text that a reader cannot see ({words}). It was not read as part of the document.",
                page=trace.number, region=text,
            ))
    return found


def _overlap_indicator(trace: PageTrace) -> Indicator | None:
    """Different visible characters printed on the same spot (text typed over text without covering it)."""
    visible = [char for char in trace.chars if char.hidden is None and not char.ocr and not char.char.isspace()]
    grid: dict[tuple[int, int], list[TracedChar]] = defaultdict(list)
    for char in visible:
        grid[(int((char.bbox[0] + char.bbox[2]) / 16), int((char.bbox[1] + char.bbox[3]) / 16))].append(char)
    clashing: list[TracedChar] = []
    for (cell_x, cell_y), chars in grid.items():
        neighbours = [other for dx in (0, 1) for dy in (-1, 0, 1) for other in grid.get((cell_x + dx, cell_y + dy), []) if (dx, dy) != (0, -1)]
        for char in chars:
            for other in neighbours:
                if other is char or other.seqno == char.seqno or other.char == char.char:
                    continue  # the same letter twice in one place is how some programs draw bold text
                if _overlap(char.bbox, other.bbox) > OVERLAP_SHARE * min(_area(char.bbox), _area(other.bbox)):
                    clashing.append(other if other.seqno > char.seqno else char)
    if len({(char.bbox, char.char) for char in clashing}) < MIN_OVERLAPPING_CHARS:
        return None
    text = "".join(dict.fromkeys(f"{char.char}" for char in sorted(clashing, key=lambda c: (round(c.bbox[1]), c.bbox[0]))))
    return indicator(
        "overlapping_text",
        f"On page {trace.number} different characters are printed on top of each other, as when a value is typed over an old one.",
        page=trace.number, region=text,
    )


def _scan_indicator(trace: PageTrace) -> Indicator | None:
    scans = [(seqno, box) for seqno, box in trace.images if trace.area and _area(box) >= SCAN_IMAGE_SHARE * trace.area]
    if not scans:
        return None
    first_scan = min(seqno for seqno, _ in scans)
    typed = [char for char in trace.chars if char.hidden is None and not char.ocr and not char.char.isspace() and char.seqno > first_scan]
    if not 0 < len(typed) <= MAX_TYPED_CHARS_ON_SCAN:
        return None
    typed_commands = {char.seqno for char in typed}
    text = " ".join("".join(char.char for char in trace.chars if char.hidden is None and char.seqno in typed_commands).split())
    return indicator(
        "text_over_scan",
        f"Page {trace.number} is a scanned image with a few typed words placed on top of it.",
        page=trace.number, region=text,
    )


def font_family(name: str) -> str:
    """'ABCDEF+Helvetica-BoldOblique' -> 'helvetica'. Bold and italic are the same family."""
    base = re.sub(r"^[A-Z]{6}\+", "", name or "").lower()
    base = re.sub(r"[^a-z0-9]+", "", base.replace(",", "-").split("-")[0]) or re.sub(r"[^a-z0-9]+", "", base)
    previous = None
    while base != previous:
        previous = base
        base = _STYLE_WORDS.sub("", base) or base
    return base


def _font_indicators(lines: list[tuple[int, list[tuple[str, float, int, str]]]]) -> list[Indicator]:
    """A value with digits set in a font the rest of the document hardly uses, in the middle of a line.

    ``lines``: (page, [(font family, size, flags, text)]) for every line of visible text.
    """
    share: Counter[str] = Counter()
    span_count: Counter[str] = Counter()
    for _, spans in lines:
        for family, _, _, text in spans:
            share[family] += len(text.strip())
            span_count[family] += 1
    total = sum(share.values()) or 1

    found: list[Indicator] = []
    for page, spans in lines:
        spans = [span for span in spans if span[3].strip()]
        if len(spans) < 2:
            continue
        for index, (family, size, flags, text) in enumerate(spans):
            if not re.search(r"\d", text) or len(text.strip()) < 2 or flags & 1:  # flag bit 0: superscript
                continue
            others = spans[:index] + spans[index + 1:]
            usual = max(others, key=lambda span: len(span[3]))
            line_text = " ".join("".join(span[3] for span in spans).split())
            rare = share[family] / total < RARE_FONT_SHARE and span_count[family] <= 3
            if family != usual[0] and rare:
                found.append(indicator(
                    "font_mismatch",
                    f"On page {page} the value '{text.strip()[:40]}' is set in a font ({family or 'unknown'}) that the "
                    f"rest of its line ({usual[0] or 'unknown'}) and of the document does not use.",
                    page=page, region=line_text,
                ))
            elif family == usual[0] and 0.5 <= abs(size - usual[1]) <= 3 and rare_size(size, lines):
                found.append(indicator(
                    "font_mismatch",
                    f"On page {page} the value '{text.strip()[:40]}' is {size:g} pt while the rest of its line is {usual[1]:g} pt.",
                    page=page, region=line_text,
                ))
    return found[:4]


def rare_size(size: float, lines: list[tuple[int, list[tuple[str, float, int, str]]]]) -> bool:
    """Is this font size used for hardly anything else in the document?"""
    sizes: Counter[float] = Counter()
    for _, spans in lines:
        for _, span_size, _, text in spans:
            sizes[round(span_size, 1)] += len(text.strip())
    return sizes[round(size, 1)] / (sum(sizes.values()) or 1) < RARE_FONT_SHARE


# ----------------------------------------------------------------- metadata
def parse_pdf_date(value: str | None) -> datetime | None:
    """"D:20260303142500+01'00'" -> an aware datetime. None when the field is empty or unreadable."""
    match = _PDF_DATE.match((value or "").strip())
    if not match:
        return None
    year, month, day, hour, minute, second, sign, offset_hours, offset_minutes = match.groups()
    try:
        offset = timedelta(hours=int(offset_hours or 0), minutes=int(offset_minutes or 0))
        zone = timezone(-offset if sign == "-" else offset)
        return datetime(int(year), int(month or 1), int(day or 1), int(hour or 0), int(minute or 0), int(second or 0), tzinfo=zone)
    except ValueError:
        return None


def _history_indicators(document: Any, data: bytes, now: datetime) -> tuple[list[Indicator], dict[str, Any]]:
    meta = document.metadata or {}
    created, modified = parse_pdf_date(meta.get("creationDate")), parse_pdf_date(meta.get("modDate"))
    shown = {
        key: str(meta.get(key))[:200] for key in ("title", "author", "creator", "producer") if meta.get(key)
    }
    if created:
        shown["created"] = created.isoformat()
    if modified:
        shown["modified"] = modified.isoformat()
    shown["pages"] = document.page_count

    found: list[Indicator] = []
    tools = f"{meta.get('producer') or ''} {meta.get('creator') or ''}".lower()
    tool = next((name for name in _EDITING_TOOLS if name in tools), None)
    if tool:
        found.append(indicator(
            "editing_tool",
            f"The file was last written by a PDF editing program ({(meta.get('producer') or meta.get('creator') or tool)[:60]}).",
        ))
    if created and modified:
        if modified < created - timedelta(minutes=1):
            found.append(indicator("metadata_inconsistent", "The file says it was last changed before it was created."))
        elif modified - created > timedelta(minutes=2):
            found.append(indicator(
                "modified_after_creation",
                f"The file was changed after it was created (created {created.day} {created:%B %Y}, "
                f"last changed {modified.day} {modified:%B %Y}).",
            ))
    if created and created > now + timedelta(days=2):
        found.append(indicator("metadata_inconsistent", f"The file's creation date ({created.day} {created:%B %Y}) lies in the future."))

    # Each save "in place" appends a new section that ends with %%EOF. A file written once has one
    # (two when it is "linearized" for fast web view).
    saves = data.count(b"%%EOF")
    expected = 2 if getattr(document, "is_fast_webaccess", False) else 1
    shown["saves"] = saves
    if saves > expected:
        found.append(indicator(
            "incremental_saves",
            f"The file was saved {saves - expected} more time(s) after it was first written; earlier versions are still inside it.",
        ))
    return found, shown


def inspect_pdf(data: bytes, now: datetime | None = None) -> tuple[list[Indicator], dict[str, Any]]:
    """All indicators for one PDF file, and its metadata."""
    import pymupdf

    document = pymupdf.open(stream=data, filetype="pdf")
    try:
        found, metadata = _history_indicators(document, data, now or datetime.now(timezone.utc))
        lines: list[tuple[int, list[tuple[str, float, int, str]]]] = []
        for number, page in enumerate(document, start=1):
            if number > MAX_PAGES_INSPECTED:
                break
            trace = trace_page(page, number)
            if trace.failed:
                continue
            found += _hidden_indicators(trace)
            for extra in (_overlap_indicator(trace), _scan_indicator(trace)):
                if extra is not None:
                    found.append(extra)
            lines += [(number, spans) for spans in _visible_lines(page, trace) or []]
        found += _font_indicators(lines)
        return found, metadata
    finally:
        document.close()
