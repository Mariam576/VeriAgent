"""What a PDF shows to a reader, and what it hides.

Concept - a PDF page is a list of drawing commands:
    "write this text here", "fill this shape", "place this image". They are
    carried out one after the other, and later commands paint over earlier
    ones. Editing an amount with a PDF editor typically leaves traces in
    exactly this list: the old text is still there, a white rectangle is drawn
    over it, and the new text is written on top, often in another font.

    Text extraction reads the commands, not the picture. It returns the old
    amount AND the new one. So a program that only extracts text sees things
    a person never sees: covered text, white text on a white page, text in
    "invisible ink" (render mode 3), text one point high. That is also the
    easiest place to hide instructions for an AI system.

This module does two things with that knowledge:

1. ``read_pdf`` returns, for every page, only the text a reader can see. The
   whole application reads documents through it, so hidden text never becomes
   evidence.
2. The same pass reports what was hidden, and other signs of editing.

Concept - ask the picture, not the commands:
    There are endless ways to hide text: white on white, under a rectangle,
    under an ellipse, under a thick white line, under an image, at 6% opacity,
    dark on a dark photo. Reasoning about the commands means writing one rule
    per trick, and each rule has honest look-alikes (white text on a dark
    rounded banner; text under a yellow highlighter; a transparent signature
    over a line of text).

    So the question is put to the picture itself. Each page is drawn twice:
    once as it is, and once with all its text taken out. Where the two
    pictures differ, text put ink on the page - a reader can see it. Where a
    character's place looks the same in both, that character changes nothing
    a reader could see, whatever the reason.

        visible          the pictures differ at the character
        covered          no difference, and something drawn later lies over it
        invisible        no difference, and the text is in "invisible ink"
        same colour      no difference (or hardly any) for any other reason
        tiny             smaller than 3 points: too small to read, whatever its colour
        off page         outside the page's visible area

    Two cases need more than the two pictures:

    * Scanned pages carry their recognised text (the OCR layer) in invisible
      ink, or under the page image. That is normal and the text is kept. It
      is only accepted on a page that really is a scan: one opaque image over
      (nearly) the whole page and hardly any typed text.
    * Where text is printed over other text - the classic altered amount - the
      picture cannot say which of the two a spot of ink belongs to. There the
      drawing order decides: text is hidden if an opaque rectangle or image
      was drawn over it later.

    Annotations (notes, stamps, typed boxes added in a PDF viewer) are drawn
    on top of the page, so they are part of the picture too: text under an
    opaque annotation is covered, text under a highlighter is not.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.forensics.model import Indicator, indicator
from app.security.guardrails import detect_prompt_injection

logger = logging.getLogger(__name__)

MIN_VISIBLE_SIZE = 3.0  # points; smaller text cannot be read on screen or paper (small print goes down to about 3.5)
RENDER_ZOOM = 2.0  # the page is drawn at 144 dots per inch for the comparison
MAX_RENDER_PIXELS = 8_000_000  # a poster-sized page is drawn smaller, not with more memory
# How much the two pictures must differ (0..255, strongest pixel of a character) ...
NO_INK = 12  # ... for the character to have put any ink on the page at all
FAINT_INK = 30  # ... for a run of text to be readable: below this it is about 12% away from its background
CORE_TOP, CORE_BOTTOM = 0.2, 0.1  # the part of a character's box that is looked at (not the gaps to the lines above and below)
CONTEST_SHARE = 0.25  # share of a character's core that another text command's character must cover to "contest" it
LARGE_TEXT = 24.0  # points; pale text this large is a watermark ("DRAFT"): not read, and not reported
OWN_INK_SHARE = 0.15  # how far a pixel must have moved towards a text's colour to count as that text's ink
MIN_SAME_SHAPES = 8  # so many different characters drawn as one and the same shape: the font does not draw what it says
SCAN_MARKS = 30  # how much a scan must vary under a piece of recognised text to count as "something is printed here"
COVER_SHARE = 0.8  # drawing-order rule: share of a character's box that must be covered to call it covered
MIN_CONTRAST = 0.08  # drawing-order rule: difference in brightness (0..1) below which text and background are one colour
SCAN_IMAGE_SHARE = 0.4  # pictures that together cover this much of a page make it a scan (a half-page document on A4 counts)
MIN_RENDER_ZOOM = 1.0  # a page so large that it could only be drawn smaller than this is not compared by picture
MAX_TYPED_CHARS_ON_SCAN = 150  # typed characters on a scan's picture beyond this are a typed page, not a note on a scan
MAX_TYPED_BODY_CHARS_ON_SCAN = 40  # typed characters next to the pictures beyond this: a typed page that has pictures
MIN_INK_ROWS = 0.4  # share of a character's height that must show ink (median of a run); honest text is at 0.6 and above
MIN_RUN_FOR_SHAPE_CHECKS = 4  # letters and digits a run needs before its ink is judged as a whole
MAX_LIGATURE_EXTRA = 2  # characters without width that may follow a drawn one ("ffi" is one shape and three characters)
MIN_STACKED = 4  # so many different characters printed on one spot are a blob, not text
MAX_MEASURED_PER_RUN = 12
PAGE_MARGIN_SHARE = 0.07  # the top and bottom strip of a page, where scanners and fax machines print their stamp
RARE_FONT_SHARE = 0.03
MIN_OVERLAPPING_CHARS = 3
OVERLAP_SHARE = 0.4  # share of the smaller character's box that another character must cover
MAX_PAGES_INSPECTED = 200  # for the checks of fonts and overlapping text; hidden text is looked for on every page
MAX_INDICATORS_COLLECTED = 60
# Characters that forms use to draw a blank ("Amount: ..........."). Text typed over them is a filled-in form.
FILLER_CHARS = frozenset("_.·•…-–—=")

HIDDEN_WORDS = {
    "invisible": "written in invisible ink",
    "same_colour": "in the same colour as its background",
    "tiny": f"smaller than {MIN_VISIBLE_SIZE:g} points",
    "covered": "covered by a shape or image drawn over it",
    "off_page": "placed outside the visible page",
    "not_drawn": "set in a font that draws other shapes than the characters it stands for",
    "slivers": "cut away or covered so that only thin slivers of it show",
    "stacked": "printed on one spot, every character on top of the last",
    "no_shape": "in the file's text, but not drawn on the page as characters",
    # The same character drawn twice at the same place, once in the colour of the paper: an outlined
    # heading (white inside, black outline). The outline is read; the inside is not a finding.
    "twin": "the inside of an outlined character",
    # The end of a value that is longer than its form field. Not shown, not read - and not a sign of anything.
    "overflow": "beyond the edge of its form field",
    # Pale text of watermark size ("DRAFT" across the page). It is left out of what is read,
    # like all text that can hardly be seen, but it is no sign of anything: nothing is reported.
    "pale_large": "a pale watermark",
    # Invisible text on a scan, at a place where the scan shows nothing. Not read. Recognised text does
    # stray onto blank paper now and then, so it is only reported when it reads like an instruction.
    "blank_scan": "invisible, at a place where the scanned page is blank",
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
_NOT_A_COVER = ("fill-text", "stroke-text", "clip-text", "clip-stroke-text", "ignore-text")
_SYMBOL_FONT = re.compile(r"dingbat|wingding|webding|symbol", re.I)

Box = tuple[float, float, float, float]


# ------------------------------------------------------------------ tracing
@dataclass(slots=True)
class TracedChar:
    char: str
    bbox: Box
    origin: tuple[float, float]  # where the character starts on its baseline
    seqno: int  # position of its text command in the page's list of drawing commands
    font: str
    size: float
    hidden: str | None  # None = a reader can see it; otherwise one of the keys of HIDDEN_WORDS
    ocr: bool = False  # the recognised text of a scanned page (invisible, or under the page image)
    annotation: bool = False  # part of a note, a stamp or a form field, not of the page itself
    opacity: float = 1.0


@dataclass
class PageTrace:
    number: int
    chars: list[TracedChar] = field(default_factory=list)
    images: list[tuple[int, Box]] = field(default_factory=list)  # every image: (sequence number, box)
    scans: list[tuple[int, Box]] = field(default_factory=list)  # opaque images that fill the page
    area: float = 0.0
    height: float = 0.0
    failed: bool = False  # the page could not be read at all: nothing is known about it
    compared: bool = True  # False: the two pictures could not be made; only the drawing order was used
    drawn_fonts: bool = False  # the page uses a font that draws its own shapes (Type 3)
    # Text that the page's text layer contains but that no drawing command was found for (text used
    # only as a clipping shape, text filled with a pattern, "replacement text" longer than what is
    # drawn). It cannot be judged, so it is not read. Filled in by _visible_lines.
    unjudged: list[str] = field(default_factory=list)

    @property
    def has_hidden(self) -> bool:
        return bool(self.unjudged) or any(char.hidden for char in self.chars)


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


def _area(box: Box) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def _overlap(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _contains_point(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _middle(box: Box) -> Box:
    """The narrow middle band of a character's box: lower-case height, clear of the lines above and below."""
    height = box[3] - box[1]
    return box[0], box[1] + 0.4 * height, box[2], box[3] - 0.27 * height


def _core(box: Box) -> Box:
    """The middle band of a character's box, where its ink is.

    The box a PDF gives for a character is as tall as the whole line, so the
    boxes of neighbouring lines touch or overlap. The band between them is
    what belongs to this character alone.
    """
    height = box[3] - box[1]
    return box[0], box[1] + CORE_TOP * height, box[2], box[3] - CORE_BOTTOM * height


# Characters whose ink lies outside the middle band: below the baseline, or up where the accents are.
_LOW_CHARS = frozenset("_")
_HIGH_CHARS = frozenset("`´'\"^~¨¯°*“”‘’")


def _band(char: str, box: Box) -> Box:
    """The part of a character's box in which its ink is expected.

    For nearly every character that is the middle band (``_core``). An
    underscore lies under it and a backtick above it; measured in the middle
    band they leave no ink, and a plain "Signed: ________" looked hidden
    (third review).
    """
    height = box[3] - box[1]
    if char in _LOW_CHARS:
        return box[0], box[3] - 0.35 * height, box[2], box[3] + 0.05 * height
    if char in _HIGH_CHARS:
        return box[0], box[1], box[2], box[1] + 0.6 * height
    return _core(box)


def _pixel_area(transform: tuple[float, ...], shape: tuple[int, ...], box: Box) -> tuple[int, int, int, int] | None:
    a, b, c, d, e, f = transform
    corners = ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3]))
    xs = [a * x + c * y + e for x, y in corners]
    ys = [b * x + d * y + f for x, y in corners]
    height, width = shape[0], shape[1]
    left, right = max(int(math.floor(min(xs))), 0), min(int(math.ceil(max(xs))), width)
    top, bottom = max(int(math.floor(min(ys))), 0), min(int(math.ceil(max(ys))), height)
    return (top, bottom, left, right) if right > left and bottom > top else None


def _contrast(picture: Any, transform: tuple[float, ...], box: Box) -> int:
    """How much the finished page varies inside ``box`` (0 = one flat colour: nothing can be read there)."""
    area = _pixel_area(transform, picture.shape, box)
    if area is None:
        return 0
    top, bottom, left, right = area
    patch = picture[top:bottom, left:right]
    return int((patch.max(axis=(0, 1)).astype(int) - patch.min(axis=(0, 1)).astype(int)).max())


class _Boxes:
    """Boxes on a coarse grid, to find quickly which of them touch a given box.

    Without it every run of text is held against every shape on the page: a
    table of 3,000 cells took a quarter of a minute (third review).
    """

    CELL = 48.0
    MAX_CELLS = 400

    def __init__(self, items: list[tuple], box_at: int) -> None:
        self._items, self._box_at = items, box_at
        self._grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._large: list[int] = []
        for index, item in enumerate(items):
            box = item[box_at]
            columns = range(int(box[0] // self.CELL), int(box[2] // self.CELL) + 1)
            rows = range(int(box[1] // self.CELL), int(box[3] // self.CELL) + 1)
            if len(columns) * len(rows) > self.MAX_CELLS:
                self._large.append(index)
                continue
            for column in columns:
                for row in rows:
                    self._grid[(column, row)].append(index)

    def touching(self, box: Box) -> list[tuple]:
        """The items whose box overlaps ``box``, in their original order."""
        found = set(self._large)
        for column in range(int(box[0] // self.CELL), int(box[2] // self.CELL) + 1):
            for row in range(int(box[1] // self.CELL), int(box[3] // self.CELL) + 1):
                found.update(self._grid.get((column, row), ()))
        return [self._items[index] for index in sorted(found) if _overlap(self._items[index][self._box_at], box) > 0]


@dataclass
class _Pictures:
    """What the two drawings of a page say, as arrays of "how different is this pixel" (0..255)."""

    content_ink: Any  # the finished page against the finished page without the page's text
    annotation_ink: Any | None  # the finished page against the page without annotations (None: it has none)
    finished: Any  # the page as a reader sees it, in colour
    background: Any  # the finished page without the page's text, in colour: what lies behind (or over) the text
    content_commands: int  # drawing commands with a lower number belong to the page; the rest to annotations
    transform: tuple[float, float, float, float, float, float]  # page coordinates -> pixels

    def _pixels(self, box: Box) -> tuple[int, int, int, int] | None:
        return _pixel_area(self.transform, self.content_ink.shape, box)

    def rows_with_ink(self, box: Box, *, annotation: bool = False) -> float:
        """The share of the pixel rows inside ``box`` in which the page's text left ink.

        A letter fills most of its height. Text cut down to two hairlines by a
        clipping shape, or showing through two slits between white boxes, has
        ink too - in a tenth of its rows. A reader sees dotted lines.
        """
        picture = self.annotation_ink if annotation else self.content_ink
        area = self._pixels(box)
        if picture is None or area is None:
            return 0.0
        top, bottom, left, right = area
        return float((picture[top:bottom, left:right] >= NO_INK).any(axis=1).mean())

    def ink(self, box: Box, *, annotation: bool = False) -> int:
        """The strongest difference inside ``box`` (page coordinates)."""
        picture = self.annotation_ink if annotation else self.content_ink
        area = self._pixels(box)
        if picture is None or area is None:
            return 0
        top, bottom, left, right = area
        return int(picture[top:bottom, left:right].max())

    def own_ink(self, box: Box, colour: tuple[float, float, float], strength: float) -> bool:
        """Is the ink inside ``box`` this text's own ink, judged by its colour?

        Where the two pictures differ, some text put ink on the page - but
        whose? A white word placed just under a black line of underscores has
        black ink in its box that is not its own. So the strongest differing
        pixels are looked at: taking the text away must have moved them from
        the colour behind TOWARDS this text's colour. Ink that moved them
        elsewhere belongs to other text, and a colour that equals what is
        behind it can leave no ink at all.
        """
        import numpy

        area = self._pixels(box)
        if area is None:
            return False
        top, bottom, left, right = area
        patch = self.content_ink[top:bottom, left:right]

        def towards_this_colour(position: int) -> bool | None:
            """None: this pixel carries no ink, or says nothing (the colour behind it is the text's own)."""
            y, x = divmod(position, patch.shape[1])
            if patch[y, x] < NO_INK:
                return None
            behind = self.background[top + y, left + x]
            want = [(wanted - float(was)) * strength for wanted, was in zip(colour, behind, strict=True)]  # a fully covered pixel
            if max(abs(value) for value in want) < NO_INK:
                return None
            now = self.finished[top + y, left + x]
            move = [float(is_now) - float(was) for is_now, was in zip(now, behind, strict=True)]
            along = sum(a * b for a, b in zip(move, want, strict=True)) / sum(value * value for value in want)
            return along >= OWN_INK_SHARE

        # The strongest pixel settles nearly every character; only if it does not, the next three are asked.
        if towards_this_colour(int(patch.argmax())):
            return True
        for position in numpy.argsort(patch, axis=None)[-4:-1][::-1]:
            if towards_this_colour(int(position)):
                return True
        return False

    def colour_behind(self, box: Box) -> tuple[float, float, float] | None:
        """The typical colour of the text-free page inside ``box`` (red, green, blue: 0..255)."""
        import numpy

        area = self._pixels(box)
        if area is None:
            return None
        top, bottom, left, right = area
        middle = numpy.median(self.background[top:bottom, left:right].reshape(-1, 3), axis=0)
        return float(middle[0]), float(middle[1]), float(middle[2])

    def marks_behind(self, box: Box) -> int:
        """How much the text-free page varies inside ``box`` (0 = one flat colour)."""
        area = self._pixels(box)
        if area is None:
            return 0
        top, bottom, left, right = area
        patch = self.background[top:bottom, left:right]
        return int((patch.max(axis=(0, 1)).astype(int) - patch.min(axis=(0, 1)).astype(int)).max())


def _difference(first: Any, second: Any) -> Any:
    import numpy

    # |first - second| per colour channel, then the strongest of the three channels. Written with
    # max/min so that everything stays in single bytes (no larger temporary arrays).
    delta = numpy.maximum(first, second) - numpy.minimum(first, second)
    return numpy.maximum(numpy.maximum(delta[..., 0], delta[..., 1]), delta[..., 2])


def _draw(page: Any, matrix: Any, *, annotations: bool, alpha: bool = False) -> Any:
    import numpy
    import pymupdf

    pixmap = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=alpha, annots=annotations)
    return numpy.frombuffer(pixmap.samples, dtype=numpy.uint8).reshape(pixmap.height, pixmap.width, pixmap.n).copy()


# Annotations that show as a small icon (a sticky note, a paper clip). The icon sits on top of whatever
# it comments on, and every reader knows to look under it: it is not drawn for the comparison.
_ICON_ANNOTATIONS = ("Text", "Popup", "FileAttachment", "Sound")


def _layer_expression(tokens: list[str], states: dict[int, bool]) -> bool:
    """A visibility expression of a PDF layer rule: [/And a [/Not b]] ... Raises ValueError when it cannot be read."""
    def value(position: int) -> tuple[bool, int]:
        token = tokens[position]
        if token == "[":
            operator = tokens[position + 1]
            position += 2
            parts = []
            while tokens[position] != "]":
                part, position = value(position)
                parts.append(part)
            if operator == "/Not" and len(parts) == 1:
                return not parts[0], position + 1
            if operator in ("/And", "/Or") and parts:
                return (all(parts) if operator == "/And" else any(parts)), position + 1
            raise ValueError("unknown operator")
        if token.isdigit() and tokens[position + 1 : position + 3] == ["0", "R"]:
            return states.get(int(token), True), position + 3
        raise ValueError("unexpected token")

    try:
        result, end = value(0)
    except IndexError as exc:
        raise ValueError("incomplete expression") from exc
    if end != len(tokens):
        raise ValueError("trailing tokens")
    return result


def settle_layer_rules(document: Any) -> int:
    """Replace every layer rule (OCMD) of the document by its outcome. Returns how many were settled.

    Content of a PDF can belong to a layer rule: "show this if layers A and B
    are both on". PDF viewers work such rules out as the standard says. The
    library that draws pages here does not for every kind of rule (third
    review: text under "all of these layers must be on", with one of them off,
    is shown by no viewer and was drawn here - and read). So each rule is worked
    out once, as the standard says, and written back as the simplest rule with
    that outcome: "always shown", or "shown if this layer is on" with a layer
    that is off. A rule that cannot be read counts as "not shown".
    """
    layers = document.get_ocgs()
    if not layers:
        return 0
    states = {xref: bool(layer.get("on", True)) for xref, layer in layers.items()}
    hidden_layer: int | None = None
    settled = 0
    for xref in range(1, document.xref_length()):
        try:
            if document.xref_get_key(xref, "Type") != ("name", "/OCMD"):
                continue
            kind, expression = document.xref_get_key(xref, "VE")
            if kind == "xref":
                expression = document.xref_object(int(expression.split()[0]), compressed=True)
                kind = "array"
            if kind == "array":
                shown = _layer_expression(re.findall(r"\[|\]|/\w+|\d+", expression), states)
            else:
                _, listed = document.xref_get_key(xref, "OCGs")
                members = [states.get(int(number), True) for number in re.findall(r"(\d+)\s+0\s+R", listed or "")]
                policy = document.xref_get_key(xref, "P")[1] or "/AnyOn"
                shown = not members or {
                    "/AllOn": all(members), "/AnyOn": any(members), "/AnyOff": not all(members), "/AllOff": not any(members),
                }.get(policy, any(members))
        except Exception:  # noqa: BLE001 - a rule that cannot be read hides what it governs
            shown = False
        if shown:
            document.update_object(xref, "<</Type/OCMD>>")
        else:
            if hidden_layer is None:
                hidden_layer = next((number for number, on in states.items() if not on), None) or document.add_ocg("not shown", on=False)
            document.update_object(xref, f"<</Type/OCMD/OCGs[{hidden_layer} 0 R]/P/AnyOn>>")
        settled += 1
    return settled


class Workshop:
    """A private copy of a document, in which pages may be taken apart.

    To draw a page without its text the text has to be removed, and that must
    not happen to the document that is being read. The copy is a copy of the
    whole file (made once, when first needed), not of single pages: settings
    that belong to the document as a whole - such as which layers are switched
    off - then apply to the copy exactly as they apply to the original.

    Both pictures of a page are drawn from the copy, so that whatever is
    prepared here (layer rules settled, comment icons put aside) holds for both.
    """

    def __init__(self, document: Any, data: bytes | None = None) -> None:
        self._document, self._data, self._copy = document, data, None

    def page(self, index: int) -> tuple[Any, Any]:
        """(the copy, its page ``index``)."""
        import pymupdf

        if self._copy is None:
            self._copy = pymupdf.open(stream=self._data if self._data is not None else self._document.tobytes(), filetype="pdf")
            try:
                if settle_layer_rules(self._copy):
                    # Which layers are on is read once, when a file is opened: write the copy out and open it again.
                    settled = self._copy.tobytes()
                    self._copy.close()
                    self._copy = pymupdf.open(stream=settled, filetype="pdf")
            except Exception as exc:  # noqa: BLE001
                logger.warning("layer rules could not be settled: %s: %s", type(exc).__name__, exc)
        return self._copy, self._copy[index]

    def close(self) -> None:
        if self._copy is not None:
            self._copy.close()
            self._copy = None


def _finished_picture(page: Any, workshop: Workshop) -> tuple[Any, Any, tuple[float, ...], bool]:
    """(the page as a reader sees it, the drawing matrix, page coordinates -> pixels, has annotations)."""
    import pymupdf

    zoom = min(RENDER_ZOOM, math.sqrt(MAX_RENDER_PIXELS / max(page.rect.width * page.rect.height, 1.0)))
    if zoom < MIN_RENDER_ZOOM:
        # A poster-sized page drawn within the memory limit shows ordinary text as a few grey
        # pixels: every line would look "hardly visible". Better no comparison than a wrong one.
        raise RuntimeError("the page is too large to be drawn at a size at which text can be judged")
    matrix = pymupdf.Matrix(zoom, zoom)
    _, twin = workshop.page(page.number)
    for annotation in twin.annots() or ():
        if annotation.type[1] in _ICON_ANNOTATIONS and not annotation.flags & pymupdf.PDF_ANNOT_IS_HIDDEN:
            annotation.set_flags(annotation.flags | pymupdf.PDF_ANNOT_IS_HIDDEN)
            annotation.update()
    annotated = twin.first_annot is not None or twin.first_widget is not None
    return _draw(twin, matrix, annotations=True), matrix, tuple(page.rotation_matrix * matrix), annotated


def _pictures(page: Any, workshop: Workshop | None = None, finished: tuple[Any, Any, tuple[float, ...], bool] | None = None) -> _Pictures:
    """Draw the page with and without its text and compare. Raises when that is not possible."""
    import pymupdf

    own_workshop = workshop is None
    workshop = workshop or Workshop(page.parent)
    try:
        picture, matrix, transform, annotated = finished or _finished_picture(page, workshop)
        copy, twin = workshop.page(page.number)
        annotation_ink = _difference(picture, _draw(twin, matrix, annotations=False)) if annotated else None

        # The same page with all its text removed. The redaction tool does that: with images and line
        # art left untouched, and no fill, it takes out every character (also inside reused page
        # parts) and nothing else. It would take the annotations along, so the page's list of
        # annotations is put aside first and put back afterwards: they are part of what a reader
        # sees, and the page must be drawn with them, text or no text.
        kind, kept = copy.xref_get_key(twin.xref, "Annots")
        if kind != "null":
            copy.xref_set_key(twin.xref, "Annots", "null")
            twin = copy.reload_page(twin)
        content_commands = len(twin.get_bboxlog())
        # The whole page, in the page's own coordinates: on a page that is stored turned (/Rotate 90),
        # "the page's rectangle" as shown and as stored are two different rectangles (third review).
        everything = (twin.rect * twin.derotation_matrix).normalize()
        twin.add_redact_annot(everything | twin.mediabox, fill=False)
        twin.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
        # (Text that the tool cannot reach - inside a repeating pattern - stays in both pictures. It then
        # makes no difference between them and counts as not visible, which is the safe side.)
        copy.xref_set_key(twin.xref, "Annots", kept if kind != "null" else "null")
        twin = copy.reload_page(twin)
        without_text = _draw(twin, matrix, annotations=True)
        if without_text.shape != picture.shape:
            raise RuntimeError("the two pictures of the page have different sizes")
        return _Pictures(_difference(picture, without_text), annotation_ink, picture, without_text, content_commands, transform)
    finally:
        if own_workshop:
            workshop.close()


def _covered_share(boxes: list[Box], area: Box) -> float:
    """The share of ``area`` that the boxes cover together (on a coarse grid, so strips and tiles add up)."""
    width, height = area[2] - area[0], area[3] - area[1]
    if width <= 0 or height <= 0 or not boxes:
        return 0.0
    cells = 40
    covered = [[False] * cells for _ in range(cells)]
    for box in boxes:
        first_x = max(int((box[0] - area[0]) / width * cells + 0.5), 0)
        last_x = min(int((box[2] - area[0]) / width * cells + 0.5), cells)
        first_y = max(int((box[1] - area[1]) / height * cells + 0.5), 0)
        last_y = min(int((box[3] - area[1]) / height * cells + 0.5), cells)
        for y in range(first_y, last_y):
            row = covered[y]
            for x in range(first_x, last_x):
                row[x] = True
    return sum(sum(row) for row in covered) / (cells * cells)


def trace_page(page: Any, number: int, workshop: Workshop | None = None) -> PageTrace:
    """Every character on the page with the verdict "visible" or the reason it is hidden."""
    import unicodedata

    # The trace works in the page's own coordinates, before any /Rotate is applied.
    visible_area: Box = tuple((page.rect * page.derotation_matrix).normalize())
    trace = PageTrace(number=number, area=_area(visible_area), height=visible_area[3] - visible_area[1])
    try:
        log = page.get_bboxlog()
        spans = page.get_texttrace()
        drawings = page.get_drawings() if any(kind == "fill-path" for kind, _ in log) else []
    except Exception as exc:  # noqa: BLE001 - an unusual PDF must not stop the upload
        logger.warning("page %d could not be traced: %s: %s", number, type(exc).__name__, exc)
        trace.failed = True
        return trace

    trace.images = [(index, tuple(box)) for index, (kind, box) in enumerate(log) if kind in ("fill-image", "fill-imgmask")]
    # A scanner stores a page as one picture, as several strips, or as a one-colour stencil. What
    # makes a page "a scan" is that pictures cover it; whether its invisible text is then accepted
    # as recognised text is decided character by character (_keep_recognised_text).
    if _covered_share([box for _, box in trace.images], visible_area) >= SCAN_IMAGE_SHARE:
        trace.scans = list(trace.images)
    if not spans:
        return trace

    own_workshop = workshop is None
    workshop = workshop or Workshop(page.parent)
    pictures: _Pictures | None = None
    finished: tuple[Any, Any, tuple[float, ...], bool] | None = None
    try:
        finished = _finished_picture(page, workshop)
        pictures = _pictures(page, workshop, finished)
    except Exception as exc:  # noqa: BLE001
        logger.warning("page %d could not be drawn for comparison (%s: %s); using the drawing order only", number, type(exc).__name__, exc)
        trace.compared = False
    finally:
        if own_workshop:
            workshop.close()
    # The finished page alone still answers one question, also when the comparison could not be made:
    # is there anything to see at this place at all?
    shown, shown_transform = (finished[0], finished[2]) if finished is not None else (None, None)

    # Everything except text that was drawn, in order: what can lie over a character.
    covers = [(index, tuple(box)) for index, (kind, box) in enumerate(log) if kind not in _NOT_A_COVER]
    # A font that draws its own shapes (Type 3) may draw anything for any character.
    try:
        drawn_fonts = any(str(font[2]).lower() == "type3" for font in page.get_fonts())
    except Exception:  # noqa: BLE001
        drawn_fonts = False
    trace.drawn_fonts = drawn_fonts
    # Opaque filled rectangles: (sequence number, box, brightness). Used by the drawing-order rule.
    fills = []
    for drawing in drawings:
        opacity = drawing.get("fill_opacity")
        if drawing.get("fill") is None or (opacity is not None and opacity < 0.9):
            continue
        if not drawing.get("items") or any(item[0] not in ("re", "qu") for item in drawing["items"]):
            continue  # only plain rectangles: the bounding box of a curve says little about what it covers
        fills.append((int(drawing.get("seqno", -1)), tuple(drawing["rect"]), _luminance(drawing["fill"])))
    if pictures is None:
        # Without the pictures only the drawing order is left. Then every shape filled later counts
        # as a cover by its outline box, whatever its form: too much is called covered rather than too little.
        known = {seqno for seqno, _, _ in fills}
        fills += [(index, tuple(box), 1.0) for index, (kind, box) in enumerate(log) if kind == "fill-path" and index not in known]
    fills_near, images_near, covers_near = _Boxes(fills, 1), _Boxes(trace.images, 1), _Boxes(covers, 1)
    try:
        fields = [tuple(widget.rect * page.derotation_matrix) for widget in page.widgets()]
    except Exception:  # noqa: BLE001
        fields = []

    # ---- pass 1: what the drawing order says about each character, and how much ink it left
    records: list[dict[str, Any]] = []
    upright_spans: set[int] = set()
    for span_number, span in enumerate(spans):
        seqno, size = int(span.get("seqno", 0)), float(span.get("size", 0.0))
        box = tuple(span["bbox"])
        font = str(span.get("font", ""))
        unseen = int(span.get("type", 0)) == 3 or float(span.get("opacity", 1.0)) < 0.05
        brightness = _luminance(span.get("color"), span.get("colorspace"))
        direction = span.get("dir") or (1.0, 0.0)
        if abs(float(direction[0])) > 0.99:
            upright_spans.add(span_number)
        touching_fills, touching_images = fills_near.touching(box), images_near.touching(box)
        later = [fill_box for fill_seq, fill_box, _ in touching_fills if fill_seq > seqno]
        later += [image_box for image_seq, image_box in touching_images if image_seq > seqno]
        behind = [item for item in touching_fills if item[0] < seqno]
        images_behind = [image_box for image_seq, image_box in touching_images if image_seq < seqno]
        from_annotation = pictures is not None and seqno >= pictures.content_commands
        # Later drawings of the page itself. (What annotations do is read off the picture.)
        last_command = pictures.content_commands if pictures is not None else len(log)
        touching_covers = covers_near.touching(box)
        anything_later = [(cover_seq, cover_box) for cover_seq, cover_box in touching_covers if seqno < cover_seq < last_command]
        colour = _rgb(span.get("color"), span.get("colorspace"))
        strength = float(span.get("opacity", 1.0))
        # A plain grey, RGB or CMYK colour can be compared with the picture; a pattern or a font that
        # paints its own colours cannot.
        plain_colour = span.get("color") is not None and span.get("colorspace") in (1, 3, 4) and not drawn_fonts

        last_drawn: int | None = None  # the last character of this run that has a width of its own
        followers = 0
        measured = 0
        previous: dict[str, Any] | None = None
        stacked: list[dict[str, Any]] = []
        for code, _glyph, origin, char_box in span["chars"]:
            char = chr(code)
            char_box = tuple(char_box)
            record = {
                "char": TracedChar(
                    char, char_box, (float(origin[0]), float(origin[1])), seqno, font, size, None,
                    annotation=from_annotation, opacity=strength,
                ),
                "span": span_number, "unseen": unseen, "annotation": from_annotation, "order": None, "ink": 255, "under": None, "below_something": False,
                "colour": colour, "strength": strength, "rivals": [], "own": True,
            }
            records.append(record)
            if char.isspace():
                record["neutral"] = True
                continue
            if char_box[2] - char_box[0] < 0.05:
                # A character without width. An accent or a joiner draws nothing of its own. Any other
                # one is the second half of a ligature - or text that is in the file and not on the
                # page: a whole sentence can be hung on one full stop that way (third review). It
                # goes with the character before it, and only a ligature's worth of them may follow.
                record["neutral"] = True
                if unicodedata.category(char) not in ("Mn", "Me", "Cf"):
                    followers += 1
                    record["follows"], record["followers"] = last_drawn, followers
                continue
            last_drawn, followers = len(records) - 1, 0
            if previous is not None and previous["char"].char != char and _overlap(previous["char"].bbox, char_box) >= 0.8 * min(
                _area(previous["char"].bbox), _area(char_box)
            ):
                stacked += [candidate for candidate in (previous, record) if candidate not in stacked]
            previous = record
            centre_x, centre_y = (char_box[0] + char_box[2]) / 2, (char_box[1] + char_box[3]) / 2
            on_image = any(_contains_point(image_box, centre_x, centre_y) for image_box in images_behind)
            # The drawing-order verdict (the only one before the pictures existed).
            if unseen:
                record["order"] = "invisible"
            elif any(_overlap(cover, char_box) >= COVER_SHARE * max(_area(char_box), 1e-6) for cover in later):
                record["order"] = "covered"
            elif not on_image:  # the colours of an image are unknown here
                under = [light for _, fill_box, light in sorted(behind, key=lambda item: item[0]) if _contains_point(fill_box, centre_x, centre_y)]
                background = under[-1] if under else 1.0  # the topmost rectangle beneath, else white paper
                if abs(brightness - background) < MIN_CONTRAST:
                    record["order"] = "same_colour"
            over_it = [cover_seq for cover_seq, cover in anything_later if _contains_point(cover, centre_x, centre_y)]
            record["under"] = min(over_it) if over_it else None  # the first later drawing of the page that lies over it
            record["plain_colour"] = plain_colour
            # For the wording of the reason only: is anything at all drawn over it later, annotations included?
            record["below_something"] = bool(over_it) or any(
                cover_seq > seqno and _contains_point(cover, centre_x, centre_y) for cover_seq, cover in touching_covers
            )
            band = _band(char, char_box)
            if shown is not None and (pictures is None or from_annotation):
                # Asked only where the comparison cannot answer: for annotations, and when it could not be made.
                record["contrast"] = _contrast(shown, shown_transform, band)
            if pictures is not None:
                record["ink"] = pictures.ink(band, annotation=from_annotation)
                if plain_colour and not from_annotation and not unseen and record["ink"] >= NO_INK:
                    record["own"] = pictures.own_ink(band, colour, strength)
                if measured < MAX_MEASURED_PER_RUN and char.isalnum() and record["ink"] >= NO_INK:
                    # A dozen letters say enough about a run: is it text, or slivers of text?
                    measured += 1
                    record["rows"] = pictures.rows_with_ink(_core(char_box), annotation=from_annotation)
        if len(stacked) >= MIN_STACKED:
            for record in stacked:
                record["stacked"] = True

    # ---- pass 2: characters of different text commands printed on the same spot ("contested")
    contested: set[int] = set()
    if pictures is not None:
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        cores = {}
        for index, record in enumerate(records):
            char = record["char"]
            if record.get("neutral") or record["annotation"]:
                continue
            cores[index] = _core(char.bbox)
            grid[(int((char.bbox[0] + char.bbox[2]) / 24), int((char.bbox[1] + char.bbox[3]) / 24))].append(index)
        for (cell_x, cell_y), members in grid.items():
            near = [other for dx in (-1, 0, 1) for dy in (-1, 0, 1) for other in grid.get((cell_x + dx, cell_y + dy), [])]
            for index in members:
                mine = cores[index]
                for other in near:
                    theirs = records[other]["char"]
                    if theirs.seqno != records[index]["char"].seqno and _overlap(mine, cores[other]) > CONTEST_SHARE * max(_area(mine), 1e-6):
                        contested.add(index)
                        contested.add(other)
                        records[index]["rivals"].append(theirs.seqno)
                        records[other]["rivals"].append(records[index]["char"].seqno)

    # ---- pass 3: what can only be said about a run of text as a whole
    # Fonts in which many different characters come out as one and the same shape.
    same_shape_fonts = _fonts_drawn_as_one_shape(records, pictures) if pictures is not None else set()
    faint_spans: set[int] = set()  # runs in a colour that is not a plain colour (a pattern), and hardly visible
    sliver_spans: set[int] = set()  # runs of which only thin slivers show
    if pictures is not None:
        by_span: dict[int, list[int]] = defaultdict(list)
        rows_by_span: dict[int, list[float]] = defaultdict(list)
        for index, record in enumerate(records):
            if record.get("neutral") or index in contested or record["ink"] < NO_INK:
                continue
            # Text of annotations is always judged this way: what is behind it is not known apart from it.
            if record["annotation"] or not record.get("plain_colour"):
                by_span[record["span"]].append(record["ink"])
            if "rows" in record and record["span"] in upright_spans:
                rows_by_span[record["span"]].append(record["rows"])
        faint_spans = {span for span, inks in by_span.items() if sorted(inks)[len(inks) // 2] < FAINT_INK}
        sliver_spans = {
            span for span, rows in rows_by_span.items()
            if len(rows) >= MIN_RUN_FOR_SHAPE_CHECKS and sorted(rows)[len(rows) // 2] < MIN_INK_ROWS
        }

    # ---- pass 4: the verdict
    for index, record in enumerate(records):
        char: TracedChar = record["char"]
        centre = (char.bbox[0] + char.bbox[2]) / 2, (char.bbox[1] + char.bbox[3]) / 2
        nothing_to_see = record.get("contrast", 255) < FAINT_INK  # the finished page is one flat colour there
        if "follows" in record:
            # A character without width that is not an accent: it is whatever the character before it is,
            # and beyond a ligature's worth of them it is text that the page does not draw.
            leader = records[record["follows"]]["char"] if record["follows"] is not None else None
            if not _contains_point(visible_area, *centre):
                char.hidden = "off_page"
            elif char.size < MIN_VISIBLE_SIZE:
                char.hidden = "tiny"
            elif record["unseen"]:
                char.hidden = "invisible"
            elif leader is None or record["followers"] > MAX_LIGATURE_EXTRA:
                char.hidden = "no_shape"
            else:
                char.hidden = leader.hidden
        elif not record.get("neutral"):
            # By its middle, not by its area: what counts is where the character stands.
            if not _contains_point(visible_area, *centre):
                char.hidden = "off_page"
            elif char.size < MIN_VISIBLE_SIZE:
                char.hidden = "tiny"
            elif record.get("stacked"):
                char.hidden = "stacked"
            elif pictures is None:
                # Only the drawing order - and, where the page could at least be drawn, the question
                # whether there is anything to see at the place at all (white text on a white picture).
                char.hidden = record["order"] or ("same_colour" if nothing_to_see else None)
            elif record["annotation"]:
                # Text of an annotation. The picture says whether it changed the page at all and
                # whether anything shows where it stands; the drawing order says whether it is covered.
                pale = record["ink"] < NO_INK or record["span"] in faint_spans or nothing_to_see
                char.hidden = record["order"] or ("same_colour" if pale else "slivers" if record["span"] in sliver_spans else None)
                if char.hidden == "same_colour" and record["ink"] < NO_INK and any(
                    # on the field's line and past its start, or under the field: where a value runs to that does not fit
                    (area[1] <= centre[1] <= area[3] and centre[0] >= area[0]) or (area[0] <= centre[0] <= area[2] and centre[1] > area[3])
                    for area in fields
                ):
                    char.hidden = "overflow"
            elif char.font in same_shape_fonts:
                char.hidden = "not_drawn"
            elif index in contested:
                char.hidden = _contested_verdict(record, pictures)
            elif record["ink"] < NO_INK or not record["own"]:
                char.hidden = _why(record)
            elif record["order"] in ("covered", "invisible") and pictures.ink(_middle(char.bbox)) < NO_INK:
                # The drawing order says "covered", the picture shows ink in the character's box. Then
                # the ink must be in the middle of the box, where only this character's own strokes
                # are - not at its edge, where a neighbouring line reaches in.
                char.hidden = _why(record)
            elif record["span"] in sliver_spans:
                char.hidden = "slivers"
            elif _is_pale(record, pictures, faint_spans):
                char.hidden = "same_colour"
            if char.hidden == "same_colour" and char.size >= LARGE_TEXT and record["ink"] >= NO_INK:
                char.hidden = "pale_large"  # a watermark: left out of the text, and not a finding
        trace.chars.append(char)

    # An outlined heading is drawn twice: filled in the colour of the paper, and outlined in black. The
    # filling leaves nothing a reader can see - and is not hidden text: the same character, at the same
    # place, in the same font, is visible.
    seen: dict[tuple[str, float, float, str], bool] = {}
    for char in trace.chars:
        if char.hidden is None and not char.char.isspace():
            seen[(*_key(char.char, char.origin), char.font)] = True
    for char in trace.chars:
        if char.hidden in ("same_colour", "pale_large") and seen.get((*_key(char.char, char.origin), char.font)):
            char.hidden = "twin"

    _keep_recognised_text(trace, records, pictures)
    return trace


def _rgb(colour: Any, colourspace: int | None = None) -> tuple[float, float, float]:
    """A grey, RGB or CMYK colour (0..1) as red, green, blue (0..255)."""
    if colour is None:
        return 0.0, 0.0, 0.0
    values = [float(v) for v in colour] if isinstance(colour, (tuple, list)) else [float(colour)]
    size = colourspace or len(values)
    if size == 1 or len(values) < 3:
        values = [values[0]] * 3
    elif size == 4 and len(values) >= 4:
        c, m, y, k = values[:4]
        values = [(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)]
    return values[0] * 255, values[1] * 255, values[2] * 255


def _contested_verdict(record: dict[str, Any], pictures: _Pictures) -> str | None:
    """A character with other text printed on the same spot: the ink there may not be its own.

    So the picture cannot vouch for it, and it must show that it is visible:

    * no ink at all on the spot          -> hidden (whatever is printed there, nothing shows)
    * the drawing order says hidden      -> hidden (invisible ink; an opaque rectangle or image drawn later)
    * a later drawing lies over it, and
      the other text came after that     -> covered (the altered amount: old text, a shape over it, new text on top)
    * its colour is the colour behind it -> same colour (white text on a white picture, with other text printed across)

    Otherwise it is visible: a value typed onto the dotted blank of a form,
    bold text made by printing every letter twice, two words that run into
    each other. In the third rule, "the other text came after" matters: when
    all the text on the spot is older than the drawing over it and ink still
    shows, the drawing is see-through (a highlighter), and nothing is covered.
    """
    if record["ink"] < NO_INK:
        return _why(record)
    # Of the drawing order's verdicts only two are taken: its third, "same colour", assumes white
    # paper wherever no rectangle was filled, and is wrong on a gradient or a picture. The colour
    # behind the character is read off the picture further down instead.
    if record["order"] in ("invisible", "covered"):
        return record["order"]
    if record["under"] is not None and any(rival > record["under"] for rival in record["rivals"]):
        return "covered"
    char: TracedChar = record["char"]
    behind = pictures.colour_behind(_core(char.bbox))
    if behind is not None:
        difference = max(abs(mine - theirs) for mine, theirs in zip(record["colour"], behind, strict=True)) * record["strength"]
        if difference < FAINT_INK:
            return "same_colour"
    return None


def _is_pale(record: dict[str, Any], pictures: _Pictures, faint_spans: set[int]) -> bool:
    """Can this text hardly be told from what is behind it?

    Judged by the colour the file gives the text against the colour the picture
    shows behind it, character by character: one pale run of text may stand
    partly on a dark banner (readable) and partly on white paper (not). The ink
    a character leaves is no measure for this - a thin comma in a readable
    grey leaves less of it than a broad letter in an unreadable one.
    """
    if not record.get("plain_colour"):
        return record["span"] in faint_spans
    if record["ink"] >= 3 * FAINT_INK:
        return False  # strong ink: no need to look closer
    behind = pictures.colour_behind(_core(record["char"].bbox))
    if behind is None:
        return False
    return max(abs(mine - theirs) for mine, theirs in zip(record["colour"], behind, strict=True)) * record["strength"] < FAINT_INK


def _fonts_drawn_as_one_shape(records: list[dict[str, Any]], pictures: _Pictures) -> set[str]:
    """Fonts whose different characters are all drawn as the same shape.

    A font can declare "this is an A" and draw a dot. A font in which eight or
    more different letters and digits produce one and the same picture says
    something to a program and nothing to a reader. (A font that draws a
    different meaningless shape for every letter, or other letters altogether -
    a 5 where the text says 8 - cannot be told from an honest one without
    reading the picture, which this program does not do.)
    """
    import numpy

    shapes: dict[str, dict[str, bytes]] = defaultdict(dict)
    for record in records:
        char: TracedChar = record["char"]
        # Every kind of font (third review): an ordinary font can be told, in the file, to draw a full
        # stop for every letter just as well.
        if record.get("neutral") or record["annotation"] or record["ink"] < NO_INK or not char.char.isalnum():
            continue
        if char.char in shapes[char.font]:
            continue
        area = pictures._pixels(char.bbox)
        if area is None:
            continue
        top, bottom, left, right = area
        patch = pictures.content_ink[top:bottom, left:right] >= NO_INK
        rows, columns = numpy.nonzero(patch)
        if not len(rows):
            continue
        # The inked part of the box, reduced to a 6 x 6 grid: the same for the same shape, wherever it stands.
        inked = patch[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
        ys = numpy.linspace(0, inked.shape[0], 7).astype(int)
        xs = numpy.linspace(0, inked.shape[1], 7).astype(int)
        grid = bytes(
            int(inked[ys[y] : max(ys[y + 1], ys[y] + 1), xs[x] : max(xs[x + 1], xs[x] + 1)].mean() >= 0.5)
            for y in range(6) for x in range(6)
        )
        proportions = bytes([min(int(4 * inked.shape[0] / max(inked.shape[1], 1)), 255)])
        shapes[char.font][char.char] = grid + proportions
    return {
        font for font, by_char in shapes.items()
        if len(by_char) >= MIN_SAME_SHAPES and len(set(by_char.values())) <= max(len(by_char) // 6, 1)
    }


def _why(record: dict[str, Any]) -> str:
    """The reason given for a character that left no ink."""
    if record["unseen"]:
        return "invisible"
    return "covered" if record["below_something"] else "same_colour"


def _keep_recognised_text(trace: PageTrace, records: list[dict[str, Any]], pictures: _Pictures | None) -> None:
    """On a scanned page, invisible text is the recognised text of the scan: keep it.

    A scan is a picture of a page. OCR software adds what it read as text in
    invisible ink over the picture (some programs put it under the picture
    instead), so that the page can be searched and copied from.

    That convention is honoured where it is plausible and nowhere else: the
    text must lie on an opaque image that fills the page, and the image must
    show something at that place - a blank white "scan" with invisible text
    on it is not a scan of that text. Whether the picture really says what
    the invisible text says cannot be known without reading the picture
    (OCR), which this version does not do.
    """
    if pictures is None:
        # Drawing order only: as before the pictures existed, invisible ink on any image counts as recognised text.
        for record in records:
            char = record["char"]
            if char.hidden == "invisible":
                centre = (char.bbox[0] + char.bbox[2]) / 2, (char.bbox[1] + char.bbox[3]) / 2
                if any(_contains_point(box, *centre) for _, box in trace.images):
                    char.hidden, char.ocr = None, True
        return
    if not trace.scans:
        return
    # A scan is a picture of a page. A page with pictures AND typed text of its own is an ordinary
    # page, and invisible text on it is hidden text (third review: only typed text drawn after the
    # picture was counted, so a photo placed under a typed invoice made the page "a scan"):
    #   * typed text next to the pictures - an invoice with a photo on its lower half - or
    #   * a page full of typed text on the picture - a letterhead behind it, an ornament over it.
    # What may stand on a scan: the line a scanner or filing system prints in the top or bottom strip,
    # and a short typed note on the picture (a certification, a received stamp).
    strip = PAGE_MARGIN_SHARE * trace.height
    beside, on_the_picture = 0, 0
    for record in records:
        char = record["char"]
        if char.hidden is not None or char.char.isspace() or record["annotation"]:
            continue
        if char.bbox[3] <= strip or char.bbox[1] >= trace.height - strip:
            continue
        centre = (char.bbox[0] + char.bbox[2]) / 2, (char.bbox[1] + char.bbox[3]) / 2
        if any(_contains_point(box, *centre) for _, box in trace.scans):
            on_the_picture += 1
        else:
            beside += 1
    if beside > MAX_TYPED_BODY_CHARS_ON_SCAN or on_the_picture > MAX_TYPED_CHARS_ON_SCAN:
        return
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        char = record["char"]
        if char.hidden not in ("invisible", "covered") or record["annotation"]:
            continue
        centre = (char.bbox[0] + char.bbox[2]) / 2, (char.bbox[1] + char.bbox[3]) / 2
        on_scan = any(_contains_point(box, *centre) for _, box in trace.scans)
        under_scan = any(seqno > char.seqno and _contains_point(box, *centre) for seqno, box in trace.scans)
        if on_scan and (char.hidden == "invisible" or under_scan):
            candidates[record["span"]].append(record)
    for members in candidates.values():
        # Judged per run of text, not per character: recognised text rarely sits exactly on the printed letters.
        marked = sum(1 for record in members if pictures.marks_behind(record["char"].bbox) >= SCAN_MARKS)
        for record in members:
            if 2 * marked >= len(members):
                record["char"].hidden, record["char"].ocr = None, True
            else:
                # Recognised text where the picture shows nothing (a cleaned margin, a page number that
                # was erased - or a sentence somebody added). It is not read. See _hidden_indicators
                # for when it is reported.
                record["char"].hidden = "blank_scan"


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

    A character of the text view that the trace does not know at all has not
    been judged - and is not read (third review: until then it counted as
    visible). That is text the file carries without drawing it: "replacement
    text" that says more than the letters it replaces, text used only as a
    clipping shape, text filled with a pattern. What was left out for this
    reason is noted in ``trace.unjudged``.
    """
    import pymupdf

    verdicts: dict[tuple[str, float, float], list[str | None]] = defaultdict(list)
    # The two views do not always agree on which character a code stands for (fonts that draw their
    # own shapes). On such pages a character the trace knows under another name is found by its place.
    by_place: dict[tuple[float, float], list[str | None]] = defaultdict(list)
    for char in trace.chars:
        verdicts[_key(char.char, char.origin)].append(char.hidden)
        by_place[_key("", char.origin)[1:]].append(char.hidden)
    try:
        raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("page %d: text structure could not be read: %s", trace.number, exc)
        return None

    # An outlined heading is two drawing commands (fill, outline) and one character in the text view.
    # Where the trace holds more verdicts for an identity than the text view has characters, the
    # verdicts of the fillings ("twin") are the ones to spare.
    wanted: Counter[tuple[str, float, float]] = Counter(
        _key(char["c"], char["origin"])
        for block in raw.get("blocks", []) for line in block.get("lines", []) for span in line.get("spans", []) for char in span.get("chars", [])
    )
    for identity, queue in verdicts.items():
        surplus = len(queue) - wanted.get(identity, 0)
        while surplus > 0 and "twin" in queue:
            queue.remove("twin")
            surplus -= 1

    trace.unjudged = []
    lines: list[tuple[Box, list[tuple[str, float, int, str]]]] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            spans, total, seen = [], 0, []
            for span in line.get("spans", []):
                kept, left_out = [], []
                followers, leader_kept = 0, False
                for char in span.get("chars", []):
                    total += 1
                    letter = char["c"]
                    queue = verdicts.get(_key(letter, char["origin"]))
                    if not queue and trace.drawn_fonts:
                        queue = by_place.get(_key("", char["origin"])[1:])
                    if queue:
                        hidden = queue.pop(0) is not None
                        followers, leader_kept = 0, not hidden
                    elif letter.isspace():
                        hidden = False  # spaces are often put in by the text view itself, between columns
                    elif trace.failed or not trace.chars:
                        hidden = False  # nothing is known about this page: it is read as it is, and the report says so
                    elif char["bbox"][2] - char["bbox"][0] < 0.05 and leader_kept and followers < MAX_LIGATURE_EXTRA:
                        followers += 1
                        hidden = False  # the second half of a ligature
                    else:
                        hidden = True
                        left_out.append(letter)
                    if not hidden:
                        kept.append(letter)
                        if not letter.isspace():
                            seen.append(char["bbox"])
                    elif left_out and letter.isspace():
                        left_out.append(" ")
                if left_out:
                    trace.unjudged.append("".join(left_out))
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


def _page_text(page: Any, trace: PageTrace, lines: list[list[tuple[str, float, int, str]]] | None) -> str:
    import pymupdf

    if trace.failed or not trace.has_hidden or lines is None:
        # Nothing is hidden (nearly every document): exactly PyMuPDF's normal text extraction.
        return page.get_text("text", flags=pymupdf.TEXTFLAGS_TEXT)
    return "".join("".join(text for _, _, _, text in spans) + "\n" for spans in lines)


def visible_text(page: Any, number: int = 1, trace: PageTrace | None = None) -> str:
    """The text of one page as a reader sees it: everything except hidden characters."""
    trace = trace or trace_page(page, number)
    lines = _visible_lines(page, trace) if not trace.failed else None
    return _page_text(page, trace, lines)


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

    unjudged = " ".join(" ".join(trace.unjudged).split())
    if unjudged:
        by_reason["no_shape"].append((0, unjudged))

    for reason, every in by_reason.items():
        if reason in ("pale_large", "twin"):
            continue  # a watermark, the inside of an outlined heading: not read and not reported
        if reason in ("blank_scan", "overflow") and not any(detect_prompt_injection(text) for _, text in every):
            # Stray recognised text on blank paper; the end of a value that does not fit its form field:
            # not read, and nothing to report.
            continue
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
            covered_boxes = _Boxes([(char.bbox,) for char in trace.chars if char.hidden == "covered"], 0)
            covered_from = min(seqno for seqno, _ in items)
            on_top = [
                char for char in trace.chars
                if char.hidden is None and not char.char.isspace() and char.seqno > covered_from
                and any(_overlap(char.bbox, box) > 0.3 * max(_area(char.bbox), 1e-6) for (box,) in covered_boxes.touching(char.bbox))
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
    import unicodedata

    # Only the page's own, solid, ordinary text. A stamp is an annotation, a "PAID" across the page is
    # see-through, and a tick box is a symbol with an X typed into it: all of them lie over other
    # text without anyone having typed a new value over an old one (third review).
    visible = [
        char for char in trace.chars
        if char.hidden is None and not char.ocr and not char.char.isspace() and not char.annotation and char.opacity >= 0.9
        and unicodedata.category(char.char) != "So" and not _SYMBOL_FONT.search(char.font)
    ]
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
                if char.char in FILLER_CHARS or other.char in FILLER_CHARS:
                    continue  # a value typed onto the dotted or underlined blank of a form
                if max(char.size, other.size) > 1.5 * min(char.size, other.size):
                    continue  # a watermark or a heading running through the text, not a retyped value
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
    """A few typed words on top of a scanned page, where the page itself has content.

    Scanners, fax machines and filing systems print a line of their own at the
    very top or bottom of a page ("Scanned 03/03/2026 09:14  Page 1 of 1"). Text
    that stays inside those strips is not reported; a value typed into the body
    of a scan is.
    """
    # "A scan" here means a page of pictures that carries recognised text. A page with a picture
    # as its background and a few typed lines (a certificate, a cover sheet) is not one.
    if not trace.scans or not any(char.ocr for char in trace.chars):
        return None
    strip = PAGE_MARGIN_SHARE * trace.height
    typed = [
        char for char in trace.chars
        if char.hidden is None and not char.ocr and not char.char.isspace() and not char.annotation
        and not (char.bbox[3] <= strip or char.bbox[1] >= trace.height - strip)  # not the scanner's own line
    ]
    if not typed:
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
            # A slightly different size is what retyping a value leaves behind. A clearly larger
            # one (11 pt label, 13 pt amount) is emphasis, and deliberate.
            elif family == usual[0] and 0.4 <= abs(size - usual[1]) < 1.5 and rare_size(size, lines):
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


def read_pdf(document: Any, data: bytes, now: datetime | None = None) -> tuple[list[str], list[Indicator], dict[str, Any]]:
    """One pass over an open PDF: (visible text of every page, indicators, metadata).

    Reading and inspecting are one job - both start from "which characters can
    a reader see?" - so they are done together and each page is drawn only once.
    """
    found, metadata = _history_indicators(document, data, now or datetime.now(timezone.utc))
    texts: list[str] = []
    lines: list[tuple[int, list[tuple[str, float, int, str]]]] = []
    unread: list[int] = []
    order_only: list[int] = []
    scanned: list[int] = []
    drawn_fonts: list[int] = []
    workshop = Workshop(document, data)
    for number, page in enumerate(document, start=1):
        trace = trace_page(page, number, workshop)
        if any(char.ocr for char in trace.chars):
            scanned.append(number)
        if trace.drawn_fonts and any(not char.char.isspace() for char in trace.chars):
            drawn_fonts.append(number)
        # Every page is read line by line against its trace: only that shows text which the file
        # carries without drawing it.
        page_lines = _visible_lines(page, trace) if not trace.failed else None
        texts.append(_page_text(page, trace, page_lines))
        if trace.failed:
            unread.append(number)
            continue
        if not trace.compared:
            order_only.append(number)
        # Hidden text is reported on whatever page it is found (third review: beyond page 200 it was
        # left out of what is read, and nobody was told). The other checks stop after a while.
        if len(found) < MAX_INDICATORS_COLLECTED:
            found += _hidden_indicators(trace)
        if number > MAX_PAGES_INSPECTED:
            continue
        for extra in (_overlap_indicator(trace), _scan_indicator(trace)):
            if extra is not None:
                found.append(extra)
        lines += [(number, spans) for spans in page_lines or []]
    found += _font_indicators(lines)
    # Said openly, because on these pages hidden text may have been read as part of the document.
    if unread:
        metadata["pages_not_checked"] = unread[:50]
    if order_only:
        metadata["pages_checked_by_drawing_order_only"] = order_only[:50]
    if scanned:
        metadata["scanned_pages"] = scanned[:50]
    if drawn_fonts:
        metadata["pages_with_drawn_fonts"] = drawn_fonts[:50]
    workshop.close()
    try:
        import pymupdf

        pymupdf.TOOLS.store_shrink(100)  # give back what the drawing of this file kept in memory
    except Exception:  # noqa: BLE001
        pass
    return texts, found, metadata


# The loader reads a file a moment before the forensic report is built from the same bytes.
# The few indicators found on the way are kept here for that moment, so the file is not
# drawn twice. Keyed by the file's hash; small and bounded.
_recent: OrderedDict[str, tuple[list[Indicator], dict[str, Any]]] = OrderedDict()
_recent_lock = threading.Lock()
_MAX_RECENT = 8


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remember_inspection(data: bytes, found: list[Indicator], metadata: dict[str, Any]) -> None:
    with _recent_lock:
        _recent[_digest(data)] = (found, metadata)
        while len(_recent) > _MAX_RECENT:
            _recent.popitem(last=False)


def inspect_pdf(data: bytes, now: datetime | None = None) -> tuple[list[Indicator], dict[str, Any]]:
    """All indicators for one PDF file, and its metadata."""
    import pymupdf

    if now is None:
        with _recent_lock:
            kept = _recent.pop(_digest(data), None)
        if kept is not None:
            return kept
    document = pymupdf.open(stream=data, filetype="pdf")
    try:
        _, found, metadata = read_pdf(document, data, now)
        return found, metadata
    finally:
        document.close()
