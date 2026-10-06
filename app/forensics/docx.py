"""Hidden text, tracked changes and file history of Word documents.

A .docx file is a zip archive of XML files. ``word/document.xml`` holds the
text as paragraphs (``w:p``) made of runs (``w:r``); a run is a piece of text
with one formatting. Three things in there matter for forensics:

    hidden runs       formatted "hidden", in the colour of their background, or too small to read.
                      Word does not show them (or nobody can read them); a text extractor would.
    tracked changes   insertions (``w:ins``) and deletions (``w:del``) that nobody accepted
                      or rejected. The file then holds two versions of the text.
    core properties   author, last editor, creation and modification time (``docProps/core.xml``).

``read_docx`` gives what the application reads: the text Word shows with all
changes accepted, without hidden runs.

What "Word shows" means here is decided from what the file says, without
laying the pages out. That covers how text is formatted and a few plain
statements about where it is put (a row one twip high, a text box pushed off
the page). It does not cover what only a layout would show - a white shape
that happens to lie over a sentence. The README lists these limits.

Concept - effective formatting:
    How a run looks is rarely written on the run itself. Word looks in five
    places, the first that says something wins:

        the run -> its character style -> its paragraph's style -> the table style -> the document defaults

    A check that only reads the run's own formatting is fooled by a style
    named "Ghost" that is set to hidden, and it accuses an honest table whose
    white header text stands on dark cells (the dark is on the cell, not on
    the run). So this module resolves both sides the way Word does - the
    colour of the text and the colour behind it - and calls a run hidden when
    the two can hardly be told apart, whatever the two colours are.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from app.forensics.model import Indicator, indicator
from app.security.guardrails import detect_prompt_injection

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
W14 = "{http://schemas.microsoft.com/office/word/2010/wordml}"
MIN_HIDING_TRANSPARENCY = 88_000  # w14:alpha counts transparency in thousandths of a percent: 88% and more cannot be read
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
MC_ALTERNATE, MC_CHOICE, MC_FALLBACK = f"{MC}AlternateContent", f"{MC}Choice", f"{MC}Fallback"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
WPS = "{http://schemas.microsoft.com/office/word/2010/wordprocessingShape}"
# What a "Choice" of alternative content may require and Word (2010 and later) understands. A choice that
# requires anything else is skipped by every program - the fallback next to it is what is shown.
_UNDERSTOOD = frozenset({
    "wps", "wpg", "wpc", "wpi", "wp14", "w14", "w15", "w16", "w16se", "w16cid", "w16cex", "w16sdtdh", "w16du",
    "a14", "a16", "cx", "cx1", "cx2", "cx3", "cx4", "cx5", "cx6", "cx7", "cx8", "aink", "am3d", "oel", "v",
})
NOTES = {
    "footnotes": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes",
    "endnotes": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes",
}
MIN_ROW_HEIGHT_TWIPS = 60  # a table row fixed at less than 3 points shows nothing of what is in it
FAR_OUTSIDE_TWIPS = 7200  # five inches left of where a paragraph belongs: off the paper
EMU_PER_TWIP = 635
NO_FILL = "no fill"  # a shading that says "none" - which is not the same as saying nothing
V = "{urn:schemas-microsoft-com:vml}"
MIN_LINE_HEIGHT_TWIPS = 60  # lines fixed at less than 3 points: nothing of the text shows
MIN_FIT_WIDTH_TWIPS = 100  # a run "fitted" into less than 5 points of width
MAX_CONDENSED_TWIPS = -160  # letters pulled 8 points closer together stand on top of each other
# Hidden text that is no sign of anything, so it is left out without a finding (unless it reads like an
# instruction): a "continued" header for later pages on a document that has one page.
QUIET_REASONS = ("in a header that", "in a footer that")
THEME_RELATIONSHIP = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"

MIN_VISIBLE_HALF_POINTS = 6  # w:sz counts half points: 6 = 3 pt, the same limit as for PDFs
MIN_TEXT_SCALE = 20  # w:w squeezes characters; below 20% of their width nothing can be read
MIN_COLOUR_DIFFERENCE = 30  # of 255, strongest channel: about 12%, the same limit as for PDFs
_OFF = ("0", "false", "off")
_HEX = re.compile(r"^[0-9A-Fa-f]{6}$")
WHITE: tuple[int, int, int] = (255, 255, 255)

Colour = tuple[int, int, int]

HIGHLIGHT_COLOURS: dict[str, Colour] = {
    "yellow": (255, 255, 0), "green": (0, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "blue": (0, 0, 255), "red": (255, 0, 0), "darkBlue": (0, 0, 128), "darkCyan": (0, 128, 128),
    "darkGreen": (0, 128, 0), "darkMagenta": (128, 0, 128), "darkRed": (128, 0, 0), "darkYellow": (128, 128, 0),
    "darkGray": (128, 128, 128), "lightGray": (192, 192, 192), "black": (0, 0, 0), "white": (255, 255, 255),
}
# The names Word uses for theme colours, and the slot of the theme they point to.
_THEME_SLOTS = {
    "dark1": "dk1", "text1": "dk1", "light1": "lt1", "background1": "lt1",
    "dark2": "dk2", "text2": "dk2", "light2": "lt2", "background2": "lt2",
    "accent1": "accent1", "accent2": "accent2", "accent3": "accent3", "accent4": "accent4", "accent5": "accent5",
    "accent6": "accent6", "hyperlink": "hlink", "followedHyperlink": "folHlink",
}
_DEFAULT_THEME: dict[str, Colour] = {
    "dk1": (0, 0, 0), "lt1": (255, 255, 255), "dk2": (68, 84, 106), "lt2": (231, 230, 230),
    "accent1": (68, 114, 196), "accent2": (237, 125, 49), "accent3": (165, 165, 165), "accent4": (255, 192, 0),
    "accent5": (91, 155, 213), "accent6": (112, 173, 71), "hlink": (5, 99, 193), "folHlink": (149, 79, 114),
}
# Parts of a table that a table style can format differently, weakest first.
_TABLE_PARTS = ("wholeTable", "band1Horz", "band2Horz", "firstCol", "lastCol", "lastRow", "firstRow")


def _hex(value: str | None) -> Colour | None:
    if value and _HEX.match(value):
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return None


def _on(element: Any) -> bool:
    """<w:vanish/> = on, <w:vanish w:val="0"/> = off."""
    return (element.get(f"{W}val") or "true").lower() not in _OFF


def _flag(properties: Any, name: str) -> bool:
    element = properties.find(f"{W}{name}") if properties is not None else None
    return element is not None and _on(element)


@dataclass
class _Place:
    """Where a run stands: what Word needs to know to decide how it looks."""

    paragraph: Any = None
    cell_fill: Colour | str | None = None  # shading set on the table cell itself (NO_FILL: set to "none")
    table_run_properties: list[Any] = field(default_factory=list)  # from the table style, strongest first
    table_fill: Colour | None = None  # shading the table style gives this cell
    backdrop: Colour | None = None  # what this container stands on: the cell around a nested table, a text box's fill


class _Formatting:
    """Resolves effective formatting from the document's styles and theme."""

    def __init__(self, document: Any) -> None:
        styles = document.styles.element
        self._styles = {style.get(f"{W}styleId"): style for style in styles.iter(f"{W}style")}
        self._defaults = styles.find(f"{W}docDefaults/{W}rPrDefault/{W}rPr")
        self._default_paragraph_style = next(
            (style.get(f"{W}styleId") for style in styles.iter(f"{W}style")
             if style.get(f"{W}type") == "paragraph" and (style.get(f"{W}default") or "0") not in _OFF),
            None,
        )
        self._theme = dict(_DEFAULT_THEME)
        try:
            from lxml import etree

            theme = etree.fromstring(document.part.part_related_by(THEME_RELATIONSHIP).blob)
            for slot in _DEFAULT_THEME:
                node = theme.find(f".//{A}clrScheme/{A}{slot}")
                if node is not None and len(node):
                    colour = _hex(node[0].get("lastClr") or node[0].get("val"))
                    if colour:
                        self._theme[slot] = colour
        except Exception:  # noqa: BLE001 - no theme, or an odd one: the standard colours apply
            pass
        # A document can say that "text 1" means the light colour (settings: clrSchemeMapping). Then black-looking
        # theme text is white.
        self._slots = dict(_THEME_SLOTS)
        shows_background = False
        try:
            settings = document.settings.element
            mapping = settings.find(f"{W}clrSchemeMapping")
            if mapping is not None:
                for written, names in (("t1", ("text1", "dark1")), ("t2", ("text2", "dark2")), ("bg1", ("background1", "light1")), ("bg2", ("background2", "light2"))):
                    target = _THEME_SLOTS.get(mapping.get(f"{W}{written}") or "")
                    if target:
                        for name in names[:1]:
                            self._slots[name] = target
            shows_background = settings.find(f"{W}displayBackgroundShape") is not None
        except Exception:  # noqa: BLE001
            pass
        # A page colour is only painted when the settings say so; otherwise the page is white whatever the file names.
        background = document.element.find(f"{W}background")
        self.page = (self._colour(background, "color", "themeColor") if background is not None and shows_background else None) or WHITE

    # ---- styles
    def _chain(self, style_id: str | None) -> list[Any]:
        """A style and the styles it is based on, nearest first."""
        chain, seen = [], set()
        while style_id and style_id in self._styles and style_id not in seen:
            seen.add(style_id)
            chain.append(self._styles[style_id])
            based_on = chain[-1].find(f"{W}basedOn")
            style_id = based_on.get(f"{W}val") if based_on is not None else None
        return chain

    def _style_of(self, properties: Any, tag: str) -> str | None:
        element = properties.find(f"{W}{tag}") if properties is not None else None
        return element.get(f"{W}val") if element is not None else None

    def _levels(self, run: Any, place: _Place) -> tuple[Any, list[Any], list[Any]]:
        """(the run's own formatting, its character style chain, its paragraph style chain) as rPr elements."""
        own = run.find(f"{W}rPr")
        character = [style.find(f"{W}rPr") for style in self._chain(self._style_of(own, "rStyle"))]
        paragraph_properties = place.paragraph.find(f"{W}pPr") if place.paragraph is not None else None
        paragraph_style = self._style_of(paragraph_properties, "pStyle") or self._default_paragraph_style
        paragraph = [style.find(f"{W}rPr") for style in self._chain(paragraph_style)]
        return own, [item for item in character if item is not None], [item for item in paragraph if item is not None]

    # ---- colours
    def _colour(self, element: Any, value: str, theme: str) -> Colour | None:
        """A colour written as (w:val, w:themeColor) or (w:fill, w:themeFill). None = automatic."""
        slot = self._slots.get(element.get(f"{W}{theme}") or "")
        if not slot:
            return _hex(element.get(f"{W}{value}"))
        red, green, blue = self._theme[slot]
        tint_name, shade_name = ("themeFillTint", "themeFillShade") if theme == "themeFill" else ("themeTint", "themeShade")
        try:
            if element.get(f"{W}{tint_name}"):  # mixed towards white
                share = int(element.get(f"{W}{tint_name}"), 16) / 255
                red, green, blue = (round(c * share + 255 * (1 - share)) for c in (red, green, blue))
            if element.get(f"{W}{shade_name}"):  # mixed towards black
                share = int(element.get(f"{W}{shade_name}"), 16) / 255
                red, green, blue = (round(c * share) for c in (red, green, blue))
        except ValueError:
            pass
        return red, green, blue

    def _shading(self, properties: Any) -> Colour | str | None:
        """What a w:shd element inside ``properties`` paints: a colour, NO_FILL ("none", said outright), or
        None when there is no such element.

        A shading is a pattern in one colour over a fill in another. "clear" shows the fill, "solid" the
        pattern colour, and "pct95" is 95% pattern colour: white text on a "95% white on black" shading
        stands on white, whatever the fill says (third review).
        """
        shading = properties.find(f"{W}shd") if properties is not None else None
        if shading is None:
            return None
        pattern = shading.get(f"{W}val") or "clear"
        fill = self._colour(shading, "fill", "themeFill")
        if pattern in ("clear", "nil"):
            return fill if fill is not None else NO_FILL
        ink = self._colour(shading, "color", "themeColor") or (0, 0, 0)
        if pattern == "solid":
            return ink
        share = 0.5  # stripes and other patterns: about half and half
        if pattern.startswith("pct"):
            try:
                share = min(max(int(pattern[3:]) / 100, 0.0), 1.0)
            except ValueError:
                pass
        under = fill if fill is not None else WHITE
        return tuple(round(share * top + (1 - share) * bottom) for top, bottom in zip(ink, under, strict=True))  # type: ignore[return-value]

    # ---- tables
    def table_part(self, table: Any, row_index: int, row_count: int, column_index: int, column_count: int) -> tuple[list[Any], Colour | None]:
        """What the table's style says about one cell: (run formatting, strongest first; fill)."""
        properties = table.find(f"{W}tblPr")
        chain = self._chain(self._style_of(properties, "tblStyle"))
        if not chain:
            return [], None
        look = properties.find(f"{W}tblLook") if properties is not None else None
        bits = 0x04A0  # Word's default: header row, first column, row bands
        if look is not None:
            try:
                bits = int(look.get(f"{W}val") or "04A0", 16)
            except ValueError:
                pass
            for attribute, bit in (("firstRow", 0x20), ("lastRow", 0x40), ("firstColumn", 0x80), ("lastColumn", 0x100), ("noHBand", 0x200)):
                written = look.get(f"{W}{attribute}")
                if written is not None:
                    bits = bits | bit if written.lower() not in _OFF else bits & ~bit
        header, footer = bool(bits & 0x20), bool(bits & 0x40)
        applies = {"wholeTable"}
        if header and row_index == 0:
            applies.add("firstRow")
        elif footer and row_index == row_count - 1:
            applies.add("lastRow")
        elif not bits & 0x200:
            applies.add("band1Horz" if (row_index - (1 if header else 0)) % 2 == 0 else "band2Horz")
        if bits & 0x80 and column_index == 0:
            applies.add("firstCol")
        if bits & 0x100 and column_index == column_count - 1:
            applies.add("lastCol")

        run_properties: list[Any] = []
        fill: Colour | None = None
        for part in reversed(_TABLE_PARTS):  # strongest first
            if part not in applies:
                continue
            for style in chain:  # nearest style first
                if part == "wholeTable":
                    holder = style
                else:
                    holder = next((item for item in style.findall(f"{W}tblStylePr") if item.get(f"{W}type") == part), None)
                if holder is None:
                    continue
                if holder.find(f"{W}rPr") is not None:
                    run_properties.append(holder.find(f"{W}rPr"))
                if fill is None:
                    found = self._shading(holder.find(f"{W}tcPr"))
                    fill = found if isinstance(found, tuple) else None
        return run_properties, fill

    # ---- the verdict
    def hidden_reason(self, run: Any, place: _Place) -> str | None:
        """Why a reader does not see this run, or None if they do."""
        own, character, paragraph = self._levels(run, place)

        # "Hidden" is a toggle: the run's own setting decides; without one, a style switches it on, and a
        # character style and a paragraph style that both switch it on cancel each other (as in Word).
        for name in ("vanish", "specVanish"):
            direct = own.find(f"{W}{name}") if own is not None else None
            if direct is not None:
                if _on(direct):
                    return "formatted as hidden text"
                continue
            from_styles = [
                _on(setting) for chain in (character, paragraph)
                if (setting := next((level.find(f"{W}{name}") for level in chain if level.find(f"{W}{name}") is not None), None)) is not None
            ]
            if sum(from_styles) % 2 == 1:
                return "formatted as hidden text"
            if not from_styles:
                # Neither the run nor its styles say anything: then the table style decides, then the defaults.
                weakest = next(
                    (level.find(f"{W}{name}") for level in (*place.table_run_properties, self._defaults)
                     if level is not None and level.find(f"{W}{name}") is not None),
                    None,
                )
                if weakest is not None and _on(weakest):
                    return "formatted as hidden text"

        # Everything else: the first place that says something wins.
        levels = [level for level in (own, *character, *paragraph, *place.table_run_properties, self._defaults) if level is not None]

        def first(tag: str) -> Any:
            return next((level.find(f"{W}{tag}") for level in levels if level.find(f"{W}{tag}") is not None), None)

        complex_script = any(first(tag) is not None and _on(first(tag)) for tag in ("cs", "rtl"))
        size = first("szCs") if complex_script and first("szCs") is not None else first("sz")
        try:
            if size is not None and 0 < int(size.get(f"{W}val") or 0) < MIN_VISIBLE_HALF_POINTS:
                return f"smaller than {MIN_VISIBLE_HALF_POINTS // 2} points"
        except ValueError:
            pass
        scale = first("w")
        try:
            if scale is not None and 0 < int((scale.get(f"{W}val") or "100").rstrip("%")) < MIN_TEXT_SCALE:
                return "squeezed until it cannot be read"
        except ValueError:
            pass

        fitted, spacing = first("fitText"), first("spacing")
        try:
            if fitted is not None and 0 < int(fitted.get(f"{W}val") or MIN_FIT_WIDTH_TWIPS) < MIN_FIT_WIDTH_TWIPS:
                return "squeezed until it cannot be read"
            if spacing is not None and int(spacing.get(f"{W}val") or 0) <= MAX_CONDENSED_TWIPS:
                return "squeezed until it cannot be read"
        except ValueError:
            pass

        # Word's text effects can make a run transparent or give it no fill at all (w14:textFill). A plain
        # colour given there is the colour Word paints, whatever w:color says.
        effect_colour: Colour | None = None
        for level in levels:
            fill = level.find(f"{W14}textFill")
            if fill is None:
                continue
            if fill.find(f"{W14}noFill") is not None:
                return "transparent text"
            alpha = fill.find(f".//{W14}alpha")
            written = (alpha.get(f"{W14}val") or "0") if alpha is not None else "0"
            try:
                transparency = float(written.rstrip("%")) * 1000 if written.endswith("%") else float(written)
            except ValueError:
                transparency = 0.0
            if transparency >= MIN_HIDING_TRANSPARENCY:
                return "transparent text"
            plain = fill.find(f"{W14}solidFill/{W14}srgbClr")
            effect_colour = _hex(plain.get(f"{W14}val")) if plain is not None else None
            break

        colour_setting = first("color")
        colour = effect_colour or (self._colour(colour_setting, "val", "themeColor") if colour_setting is not None else None)
        if colour is None:
            return None  # automatic: Word itself picks black or white, whichever can be read
        behind = self._behind(levels, place)
        if max(abs(mine - theirs) for mine, theirs in zip(colour, behind, strict=True)) < MIN_COLOUR_DIFFERENCE:
            return "white text" if colour == WHITE and behind == WHITE else "in the colour of its background"
        return None

    def _behind(self, run_levels: list[Any], place: _Place) -> Colour:
        """The colour behind a run: its highlight, its own shading, the paragraph's, the cell's, or the page.

        At each of these, "no fill" said outright ends the search there: a run that switches its style's
        dark shading off stands on whatever is behind the paragraph, not on the dark (third review).
        """
        for level in run_levels:
            highlight = level.find(f"{W}highlight")
            if highlight is not None and (highlight.get(f"{W}val") or "none") in HIGHLIGHT_COLOURS:
                return HIGHLIGHT_COLOURS[highlight.get(f"{W}val")]
            fill = self._shading(level)
            if isinstance(fill, tuple):
                return fill
            if fill == NO_FILL:
                break
        if place.paragraph is not None:
            properties = place.paragraph.find(f"{W}pPr")
            fill = self._shading(properties)
            if fill is None:
                style = self._style_of(properties, "pStyle") or self._default_paragraph_style
                fill = next((found for item in self._chain(style) if (found := self._shading(item.find(f"{W}pPr"))) is not None), None)
            if isinstance(fill, tuple):
                return fill
        if isinstance(place.cell_fill, tuple):
            return place.cell_fill
        if place.cell_fill != NO_FILL and place.table_fill is not None:
            return place.table_fill
        return place.backdrop or self.page

    def backdrop(self, place: _Place) -> Colour:
        """The colour a container inside this place stands on (a table nested in a cell)."""
        return self._behind([], _Place(None, place.cell_fill, [], place.table_fill, place.backdrop))


def _run_text(run: Any) -> str:
    parts = []
    for child in run:
        if child.tag == f"{W}t":
            parts.append(child.text or "")
        elif child.tag == f"{W}tab":
            parts.append("\t")
        elif child.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
        elif child.tag == f"{W}noBreakHyphen":
            parts.append("-")
    return "".join(parts)


@dataclass
class DocxReading:
    lines: list[str] = field(default_factory=list)  # what a reader sees, in the order of the document
    hidden: dict[str, list[str]] = field(default_factory=dict)  # reason -> the hidden pieces of text


class _Reader:
    def __init__(self, document: Any) -> None:
        self._document = document
        self._formatting = _Formatting(document)
        self.reading = DocxReading()
        self._field: list[str] = []  # open fields, innermost last: "code" until the separator, then "result"
        self.deleted_rows = 0

    # ---- which parts a reader sees
    def _pages(self) -> int | None:
        """The number of pages the file says it has (Word writes it on saving). None: it does not say."""
        try:
            from lxml import etree

            for part in self._document.part.package.iter_parts():
                if str(part.partname).lower() == "/docprops/app.xml":
                    pages = etree.fromstring(part.blob).find("{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages")
                    return int(pages.text) if pages is not None and (pages.text or "").strip().isdigit() else None
        except Exception:  # noqa: BLE001
            return None
        return None

    def read(self) -> DocxReading:
        seen: set[int] = set()
        pages = self._pages()
        one_page = pages == 1
        try:
            odd_and_even = bool(self._document.settings.odd_and_even_pages_header_footer)
        except Exception:  # noqa: BLE001
            odd_and_even = False

        def part(holder: Any, shown: bool, where: str) -> None:
            # Sections usually share one header: each is read once.
            if holder is None or id(holder._element) in seen or holder.is_linked_to_previous:
                return
            seen.add(id(holder._element))
            if shown:
                self._blocks(holder._element, _Place(), self.reading.lines)
                return
            # A header that only appears from the second page on, in a document of one page: no reader
            # ever sees it (third review). It is read into nowhere and reported.
            unseen: list[str] = []
            self._blocks(holder._element, _Place(), unseen)
            text = " ".join(" ".join(unseen).split())
            if text:
                self.reading.hidden.setdefault(f"in a {where} that the document's only page does not show", []).append(text)

        def headers_or_footers(kind: str) -> None:
            for index, section in enumerate(self._document.sections):
                own_first = bool(section.different_first_page_header_footer)
                if own_first:
                    part(getattr(section, f"first_page_{kind}"), True, kind)
                # The ordinary header: on every page - or, where the first page has its own, from the second on.
                part(getattr(section, kind), not (own_first and one_page and index == 0), kind)
                if odd_and_even:
                    part(getattr(section, f"even_page_{kind}"), not one_page, kind)

        headers_or_footers("header")
        self._blocks(self._document.element.body, _Place(), self.reading.lines)
        self._notes()
        headers_or_footers("footer")
        return self.reading

    def _notes(self) -> None:
        """Footnotes and endnotes: printed on the page, kept in files of their own."""
        try:
            from lxml import etree

            for name, relationship in NOTES.items():
                try:
                    blob = self._document.part.part_related_by(relationship).blob
                except KeyError:
                    continue
                for note in etree.fromstring(blob):
                    if note.get(f"{W}type") in ("separator", "continuationSeparator", "continuationNotice"):
                        continue  # the rule above the notes, not a note
                    self._blocks(note, _Place(), self.reading.lines)
        except Exception:  # noqa: BLE001 - notes are an extra; a file with odd ones is still read
            return

    # ---- blocks
    def _blocks(self, container: Any, place: _Place, lines: list[str]) -> None:
        """Paragraphs and tables in ``container``, in order. Content controls (w:sdt) are looked into."""
        for child in container:
            if child.tag == f"{W}p":
                text = self._paragraph(child, place)
                if text.strip():
                    lines.append(text)
            elif child.tag == f"{W}tbl":
                self._table(child, lines, place)
            elif child.tag in (f"{W}sdt", f"{W}sdtContent", f"{W}customXml", f"{W}smartTag"):
                self._blocks(child, place, lines)
            elif child.tag == MC_ALTERNATE:
                chosen = self._alternative(child)
                if chosen is not None:
                    self._blocks(chosen, place, lines)

    @staticmethod
    def _alternative(element: Any) -> Any:
        """Of alternative content, the part that is shown: the first choice a program understands, else the fallback.

        A choice that requires something nobody knows is shown by nobody (third review: its text was read,
        and the fallback that everyone sees was not).
        """
        for choice in element.findall(MC_CHOICE):
            if set((choice.get("Requires") or "").split()) <= _UNDERSTOOD:
                return choice
        return element.find(MC_FALLBACK)

    def _table(self, table: Any, lines: list[str], around: _Place) -> None:
        # A table inside a cell stands on that cell's colour.
        backdrop = self._formatting.backdrop(around)
        rows = [row for row in table.iter(f"{W}tr") if self._nearest(row, f"{W}tbl") is table]
        for row_index, row in enumerate(rows):
            properties = row.find(f"{W}trPr")
            if properties is not None and properties.find(f"{W}del") is not None:
                self.deleted_rows += 1  # a row deleted with change tracking on: gone once the changes are accepted
                continue
            height = properties.find(f"{W}trHeight") if properties is not None else None
            squashed = False
            if height is not None and height.get(f"{W}hRule") == "exact":
                try:
                    squashed = int(height.get(f"{W}val") or MIN_ROW_HEIGHT_TWIPS) < MIN_ROW_HEIGHT_TWIPS
                except ValueError:
                    pass
            cells = [cell for cell in row.iter(f"{W}tc") if self._nearest(cell, f"{W}tr") is row]
            texts = []
            for column_index, cell in enumerate(cells):
                run_properties, table_fill = self._formatting.table_part(table, row_index, len(rows), column_index, len(cells))
                place = _Place(
                    cell_fill=self._formatting._shading(cell.find(f"{W}tcPr")), table_run_properties=run_properties,
                    table_fill=table_fill, backdrop=backdrop,
                )
                inside: list[str] = []
                self._blocks(cell, place, inside)
                if inside:
                    texts.append("\n".join(inside).strip())
            if not texts:
                continue
            if squashed:
                self.reading.hidden.setdefault("in a table row too low to show it", []).append(" ".join(" ".join(texts).split()))
            else:
                lines.append(" | ".join(texts))

    @staticmethod
    def _nearest(element: Any, tag: str) -> Any:
        parent = element.getparent()
        while parent is not None and parent.tag != tag:
            parent = parent.getparent()
        return parent

    # ---- paragraphs
    def _far_outside(self, paragraph: Any) -> bool:
        """Is the paragraph pushed off the paper by its own settings (an indent or a frame far to the left)?"""
        properties = paragraph.find(f"{W}pPr")
        if properties is None:
            return False
        try:
            indent = properties.find(f"{W}ind")
            if indent is not None:
                left = indent.get(f"{W}left") or indent.get(f"{W}start") or "0"
                if int(left) <= -FAR_OUTSIDE_TWIPS:
                    return True
            frame = properties.find(f"{W}framePr")
            if frame is not None and any(
                not -FAR_OUTSIDE_TWIPS < int(frame.get(f"{W}{axis}") or 0) < 3 * FAR_OUTSIDE_TWIPS for axis in ("x", "y")
            ):
                return True  # a frame put five inches left of the page, or fifteen inches down it
            lines = properties.find(f"{W}spacing")
            if lines is not None and lines.get(f"{W}lineRule") == "exact" and 0 < int(lines.get(f"{W}line") or MIN_LINE_HEIGHT_TWIPS) < MIN_LINE_HEIGHT_TWIPS:
                return True  # lines fixed at a height in which nothing shows
        except ValueError:
            return False
        return False

    def _paragraph(self, paragraph: Any, place: _Place) -> str:
        """One paragraph as Word shows it once all tracked changes are accepted.

        Inserted text (``w:ins``) and hyperlinks are included. Deleted text is
        stored as ``w:delText`` and is never collected; text that was moved
        away (``w:moveFrom``) is left out. A text box anchored in the
        paragraph follows it on lines of its own; Word stores every text box
        twice (once for old versions of Word), and the second copy is skipped.
        """
        here = _Place(paragraph, place.cell_fill, place.table_run_properties, place.table_fill, place.backdrop)
        own: list[str] = []
        boxes: list[str] = []
        self._runs(paragraph, here, own, boxes)
        if self._far_outside(paragraph):
            text = " ".join("".join(own).split())
            if text:
                self.reading.hidden.setdefault("placed outside the page", []).append(text)
            own = []
        return "\n".join(["".join(own), *boxes]) if boxes else "".join(own)

    def _box(self, content: Any, place: _Place) -> tuple[_Place, bool]:
        """(where the text of a text box stands, True if the box is placed outside the page)."""
        shape = self._nearest(content, f"{WPS}wsp")
        backdrop = self._formatting.backdrop(place)
        if shape is not None:
            fill = shape.find(f"{WPS}spPr/{A}solidFill")
            colour = _hex(fill[0].get("val")) if fill is not None and len(fill) and fill[0].tag == f"{A}srgbClr" else None
            if colour is None and fill is not None and len(fill) and fill[0].tag == f"{A}schemeClr":
                slot = {"tx1": "dk1", "bg1": "lt1", "tx2": "dk2", "bg2": "lt2"}.get(fill[0].get("val") or "", fill[0].get("val") or "")
                colour = self._formatting._theme.get(slot)
            backdrop = colour or backdrop
        outside = False
        anchor = self._nearest(content, f"{WP}anchor")
        holder = anchor if anchor is not None else self._nearest(content, f"{WP}inline")
        described = holder.find(f"{WP}docPr") if holder is not None else None
        if described is not None and (described.get("hidden") or "0").lower() not in _OFF:
            outside = True  # switched off in Word's selection pane: not shown, not printed
        old_shape = self._nearest(content, f"{V}shape") or self._nearest(content, f"{V}rect")
        if old_shape is not None and "visibility:hidden" in (old_shape.get("style") or "").replace(" ", "").lower():
            outside = True
        if anchor is not None:
            try:
                extent = anchor.find(f"{WP}extent")
                size = {"H": int(extent.get("cx") or 0), "V": int(extent.get("cy") or 0)} if extent is not None else {"H": 0, "V": 0}
                section = self._document.sections[0]
                paper = {"H": int(section.page_width or 0), "V": int(section.page_height or 0)}
                for axis in ("H", "V"):
                    offset = anchor.find(f"{WP}position{axis}/{WP}posOffset")
                    if offset is None or not (offset.text or "").strip().lstrip("-").isdigit():
                        continue
                    start = int(offset.text)
                    # Wholly beyond the left or top edge (with room for the margin it may be measured from), or beyond the far edge.
                    if start + size[axis] < -2 * FAR_OUTSIDE_TWIPS * EMU_PER_TWIP // 5 or (paper[axis] and start > paper[axis]):
                        outside = True
            except (ValueError, TypeError):
                outside = False
        return _Place(backdrop=backdrop), outside

    def _runs(self, container: Any, place: _Place, texts: list[str], boxes: list[str]) -> None:
        for child in container:
            if child.tag in (MC_FALLBACK, f"{W}moveFrom", f"{W}del"):
                continue
            if child.tag == MC_ALTERNATE:
                chosen = self._alternative(child)
                if chosen is not None:
                    self._runs(chosen, place, texts, boxes)
                continue
            if child.tag == f"{W}txbxContent":
                inside, outside = self._box(child, place)
                if outside:
                    unseen: list[str] = []
                    self._blocks(child, inside, unseen)
                    text = " ".join(" ".join(unseen).split())
                    if text:
                        self.reading.hidden.setdefault("in a text box that is switched off or placed outside the page", []).append(text)
                else:
                    self._blocks(child, inside, boxes)
                continue
            if child.tag == f"{W}r":
                # A field is stored as: begin, its code, a separator, its result, end. The code is an
                # instruction to Word ("PAGE", "MERGEFIELD Name") and is never shown - whatever kind of
                # text element it is written in.
                for mark in child.findall(f"{W}fldChar"):
                    kind = mark.get(f"{W}fldCharType")
                    if kind == "begin":
                        self._field.append("code")
                    elif kind == "separate" and self._field:
                        self._field[-1] = "result"
                    elif kind == "end" and self._field:
                        self._field.pop()
                text = _run_text(child)
                if "code" in self._field:
                    if text.strip():
                        self.reading.hidden.setdefault("written inside a field code, which Word does not show", []).append(text.strip())
                    continue
                reason = self._formatting.hidden_reason(child, place)
                if reason is None:
                    texts.append(text)
                elif text.strip():
                    self.reading.hidden.setdefault(reason, []).append(text.strip())
            self._runs(child, place, texts, boxes)


def read_docx(document: Any) -> DocxReading:
    """What a reader of the Word document sees, and what is hidden from them."""
    return _Reader(document).read()


def inspect_docx(data: bytes) -> tuple[list[Indicator], dict[str, Any]]:
    import docx

    document = docx.Document(io.BytesIO(data))
    body = document.element.body
    found: list[Indicator] = []

    # ---- hidden runs
    reader = _Reader(document)
    for reason, texts in reader.read().hidden.items():
        text = " ".join(texts)
        kind = "hidden_instruction" if detect_prompt_injection(text) else "hidden_text"
        if kind == "hidden_text" and reason.startswith(QUIET_REASONS):
            continue  # not shown and not read - and nothing anyone did wrong
        extra = " and reads like instructions to an AI system" if kind == "hidden_instruction" else ""
        found.append(indicator(
            kind,
            f"The document contains text that a reader does not see ({reason}){extra}. It was not read as part of the document.",
            region=text,
        ))

    # ---- tracked changes
    inserted = [element for element in body.iter(f"{W}ins") if element.find(f".//{W}r") is not None]
    deleted = [element for element in body.iter(f"{W}del") if element.find(f".//{W}delText") is not None]
    if inserted or deleted or reader.deleted_rows:
        removed = " ".join("".join(node.text or "" for node in element.iter(f"{W}delText")) for element in deleted).strip()
        added = " ".join("".join(_run_text(run) for run in element.iter(f"{W}r")) for element in inserted).strip()
        region = "; ".join(part for part in (f"deleted: {removed}" if removed else "", f"inserted: {added}" if added else "") if part)
        found.append(indicator(
            "tracked_changes",
            f"The document contains tracked changes that were never accepted or rejected ({len(inserted)} insertion(s), "
            f"{len(deleted) + reader.deleted_rows} deletion(s)). It was read with the changes applied.",
            region=region,
        ))

    # ---- file history
    properties = document.core_properties
    metadata: dict[str, Any] = {}
    for name in ("author", "last_modified_by", "title"):
        value = getattr(properties, name, None)
        if value:
            metadata[name] = str(value)[:200]
    created, modified = properties.created, properties.modified
    if created:
        metadata["created"] = created.isoformat()
    if modified:
        metadata["modified"] = modified.isoformat()
    if properties.revision:
        metadata["revision"] = properties.revision
    if created and modified:
        if modified < created - timedelta(minutes=1):
            found.append(indicator("metadata_inconsistent", "The file says it was last changed before it was created."))
        elif modified - created > timedelta(minutes=2):
            found.append(indicator(
                "modified_after_creation",
                f"The file was changed after it was created (created {created.day} {created:%B %Y}, "
                f"last changed {modified.day} {modified:%B %Y}).",
            ))
    author, editor = (properties.author or "").strip(), (properties.last_modified_by or "").strip()
    if author and editor and author.lower() != editor.lower():
        found.append(indicator(
            "different_last_editor", "The person who last saved the file is not the person named as its author.",
            region=f"author: {author}; last saved by: {editor}",
        ))
    return found, metadata
