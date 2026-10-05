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
MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
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
    cell_fill: Colour | None = None  # shading set on the table cell itself
    table_run_properties: list[Any] = field(default_factory=list)  # from the table style, strongest first
    table_fill: Colour | None = None  # shading the table style gives this cell


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
        background = document.element.find(f"{W}background")
        self.page = (self._colour(background, "color", "themeColor") if background is not None else None) or WHITE

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
        slot = _THEME_SLOTS.get(element.get(f"{W}{theme}") or "")
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

    def _shading(self, properties: Any) -> Colour | None:
        """The fill of a w:shd element inside ``properties`` (None when there is none, or it is automatic)."""
        shading = properties.find(f"{W}shd") if properties is not None else None
        if shading is None:
            return None
        if (shading.get(f"{W}val") or "clear") == "solid":  # a solid pattern is painted in the pattern colour
            return self._colour(shading, "color", "themeColor") or (0, 0, 0)
        return self._colour(shading, "fill", "themeFill")

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
                    fill = self._shading(holder.find(f"{W}tcPr"))
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

        # Everything else: the first place that says something wins.
        levels = [level for level in (own, *character, *paragraph, *place.table_run_properties, self._defaults) if level is not None]

        def first(tag: str) -> Any:
            return next((level.find(f"{W}{tag}") for level in levels if level.find(f"{W}{tag}") is not None), None)

        size = first("sz")
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

        colour_setting = first("color")
        colour = self._colour(colour_setting, "val", "themeColor") if colour_setting is not None else None
        if colour is None:
            return None  # automatic: Word itself picks black or white, whichever can be read
        behind = self._behind(levels, place)
        if max(abs(mine - theirs) for mine, theirs in zip(colour, behind, strict=True)) < MIN_COLOUR_DIFFERENCE:
            return "white text" if colour == WHITE and behind == WHITE else "in the colour of its background"
        return None

    def _behind(self, run_levels: list[Any], place: _Place) -> Colour:
        """The colour behind a run: its highlight, its own shading, the paragraph's, the cell's, or the page."""
        for level in run_levels:
            highlight = level.find(f"{W}highlight")
            if highlight is not None and (highlight.get(f"{W}val") or "none") in HIGHLIGHT_COLOURS:
                return HIGHLIGHT_COLOURS[highlight.get(f"{W}val")]
            fill = self._shading(level)
            if fill is not None:
                return fill
        if place.paragraph is not None:
            properties = place.paragraph.find(f"{W}pPr")
            fill = self._shading(properties)
            if fill is None:
                style = self._style_of(properties, "pStyle") or self._default_paragraph_style
                fill = next((found for item in self._chain(style) if (found := self._shading(item.find(f"{W}pPr"))) is not None), None)
            if fill is not None:
                return fill
        return place.cell_fill or place.table_fill or self.page


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

    def read(self) -> DocxReading:
        seen: set[int] = set()

        def once(part: Any) -> None:  # sections usually share one header: read it once
            if part is not None and id(part._element) not in seen and not part.is_linked_to_previous:
                seen.add(id(part._element))
                self._blocks(part._element, _Place(), self.reading.lines)

        for section in self._document.sections:
            once(section.header)
        self._blocks(self._document.element.body, _Place(), self.reading.lines)
        for section in self._document.sections:
            once(section.footer)
        return self.reading

    def _blocks(self, container: Any, place: _Place, lines: list[str]) -> None:
        """Paragraphs and tables in ``container``, in order. Content controls (w:sdt) are looked into."""
        for child in container:
            if child.tag == f"{W}p":
                text = self._paragraph(child, place)
                if text.strip():
                    lines.append(text)
            elif child.tag == f"{W}tbl":
                self._table(child, lines)
            elif child.tag in (f"{W}sdt", f"{W}sdtContent", f"{W}customXml", f"{W}smartTag"):
                self._blocks(child, place, lines)

    def _table(self, table: Any, lines: list[str]) -> None:
        rows = [row for row in table.iter(f"{W}tr") if self._nearest(row, f"{W}tbl") is table]
        for row_index, row in enumerate(rows):
            cells = [cell for cell in row.iter(f"{W}tc") if self._nearest(cell, f"{W}tr") is row]
            texts = []
            for column_index, cell in enumerate(cells):
                run_properties, table_fill = self._formatting.table_part(table, row_index, len(rows), column_index, len(cells))
                place = _Place(
                    cell_fill=self._formatting._shading(cell.find(f"{W}tcPr")), table_run_properties=run_properties, table_fill=table_fill,
                )
                inside: list[str] = []
                self._blocks(cell, place, inside)
                if inside:
                    texts.append("\n".join(inside).strip())
            if texts:
                lines.append(" | ".join(texts))

    @staticmethod
    def _nearest(element: Any, tag: str) -> Any:
        parent = element.getparent()
        while parent is not None and parent.tag != tag:
            parent = parent.getparent()
        return parent

    def _paragraph(self, paragraph: Any, place: _Place) -> str:
        """One paragraph as Word shows it once all tracked changes are accepted.

        Inserted text (``w:ins``) and hyperlinks are included. Deleted text is
        stored as ``w:delText`` and is never collected; text that was moved
        away (``w:moveFrom``) is left out. A text box anchored in the
        paragraph follows it on lines of its own; Word stores every text box
        twice (once for old versions of Word), and the second copy is skipped.
        """
        here = _Place(paragraph, place.cell_fill, place.table_run_properties, place.table_fill)
        own: list[str] = []
        boxes: list[str] = []
        self._runs(paragraph, here, own, boxes)
        return "\n".join(["".join(own), *boxes]) if boxes else "".join(own)

    def _runs(self, container: Any, place: _Place, texts: list[str], boxes: list[str]) -> None:
        for child in container:
            if child.tag in (MC_FALLBACK, f"{W}moveFrom", f"{W}del"):
                continue
            if child.tag == f"{W}txbxContent":
                self._blocks(child, _Place(), boxes)
                continue
            if child.tag == f"{W}r":
                reason = self._formatting.hidden_reason(child, place)
                text = _run_text(child)
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
    for reason, texts in read_docx(document).hidden.items():
        text = " ".join(texts)
        kind = "hidden_instruction" if detect_prompt_injection(text) else "hidden_text"
        extra = " and reads like instructions to an AI system" if kind == "hidden_instruction" else ""
        found.append(indicator(
            kind,
            f"The document contains text that a reader does not see ({reason}){extra}. It was not read as part of the document.",
            region=text,
        ))

    # ---- tracked changes
    inserted = [element for element in body.iter(f"{W}ins") if element.find(f".//{W}r") is not None]
    deleted = [element for element in body.iter(f"{W}del") if element.find(f".//{W}delText") is not None]
    if inserted or deleted:
        removed = " ".join("".join(node.text or "" for node in element.iter(f"{W}delText")) for element in deleted).strip()
        added = " ".join("".join(_run_text(run) for run in element.iter(f"{W}r")) for element in inserted).strip()
        region = "; ".join(part for part in (f"deleted: {removed}" if removed else "", f"inserted: {added}" if added else "") if part)
        found.append(indicator(
            "tracked_changes",
            f"The document contains tracked changes that were never accepted or rejected ({len(inserted)} insertion(s), "
            f"{len(deleted)} deletion(s)). It was read with the changes applied.",
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
