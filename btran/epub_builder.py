"""Render translated structural blocks as a standards-compliant EPUB 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
import mimetypes
from pathlib import Path
import re

from ebooklib import epub

from btran.schema import PageResult, SourceBlock


CSS = """
body { font-family: Georgia, serif; line-height: 1.6; margin: 1em auto; max-width: 42em; }
figure { margin: 1em 0; }
img { max-width: 100%; height: auto; }
figcaption { font-style: italic; }
table { border-collapse: collapse; }
th, td { border: 1px solid #888; padding: .3em; }
aside[epub\\:type="footnote"] { font-size: .9em; }
"""


class _SemanticChapter(epub.EpubHtml):
    """EbookLib HTML item that retains deliberately authored XHTML unchanged."""

    def get_content(self, default=None) -> bytes:  # type: ignore[override]
        return self.content if isinstance(self.content, bytes) else self.content.encode("utf-8")


@dataclass
class _PageFragment:
    page: PageResult
    blocks: list[SourceBlock]


@dataclass
class _Chapter:
    title: str
    fragments: list[_PageFragment] = field(default_factory=list)


def _escaped(value: str) -> str:
    return escape(value, quote=True)


def _block_kind(block: SourceBlock) -> str:
    return block.type.lower().replace("-", "_").replace(" ", "_")


def _heading_level(block: SourceBlock) -> int | None:
    kind = _block_kind(block)
    match = re.fullmatch(r"(?:heading|h)_?([123])", kind)
    if match:
        return int(match.group(1))
    if kind in {"heading", "title"}:
        return 1
    return None


def _block_text(block: SourceBlock, translations: dict[str, str]) -> str:
    return translations.get(block.id, block.text)


def _anchor(block_id: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", block_id).strip("-.")
    return value or fallback


def _list_html(text: str, ordered: bool) -> str:
    tag = "ol" if ordered else "ul"
    items = [line.strip() for line in text.splitlines() if line.strip()]
    return f"<{tag}>" + "".join(f"<li>{_escaped(item)}</li>" for item in items) + f"</{tag}>"


def _table_html(text: str) -> str:
    rows = [line for line in text.splitlines() if line.strip()]
    if not rows:
        return "<table><tbody></tbody></table>"
    delimiter = "\t" if any("\t" in row for row in rows) else "|"
    cells = [[cell.strip() for cell in row.strip("|").split(delimiter)] for row in rows]
    header = "".join(f"<th>{_escaped(cell)}</th>" for cell in cells[0])
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{_escaped(cell)}</td>" for cell in row) + "</tr>"
        for row in cells[1:]
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>"


def _image_html(page: PageResult, image_name: str) -> str:
    descriptions = page.illustrations or page.image_descriptions
    alt = descriptions[0] if descriptions else f"Illustration from page {page.page_number}"
    return f'<img src="../images/{_escaped(image_name)}" alt="{_escaped(alt)}"/>'


def _render_fragment(fragment: _PageFragment, image_name: str | None) -> tuple[str, list[tuple[int, str, str]]]:
    """Render one source page and return its XHTML plus h2/h3 TOC entries."""
    page = fragment.page
    translations = {item.block_id: item.translated_text for item in page.translated_blocks}
    headings: list[tuple[int, str, str]] = []
    parts: list[str] = []
    image_emitted = False

    for index, block in enumerate(sorted(fragment.blocks, key=lambda item: item.reading_order), start=1):
        text = _block_text(block, translations)
        kind = _block_kind(block)
        level = _heading_level(block)
        if level is not None:
            anchor = _anchor(block.id, f"heading-{page.page_number}-{index}")
            parts.append(f'<h{level} id="{anchor}">{_escaped(text)}</h{level}>')
            if level > 1:
                headings.append((level, anchor, text))
        elif kind in {"ordered_list", "orderedlist", "ol", "numbered_list"}:
            parts.append(_list_html(text, ordered=True))
        elif kind in {"unordered_list", "unorderedlist", "ul", "bullet_list", "list"}:
            parts.append(_list_html(text, ordered=False))
        elif kind == "table":
            parts.append(_table_html(text))
        elif kind in {"caption", "figure_caption"}:
            caption = f"<figcaption>{_escaped(text)}</figcaption>"
            if image_name is not None and not image_emitted:
                parts.append(f"<figure>{_image_html(page, image_name)}{caption}</figure>")
                image_emitted = True
            else:
                parts.append(f"<figure>{caption}</figure>")
        elif kind in {"footnote", "note"}:
            parts.append(f'<aside epub:type="footnote" role="doc-footnote">{_escaped(text)}</aside>')
        else:
            parts.append(f"<p>{_escaped(text)}</p>")

    if image_name is not None and not image_emitted:
        parts.append(f"<figure>{_image_html(page, image_name)}</figure>")
    return "\n".join(parts), headings


def _reconstruct_chapters(pages: list[PageResult]) -> list[_Chapter]:
    """Start a new chapter at each level-one heading; legacy pages stand alone."""
    chapters: list[_Chapter] = []
    current: _Chapter | None = None
    for page in sorted(pages, key=lambda item: item.page_number):
        blocks = sorted(page.blocks, key=lambda item: item.reading_order)
        if not blocks:
            chapters.append(_Chapter(f"Page {page.page_number}", [_PageFragment(page, [])]))
            current = None
            continue
        heading = next((block for block in blocks if _heading_level(block) == 1), None)
        translations = {item.block_id: item.translated_text for item in page.translated_blocks}
        if heading is not None:
            current = _Chapter(_block_text(heading, translations))
            chapters.append(current)
        elif current is None:
            current = _Chapter(f"Page {page.page_number}")
            chapters.append(current)
        current.fragments.append(_PageFragment(page, blocks))
    return chapters


def _toc_children(chapter: _SemanticChapter, headings: list[tuple[int, str, str]]) -> list:
    children: list = []
    latest_h2: epub.Link | None = None
    for level, anchor, title in headings:
        link = epub.Link(f"{chapter.file_name}#{anchor}", title, uid=f"toc-{chapter.id}-{anchor}")
        if level == 2:
            latest_h2 = link
            children.append(link)
        elif latest_h2 is not None:
            if children[-1] is latest_h2:
                children[-1] = (latest_h2, [link])
            else:
                children[-1][1].append(link)
        else:
            children.append(link)
    return children


def build_epub(
    page_results: list[PageResult],
    output_path: Path,
    title: str = "Translated Book",
    author: str = "Unknown",
    source_lang: str = "en",
    target_lang: str = "en",
    embed_images: bool = False,
) -> None:
    """Build an EPUB 3 from translated structural PageResult blocks."""
    if not page_results:
        raise ValueError("page_results must not be empty")

    book = epub.EpubBook()
    book.set_identifier("btran-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"))
    book.set_title(title)
    book.add_author(author)
    book.set_language(target_lang)

    style = epub.EpubItem(
        uid="style",
        file_name="style/default.css",
        media_type="text/css",
        content=CSS.encode("utf-8"),
    )
    book.add_item(style)

    image_names: dict[int, str] = {}
    if embed_images:
        for page in sorted(page_results, key=lambda item: item.page_number):
            image_path = Path(page.image_path) if page.image_path else None
            if image_path is None or not image_path.is_file():
                continue
            media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            image_name = image_path.name
            image_names[page.page_number] = image_name
            book.add_item(
                epub.EpubItem(
                    uid=f"image-{page.page_number}",
                    file_name=f"images/{image_name}",
                    media_type=media_type,
                    content=image_path.read_bytes(),
                )
            )

    chapters: list[_SemanticChapter] = []
    toc: list = []
    for number, chapter_data in enumerate(_reconstruct_chapters(page_results), start=1):
        body_parts: list[str] = []
        toc_headings: list[tuple[int, str, str]] = []
        for fragment in chapter_data.fragments:
            if fragment.blocks:
                content, headings = _render_fragment(
                    fragment, image_names.get(fragment.page.page_number)
                )
                body_parts.append(content)
                toc_headings.extend(headings)
            else:
                body_parts.append(f"<p>{_escaped(fragment.page.page_text)}</p>")
        xhtml = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            'xmlns:epub="http://www.idpf.org/2007/ops" '
            f'xml:lang="{_escaped(target_lang)}" lang="{_escaped(target_lang)}">\n'
            f"<head><title>{_escaped(chapter_data.title)}</title>"
            '<link rel="stylesheet" type="text/css" href="../style/default.css"/></head>\n'
            '<body epub:type="bodymatter" role="document">\n'
            + "\n".join(body_parts)
            + "\n</body>\n</html>"
        )
        chapter = _SemanticChapter(
            uid=f"chapter-{number}",
            file_name=f"text/chapter_{number:03}.xhtml",
            title=chapter_data.title,
            lang=target_lang,
        )
        chapter.content = xhtml.encode("utf-8")
        book.add_item(chapter)
        chapters.append(chapter)
        children = _toc_children(chapter, toc_headings)
        toc.append((chapter, children) if children else chapter)

    book.toc = toc
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav(title="Contents"))
    book.guide = [{"type": "text", "item": chapters[0]}]
    epub.write_epub(str(output_path), book, {"epub3_landmark": True})
