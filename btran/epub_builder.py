"""Render translated structural blocks as a standards-compliant EPUB 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import tempfile

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


class EpubCheckError(RuntimeError):
    """Raised when EPUBCheck cannot validate a generated EPUB."""


def _xml_safe(value: str) -> str:
    """Replace XML 1.0-forbidden code points from untrusted input."""
    return "".join(
        char
        if ord(char) in {0x9, 0xA, 0xD}
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
        else "\ufffd"
        for char in value
    )


def _escaped(value: str) -> str:
    return escape(_xml_safe(value), quote=True)


def check_epub(path: Path, executable: str = "epubcheck") -> None:
    """Raise EpubCheckError unless *path* passes the configured EPUBCheck binary."""
    try:
        result = subprocess.run(
            [executable, str(path)], text=True, capture_output=True, check=False
        )
    except OSError as error:
        raise EpubCheckError(f"unable to run EPUBCheck ({executable}): {error}") from error
    if result.returncode:
        output = (result.stdout + result.stderr).strip()
        raise EpubCheckError(output or f"EPUBCheck exited with status {result.returncode}")


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


def _translation_map(page: PageResult) -> dict[str, str]:
    """Join translations to source blocks by ID, preserving first duplicate deterministically."""
    translations: dict[str, str] = {}
    for translated in page.translated_blocks:
        translations.setdefault(translated.block_id, translated.translated_text)
    return translations


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


def _figure_html(page: PageResult, image_name: str | None, caption: str) -> str:
    image = _image_html(page, image_name) if image_name is not None else ""
    return f"<figure>{image}<figcaption>{_escaped(caption)}</figcaption></figure>"


def _render_fragment(fragment: _PageFragment, image_name: str | None) -> tuple[str, list[tuple[int, str, str]]]:
    """Render one source page and return its XHTML plus h2/h3 TOC entries."""
    page = fragment.page
    translations = _translation_map(page)
    headings: list[tuple[int, str, str]] = []
    parts: list[str] = []
    image_emitted = False
    blocks = sorted(fragment.blocks, key=lambda item: item.reading_order)
    index = 0

    while index < len(blocks):
        block = blocks[index]
        text = _block_text(block, translations)
        kind = _block_kind(block)
        level = _heading_level(block)
        if kind == "list_item":
            items: list[str] = []
            while index < len(blocks) and _block_kind(blocks[index]) == "list_item":
                item = blocks[index]
                items.append(f"<li>{_escaped(_block_text(item, translations))}</li>")
                index += 1
            parts.append("<ul>" + "".join(items) + "</ul>")
            continue
        if level is not None:
            anchor = _anchor(block.id, f"heading-{page.page_number}-{index + 1}")
            parts.append(f'<h{level} id="{anchor}">{_escaped(text)}</h{level}>')
            if level > 1:
                headings.append((level, anchor, _xml_safe(text)))
        elif kind in {"ordered_list", "orderedlist", "ol", "numbered_list"}:
            parts.append(_list_html(text, ordered=True))
        elif kind in {"unordered_list", "unorderedlist", "ul", "bullet_list", "list"}:
            parts.append(_list_html(text, ordered=False))
        elif kind == "table":
            parts.append(_table_html(text))
        elif kind in {"caption", "figure_caption", "illustration"}:
            if image_name is not None and not image_emitted:
                parts.append(_figure_html(page, image_name, text))
                image_emitted = True
            else:
                parts.append(_figure_html(page, None, text))
        elif kind in {"footnote", "note"}:
            parts.append(f'<aside epub:type="footnote" role="doc-footnote">{_escaped(text)}</aside>')
        elif kind in {"pull_quote", "quote"}:
            parts.append(f"<blockquote>{_escaped(text)}</blockquote>")
        elif kind == "page_number":
            anchor = _anchor(block.id, f"page-{page.page_number}-{index + 1}")
            parts.append(
                f'<span id="{anchor}" epub:type="pagebreak" role="doc-pagebreak" '
                f'aria-label="{_escaped(text)}">{_escaped(text)}</span>'
            )
        else:
            parts.append(f"<p>{_escaped(text)}</p>")
        index += 1

    if image_name is not None and not image_emitted:
        parts.append(f"<figure>{_image_html(page, image_name)}</figure>")
    return "\n".join(parts), headings


def _reconstruct_chapters(pages: list[PageResult]) -> list[_Chapter]:
    """Start a new chapter at every level-one heading; legacy pages stand alone."""
    chapters: list[_Chapter] = []
    current: _Chapter | None = None
    for page in sorted(pages, key=lambda item: item.page_number):
        blocks = sorted(page.blocks, key=lambda item: item.reading_order)
        if not blocks:
            chapters.append(_Chapter(f"Page {page.page_number}", [_PageFragment(page, [])]))
            current = None
            continue

        fragment_blocks: list[SourceBlock] = []
        translations = _translation_map(page)
        for block in blocks:
            if _heading_level(block) == 1:
                if fragment_blocks:
                    if current is None:
                        current = _Chapter(f"Page {page.page_number}")
                        chapters.append(current)
                    current.fragments.append(_PageFragment(page, fragment_blocks))
                    fragment_blocks = []
                current = _Chapter(_xml_safe(_block_text(block, translations)))
                chapters.append(current)
            fragment_blocks.append(block)

        if current is None:
            current = _Chapter(f"Page {page.page_number}")
            chapters.append(current)
        current.fragments.append(_PageFragment(page, fragment_blocks))
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
    epub_check: bool = False,
    epub_check_path: str = "epubcheck",
) -> None:
    """Build an EPUB 3 from translated structural PageResult blocks."""
    if not page_results:
        raise ValueError("page_results must not be empty")

    safe_title = _xml_safe(title)
    safe_author = _xml_safe(author)
    safe_target_lang = _xml_safe(target_lang)
    book = epub.EpubBook()
    book.set_identifier("btran-" + re.sub(r"[^a-z0-9]+", "-", safe_title.lower()).strip("-"))
    book.set_title(safe_title)
    book.add_author(safe_author)
    book.set_language(safe_target_lang)

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
                if fragment.page.translated_text:
                    body_parts.append(f"<p>{_escaped(fragment.page.translated_text)}</p>")
        xhtml = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            'xmlns:epub="http://www.idpf.org/2007/ops" '
            f'xml:lang="{_escaped(safe_target_lang)}" lang="{_escaped(safe_target_lang)}">\n'
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
            lang=safe_target_lang,
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, suffix=".epub", delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
        epub.write_epub(str(temp_path), book, {"epub3_landmark": True})
        if epub_check:
            check_epub(temp_path, epub_check_path)
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
