"""Compile PageResult JSON files into an EPUB."""

import mimetypes
from pathlib import Path

from ebooklib import epub

from btran.schema import PageResult

CSS = """
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.7;
    margin: 1em 2em;
    max-width: 40em;
}
h1 {
    font-size: 1.4em;
    margin-bottom: 0.5em;
}
h2 {
    font-size: 1.1em;
    color: #555;
}
hr {
    margin: 1.5em 0;
    border: none;
    border-top: 1px solid #ccc;
}
img {
    max-width: 100%;
}
.page-image {
    text-align: center;
    margin-bottom: 1em;
}
.original p, .translated p {
    text-align: justify;
}
"""


def _to_html_paragraphs(text: str) -> str:
    """Convert plain text to HTML paragraphs with <br/> for newlines."""
    return text.replace("\n", "<br/>")


def build_epub(
    page_results: list[PageResult],
    output_path: Path,
    title: str = "Translated Book",
    author: str = "Unknown",
    source_lang: str = "en",
    target_lang: str = "en",
    embed_images: bool = False,
) -> None:
    """Build EPUB from sorted page results.

    Pages are sorted by page_number internally so callers don't need to
    pre-sort.
    """
    if not page_results:
        raise ValueError("page_results must not be empty")

    book = epub.EpubBook()
    book.set_identifier("btran-" + title.replace(" ", "-").lower())
    book.set_title(title)
    book.add_author(author)
    book.set_language(target_lang)
    book.add_metadata("DC", "sourceLanguage", source_lang)

    # CSS
    style = epub.EpubItem(
        uid="style",
        file_name="style/default.css",
        media_type="text/css",
        content=CSS.encode("utf-8"),
    )
    book.add_item(style)

    # Sort pages
    sorted_pages = sorted(page_results, key=lambda p: p.page_number)

    # Build chapters
    chapters: list[epub.EpubHtml] = []
    for page in sorted_pages:
        chapter = epub.EpubHtml(
            title=f"Page {page.page_number}",
            file_name=f"page_{page.page_number}.xhtml",
            lang=target_lang,
        )

        orig_html = _to_html_paragraphs(page.page_text)
        trans_html = _to_html_paragraphs(page.translated_text)

        # Build image tag if embedding
        img_tag = ""
        if embed_images:
            img_path = Path(page.image_path) if page.image_path else None
            if img_path and img_path.exists():
                mime_type, _ = mimetypes.guess_type(str(img_path))
                if mime_type is None:
                    mime_type = "image/jpeg"

                with open(img_path, "rb") as f:
                    img_data = f.read()

                img_filename = f"images/{img_path.name}"
                img_item = epub.EpubItem(
                    uid=f"img_page_{page.page_number}",
                    file_name=img_filename,
                    media_type=mime_type,
                    content=img_data,
                )
                book.add_item(img_item)
                img_tag = (
                    f'<div class="page-image">'
                    f'<img src="../{img_filename}" alt="Page {page.page_number} image"/>'
                    f"</div>\n"
                )

        content = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{target_lang}">\n'
            f"<head><title>Page {page.page_number}</title></head>\n"
            "<body>\n"
            f"{img_tag}"
            f"<h1>{target_lang} — Page {page.page_number}</h1>\n"
            '<div class="original">\n'
            f"<h2>Original ({source_lang})</h2>\n"
            f"<p>{orig_html}</p>\n"
            "</div>\n"
            "<hr/>\n"
            '<div class="translated">\n'
            f"<h2>{target_lang}</h2>\n"
            f"<p>{trans_html}</p>\n"
            "</div>\n"
            "</body>\n"
            "</html>"
        )

        chapter.content = content.encode("utf-8")
        chapter.add_item(style)
        book.add_item(chapter)
        chapters.append(chapter)

    # TOC (flat list)
    book.toc = chapters

    # Spine
    book.spine = ["nav"] + chapters

    # NCX and NAV
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Write
    epub.write_epub(str(output_path), book, {})
