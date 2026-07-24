"""Tests for epub_builder.py — TDD."""

import tempfile
from pathlib import Path

import ebooklib
import pytest
from ebooklib import epub

from btran.epub_builder import build_epub
from btran.schema import PageResult


def make_page(page_number: int, page_text: str = "", translated_text: str = "") -> PageResult:
    """Helper to create a synthetic PageResult with minimal required fields."""
    return PageResult(
        page_number=page_number,
        image_path="",
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="en",
        target_lang="en",
        page_text=page_text,
        translated_text=translated_text,
    )


# ---------------------------------------------------------------------------
# 1. Build EPUB from 2 synthetic PageResults → verify file exists and valid
# ---------------------------------------------------------------------------
def test_build_epub_creates_valid_file():
    pages = [
        make_page(1, "Hello world.", "Hola mundo."),
        make_page(2, "Goodbye.", "Adiós."),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.epub"
        build_epub(pages, out, title="Test Book", author="Tester")

        assert out.exists()
        assert out.stat().st_size > 0

        # Basic EPUB validity: must be a ZIP with META-INF/container.xml
        import zipfile

        with zipfile.ZipFile(out, "r") as zf:
            names = zf.namelist()
            assert "META-INF/container.xml" in names


# ---------------------------------------------------------------------------
# 2. Round-trip: open EPUB with ebooklib and verify metadata
# ---------------------------------------------------------------------------
def test_build_epub_roundtrip_metadata():
    pages = [
        make_page(1, "Original 1", "Translated 1"),
        make_page(2, "Original 2", "Translated 2"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "roundtrip.epub"
        build_epub(pages, out, title="My Title", author="My Author")

        book = epub.read_epub(str(out))
        assert book.get_metadata("DC", "title")[0][0] == "My Title"
        assert book.get_metadata("DC", "creator")[0][0] == "My Author"


# ---------------------------------------------------------------------------
# 3. Chapters appear in page_number order
# ---------------------------------------------------------------------------
def test_chapter_order():
    pages = [
        make_page(3, "Third", "Dritter"),
        make_page(1, "First", "Erster"),
        make_page(2, "Second", "Zweiter"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "order.epub"
        build_epub(pages, out)

        book = epub.read_epub(str(out))
        # Get document items (chapters) in spine order
        spine_ids = [itemref[0] for itemref in book.spine if itemref[0] != "nav"]
        chapter_items = [book.get_item_with_id(itemid) for itemid in spine_ids]

        # Each chapter title should be "en — Page N" in order 1, 2, 3
        for idx, item in enumerate(chapter_items):
            content = item.get_content().decode("utf-8")
            expected_page = idx + 1
            assert f"Page {expected_page}" in content, (
                f"Expected Page {expected_page} at position {idx}"
            )


# ---------------------------------------------------------------------------
# 4. Empty page_results → raises ValueError
# ---------------------------------------------------------------------------
def test_empty_pages_raises_valueerror():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "empty.epub"
        with pytest.raises(ValueError, match="page_results"):
            build_epub([], out)


# ---------------------------------------------------------------------------
# 5. Content verification: original and translated text present
# ---------------------------------------------------------------------------
def test_chapter_content_contains_both_texts():
    pages = [
        make_page(1, "Original text\nline two", "Translated text\nline two"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "content.epub"
        build_epub(pages, out)

        book = epub.read_epub(str(out))
        spine_ids = [itemref[0] for itemref in book.spine if itemref[0] != "nav"]
        item = book.get_item_with_id(spine_ids[0])
        html = item.get_content().decode("utf-8")

        # Original text converted with <br/>
        assert "Original text<br/>" in html
        # Translated text converted with <br/>
        assert "Translated text<br/>" in html
        # Horizontal rule separator
        assert "<hr/>" in html
        # Original header with source_lang
        assert "Original (en)" in html
        # Translated header with target_lang
        assert "<h2>en</h2>" in html


# ---------------------------------------------------------------------------
# 6. Test with custom source_lang / target_lang
# ---------------------------------------------------------------------------
def test_custom_language_labels():
    pages = [make_page(1, "foo", "bar")]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "lang.epub"
        build_epub(pages, out, source_lang="de", target_lang="fr")

        book = epub.read_epub(str(out))
        spine_ids = [itemref[0] for itemref in book.spine if itemref[0] != "nav"]
        html = book.get_item_with_id(spine_ids[0]).get_content().decode("utf-8")

        assert "Original (de)" in html
        assert "<h2>fr</h2>" in html


# ---------------------------------------------------------------------------
# 7. Image embedding
# ---------------------------------------------------------------------------
def test_embed_image(tmp_path):
    from PIL import Image

    # Create a tiny test image
    img_path = tmp_path / "test_page.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path)

    page = PageResult(
        page_number=1,
        image_path=str(img_path),
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="en",
        target_lang="en",
        page_text="A page with image",
        translated_text="Una página con imagen",
    )
    out = tmp_path / "with_image.epub"
    build_epub([page], out, embed_images=True)

    book = epub.read_epub(str(out))
    image_items = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))
    assert len(image_items) > 0

    spine_ids = [itemref[0] for itemref in book.spine if itemref[0] != "nav"]
    html = book.get_item_with_id(spine_ids[0]).get_content().decode("utf-8")
    assert "<img" in html


def test_embed_image_skipped_when_disabled(tmp_path):
    from PIL import Image

    img_path = tmp_path / "test_page.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path)

    page = PageResult(
        page_number=1,
        image_path=str(img_path),
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="en",
        target_lang="en",
        page_text="A page with image",
        translated_text="Una página con imagen",
    )
    out = tmp_path / "no_image.epub"
    build_epub([page], out, embed_images=False)

    book = epub.read_epub(str(out))
    image_items = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))
    assert len(image_items) == 0


def test_embed_image_missing_file_no_error(tmp_path):
    page = PageResult(
        page_number=1,
        image_path="/nonexistent/image.png",
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="en",
        target_lang="en",
        page_text="A page",
        translated_text="Una página",
    )
    out = tmp_path / "missing_img.epub"
    # Should not raise — just skip the missing image
    build_epub([page], out, embed_images=True)
    assert out.exists()
