"""Behavior tests for the semantic EPUB 3 renderer."""

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from btran.epub_builder import build_epub
from btran.schema import PageResult, SourceBlock, TranslatedBlock


EPUB_NS = "http://www.idpf.org/2007/ops"


def make_page(
    page_number: int,
    *,
    page_text: str = "",
    translated_text: str = "",
    blocks: list[SourceBlock] | None = None,
    translated_blocks: list[TranslatedBlock] | None = None,
    image_path: str = "",
    illustrations: list[str] | None = None,
) -> PageResult:
    return PageResult(
        page_number=page_number,
        image_path=image_path,
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="en",
        target_lang="fr",
        page_text=page_text,
        translated_text=translated_text,
        blocks=blocks or [],
        translated_blocks=translated_blocks or [],
        illustrations=illustrations or [],
    )


def block(block_id: str, kind: str, text: str, order: int) -> SourceBlock:
    return SourceBlock(id=block_id, type=kind, text=text, reading_order=order)


def translated(block_id: str, text: str) -> TranslatedBlock:
    return TranslatedBlock(block_id=block_id, translated_text=text)


def epub_files(epub_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(epub_path) as archive:
        return {
            name: archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.endswith(".xhtml")
        }


def chapter_documents(epub_path: Path) -> list[str]:
    return [content for name, content in epub_files(epub_path).items() if "/chapter_" in name]


def test_build_epub_rejects_empty_page_results(tmp_path):
    with pytest.raises(ValueError, match="page_results"):
        build_epub([], tmp_path / "empty.epub")


def test_semantic_blocks_render_as_escaped_xhtml_elements(tmp_path):
    page = make_page(
        1,
        blocks=[
            block("title", "heading_1", "Source title", 1),
            block("section", "heading_2", "Source section", 2),
            block("subsection", "heading_3", "Source subsection", 3),
            block("paragraph", "paragraph", "Source paragraph", 4),
            block("unordered", "unordered_list", "Apple\nPear", 5),
            block("ordered", "ordered_list", "One\nTwo", 6),
            block("table", "table", "Name\tValue\nA & B\t<C>", 7),
            block("caption", "caption", "Source caption", 8),
            block("note", "footnote", "Source note", 9),
        ],
        translated_blocks=[
            translated("title", "Titre & <Livre>"),
            translated("section", "Section"),
            translated("subsection", "Sous-section"),
            translated("paragraph", "Texte & <important>"),
            translated("unordered", "Pomme\nPoire"),
            translated("ordered", "Un\nDeux"),
            translated("table", "Nom\tValeur\nA & B\t<C>"),
            translated("caption", "Légende & <image>"),
            translated("note", "Note & <référence>"),
        ],
    )
    output = tmp_path / "semantic.epub"

    build_epub([page], output, title="Semantic Book", target_lang="fr")

    chapters = chapter_documents(output)
    assert len(chapters) == 1
    chapter = chapters[0]
    ET.fromstring(chapter)  # XHTML must be well-formed XML.
    assert '<h1 id="title">Titre &amp; &lt;Livre&gt;</h1>' in chapter
    assert '<h2 id="section">Section</h2>' in chapter
    assert '<h3 id="subsection">Sous-section</h3>' in chapter
    assert "<p>Texte &amp; &lt;important&gt;</p>" in chapter
    assert "<ul><li>Pomme</li><li>Poire</li></ul>" in chapter
    assert "<ol><li>Un</li><li>Deux</li></ol>" in chapter
    assert "<table>" in chapter
    assert "<th>Nom</th><th>Valeur</th>" in chapter
    assert "<td>A &amp; B</td><td>&lt;C&gt;</td>" in chapter
    assert "<figcaption>Légende &amp; &lt;image&gt;</figcaption>" in chapter
    assert f'<aside {"epub:type"}="footnote" role="doc-footnote">' in chapter
    assert "Note &amp; &lt;référence&gt;" in chapter


def test_heading_one_reconstructs_chapters_and_hierarchical_toc(tmp_path):
    pages = [
        make_page(
            1,
            blocks=[
                block("first", "heading_1", "First chapter", 1),
                block("first-section", "heading_2", "First section", 2),
                block("first-text", "paragraph", "First text", 3),
            ],
        ),
        make_page(
            2,
            blocks=[
                block("second", "heading_1", "Second chapter", 1),
                block("second-section", "heading_2", "Second section", 2),
                block("second-subsection", "heading_3", "Second subsection", 3),
            ],
        ),
    ]
    output = tmp_path / "toc.epub"

    build_epub(pages, output, title="Contents", target_lang="fr")

    files = epub_files(output)
    chapters = chapter_documents(output)
    assert len(chapters) == 2
    nav = next(content for name, content in files.items() if name.endswith("nav.xhtml"))
    ET.fromstring(nav)
    assert f'xmlns:epub="{EPUB_NS}"' in nav
    assert '<nav epub:type="toc"' in nav
    assert 'role="doc-toc"' in nav
    assert "First chapter" in nav and "First section" in nav
    assert "Second chapter" in nav and "Second subsection" in nav
    # h2/h3 entries make nested ordered lists rather than a flat page list.
    assert nav.count("<ol>") >= 3


def test_image_and_caption_share_figure_and_alt_uses_illustration_description(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color="red").save(image_path)
    page = make_page(
        1,
        image_path=str(image_path),
        illustrations=['A "red" & <blue> diagram'],
        blocks=[
            block("heading", "heading_1", "Pictures", 1),
            block("caption", "caption", "A diagram", 2),
        ],
    )
    output = tmp_path / "image.epub"

    build_epub([page], output, embed_images=True, target_lang="fr")

    chapter = chapter_documents(output)[0]
    assert "<figure>" in chapter
    assert '<img src="../images/page.png" alt="A &quot;red&quot; &amp; &lt;blue&gt; diagram"/>' in chapter
    assert "<figcaption>A diagram</figcaption>" in chapter
    assert chapter.index("<img") < chapter.index("<figcaption")


def test_document_has_language_and_bodymatter_landmark_roles(tmp_path):
    output = tmp_path / "metadata.epub"
    build_epub([make_page(1, blocks=[block("h", "heading_1", "Chapter", 1)])], output, target_lang="fr")

    chapter = chapter_documents(output)[0]
    assert '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="fr" lang="fr">' in chapter
    assert '<body epub:type="bodymatter" role="document">' in chapter
    nav = next(content for name, content in epub_files(output).items() if name.endswith("nav.xhtml"))
    assert 'epub:type="landmarks"' in nav
    assert 'epub:type="bodymatter"' in nav


def test_flat_text_fallback_uses_escaped_page_text_when_blocks_are_absent(tmp_path):
    output = tmp_path / "legacy.epub"
    build_epub([make_page(7, page_text="Legacy & <plain> text")], output, target_lang="fr")

    chapter = chapter_documents(output)[0]
    assert "<p>Legacy &amp; &lt;plain&gt; text</p>" in chapter
    assert "translated_text" not in chapter


def test_epubcheck_reports_zero_errors_for_semantic_epub(tmp_path):
    output = tmp_path / "checked.epub"
    build_epub(
        [
            make_page(
                1,
                blocks=[
                    block("h", "heading_1", "Checked", 1),
                    block("section", "heading_2", "Checked section", 2),
                    block("p", "paragraph", "A valid EPUB 3 document.", 3),
                    block("item-one", "list_item", "First item", 4),
                    block("item-two", "list_item", "Second item", 5),
                    block("table", "table", "Name\tValue\nOne\t1", 6),
                    block("caption", "caption", "A figure caption", 7),
                    block("note", "footnote", "A footnote", 8),
                    block("quote", "pull_quote", "A pull quote", 9),
                    block("illustration", "illustration", "A diagram", 10),
                ],
            )
        ],
        output,
        title="EPUBCheck fixture",
        author="btran",
        target_lang="en",
        epub_check=True,
        epub_check_path="/home/exedev/.local/bin/epubcheck",
    )


def test_flat_text_fallback_preserves_legacy_translation(tmp_path):
    output = tmp_path / "legacy-translation.epub"
    build_epub(
        [make_page(1, page_text="Original", translated_text="Translated & <safe>")],
        output,
        target_lang="fr",
    )

    chapter = chapter_documents(output)[0]
    assert "Original" in chapter
    assert "Translated &amp; &lt;safe&gt;" in chapter


def test_supported_blocks_render_semantic_list_quote_and_illustration(tmp_path):
    output = tmp_path / "supported-blocks.epub"
    build_epub(
        [
            make_page(
                1,
                blocks=[
                    block("heading", "heading", "Chapter", 0),
                    block("one", "list_item", "Source one", 1),
                    block("two", "list_item", "Source two", 2),
                    block("quote", "pull_quote", "Source quote", 3),
                    block("illustration", "illustration", "Source diagram", 4),
                ],
                # Join by source block IDs, not translation list position.
                translated_blocks=[
                    translated("illustration", "Translated diagram"),
                    translated("quote", "Translated quote"),
                    translated("two", "Translated two"),
                    translated("heading", "Chapter"),
                    translated("one", "Translated one"),
                ],
            )
        ],
        output,
    )

    chapter = chapter_documents(output)[0]
    assert "<ul><li>Translated one</li><li>Translated two</li></ul>" in chapter
    assert "<blockquote>Translated quote</blockquote>" in chapter
    assert "<figure><figcaption>Translated diagram</figcaption></figure>" in chapter


def test_each_level_one_heading_starts_a_chapter_even_within_one_page(tmp_path):
    output = tmp_path / "two-headings.epub"
    build_epub(
        [
            make_page(
                1,
                blocks=[
                    block("first", "heading_1", "First", 0),
                    block("first-text", "paragraph", "First text", 1),
                    block("second", "heading_1", "Second", 2),
                    block("second-text", "paragraph", "Second text", 3),
                ],
            )
        ],
        output,
    )

    chapters = chapter_documents(output)
    assert len(chapters) == 2
    assert "First text" in chapters[0] and "Second text" not in chapters[0]
    assert "Second text" in chapters[1] and "First text" not in chapters[1]


def test_untrusted_control_characters_are_replaced_before_xhtml_serialization(tmp_path):
    output = tmp_path / "safe-xhtml.epub"
    build_epub(
        [make_page(1, blocks=[block("p", "paragraph", "bad\x08text", 0)])],
        output,
        target_lang="en\x08",
    )

    chapter = chapter_documents(output)[0]
    assert "\x08" not in chapter
    assert "bad\ufffdtext" in chapter
    ET.fromstring(chapter)


def test_untrusted_heading_text_is_safe_in_toc_metadata(tmp_path):
    output = tmp_path / "safe-toc.epub"
    build_epub(
        [make_page(1, blocks=[block("h", "heading", "bad\x08heading", 0)])],
        output,
    )

    documents = epub_files(output)
    assert "bad\ufffdheading" in next(
        content for name, content in documents.items() if name.endswith("nav.xhtml")
    )
    for document in documents.values():
        ET.fromstring(document)


def test_epubcheck_failure_keeps_existing_output_unchanged(tmp_path, monkeypatch):
    output = tmp_path / "existing.epub"
    output.write_bytes(b"known-good-output")

    def fail_check(path, executable):
        assert path != output
        raise RuntimeError("invalid epub")

    monkeypatch.setattr("btran.epub_builder.check_epub", fail_check, raising=False)
    with pytest.raises(RuntimeError, match="invalid epub"):
        build_epub(
            [make_page(1, blocks=[block("h", "heading", "Chapter", 0)])],
            output,
            epub_check=True,
        )

    assert output.read_bytes() == b"known-good-output"
