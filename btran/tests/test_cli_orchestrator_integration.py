"""CLI integration tests against the merged orchestrator; model leaves are mocked."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.cli import main
from btran.schema import PageExtraction, SourceBlock, TermMention, TranslatedBlock


def _image(path: Path, color: tuple[int, int, int] = (1, 2, 3)) -> None:
    Image.new("RGB", (600, 600), color).save(path)


def _extraction(page: int, path: Path, term: str = "") -> PageExtraction:
    block = SourceBlock(f"page_{page}_block_0", "paragraph", f"source {term}".strip(), 0)
    return PageExtraction(
        page, str(path), "a" * 64, f"{page:016x}", "ja", "test-model",
        blocks=[block], term_mentions=[TermMention(term, block.id)] if term else [],
    )


def _translations(source: PageExtraction) -> list[TranslatedBlock]:
    return [TranslatedBlock(block.id, "translation island") for block in source.blocks]


def _argv(input_dir: Path, output: Path, work: Path, *options: str) -> list[str]:
    return ["btran", str(input_dir), str(output), "--target-lang", "en", "--intermediate-dir", str(work), *options]


def test_cli_runs_merged_orchestrator_with_default_manifest_budget_and_epubcheck(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_path = input_dir / "page.png"
    _image(source_path)
    source = _extraction(1, source_path, "島")
    captured: dict[str, object] = {}

    def consolidate(mentions, *, source_lang, target_lang, pi_call, token_budget):
        captured.update(mentions=mentions, source_lang=source_lang, target_lang=target_lang, token_budget=token_budget)
        from btran.terminology import freeze_terminology
        from btran.schema import TerminologyEntry
        return freeze_terminology(
            [TerminologyEntry("island", ["島"], "island", [source.blocks[0].id], 1.0)],
            source_lang=source_lang,
            target_lang=target_lang,
        )

    with patch.object(sys, "argv", _argv(input_dir, tmp_path / "book.epub", tmp_path / "work", "--glossary-budget", "99999", "--epub-check", "--epub-check-path", "strict-check")), \
         patch("btran.cli.shutil.which", return_value="/usr/bin/pi"), \
         patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)) as extract, \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=lambda _: "{}"), \
         patch("btran.orchestrator.consolidate_terminology", side_effect=consolidate), \
         patch("btran.orchestrator.translate_blocks", AsyncMock(return_value=_translations(source))), \
         patch("btran.orchestrator.build_epub") as build:
        main()

    assert extract.await_count == 1
    assert captured["source_lang"] == "ja"
    assert captured["token_budget"] == 99_999
    assert (input_dir / "manifest.json").is_file()
    assert build.call_args.kwargs["epub_check"] is True
    assert build.call_args.kwargs["epub_check_path"] == "strict-check"


def test_cli_uses_explicit_manifest_and_streams_one_terminal_page_error(tmp_path: Path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    ignored = input_dir / "ignored.png"
    selected = input_dir / "selected.png"
    _image(ignored)
    _image(selected)
    manifest = tmp_path / "selected.json"
    manifest.write_text(json.dumps({
        "input_dir": str(input_dir),
        "pages": [{"filename": "selected.png", "page_number": 1, "status": "pending"}],
        "total_pages": 1,
    }))

    with patch.object(sys, "argv", _argv(input_dir, tmp_path / "book.epub", tmp_path / "work", "--max-retries", "1", "--manifest-path", str(manifest))), \
         patch("btran.cli.shutil.which", return_value="/usr/bin/pi"), \
         patch("btran.orchestrator.extract_page", AsyncMock(side_effect=RuntimeError("OCR unavailable"))) as extract:
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    assert extract.await_count == 1
    captured = capsys.readouterr()
    assert "btran — auto-detecting source languages → en" in captured.out
    assert "translating 2 images" not in captured.out
    stderr = captured.err
    assert stderr.count("[btran] page 1 failed: RuntimeError: OCR unavailable") == 1
    assert "1 page(s) failed — no EPUB produced." in stderr


def test_cli_mandatory_preflight_blocks_model_leaves(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _image(input_dir / "tiny.png")
    Image.new("RGB", (10, 10)).save(input_dir / "tiny.png")

    with patch.object(sys, "argv", _argv(input_dir, tmp_path / "book.epub", tmp_path / "work")), \
         patch("btran.cli.shutil.which", return_value="/usr/bin/pi"), \
         patch("btran.orchestrator.extract_page", AsyncMock()) as extract:
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    extract.assert_not_awaited()
