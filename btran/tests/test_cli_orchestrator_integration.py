"""CLI mode integration against immutable executor; external model leaves mocked."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

from btran.artifacts import RevisionStore
from btran.cli import main
from btran.schema import PageExtraction, SourceBlock


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("BTRAN_"):
            monkeypatch.delenv(name, raising=False)


def _image(path: Path, color: tuple[int, int, int] = (1, 2, 3)) -> None:
    Image.new("RGB", (600, 600), color).save(path)


def _extraction(page: int, path: Path, *, language: str, text: str) -> PageExtraction:
    return PageExtraction(
        page, str(path), "a" * 64, f"{page:016x}", language, "test-model",
        blocks=[SourceBlock(f"page_{page}_block_0", "paragraph", text, 0)],
    )


def _argv(input_dir: Path, output: Path, work: Path, *options: str) -> list[str]:
    return ["btran", str(input_dir), str(output), "--workspace", str(work), *options]


def _epub_text(path: Path, suffix: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return next(archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith(suffix))


def test_cli_native_mode_preserves_mixed_detected_languages_without_model_leaves(tmp_path: Path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    japanese, english = input_dir / "a.png", input_dir / "b.png"
    _image(japanese); _image(english, (4, 5, 6))
    output, work = tmp_path / "book.epub", tmp_path / "work"
    sources = AsyncMock(side_effect=[
        _extraction(1, japanese, language="ja", text="日本語"),
        _extraction(2, english, language="en", text="English"),
    ])
    terminology_model = Mock(side_effect=AssertionError("native mode must not build terminology model call"))
    translation_model = AsyncMock(side_effect=AssertionError("native mode must not translate"))

    with patch.object(sys, "argv", _argv(input_dir, output, work)), \
         patch("btran.source_extractor.extract_page", sources), \
         patch("btran.orchestrator.make_pi_consolidation_call", terminology_model), \
         patch("btran.translator._pi_json", translation_model):
        main()

    assert sources.await_count == 2
    terminology_model.assert_not_called()
    translation_model.assert_not_awaited()
    assert "btran mode=native" in capsys.readouterr().out
    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        chapters = "\n".join(archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith(".xhtml"))
    package = _epub_text(output, ".opf")
    assert "日本語" in chapters and "English" in chapters
    assert 'lang="ja" xml:lang="ja"' in chapters
    assert 'lang="en" xml:lang="en"' in chapters
    assert "<dc:language>en</dc:language>" in package
    assert "<dc:language>ja</dc:language>" in package


def test_cli_explicit_target_selects_translated_mode(tmp_path: Path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = input_dir / "page.png"
    _image(image)
    output, work = tmp_path / "book.epub", tmp_path / "work"
    source = AsyncMock(return_value=_extraction(1, image, language="ja", text="島"))
    terminology_model = Mock(return_value=lambda _: "{}")
    translation_model = AsyncMock(return_value={"translated_text": "Island"})

    with patch.object(sys, "argv", _argv(input_dir, output, work, "--target-lang", "en")), \
         patch("btran.source_extractor.extract_page", source), \
         patch("btran.orchestrator.make_pi_consolidation_call", terminology_model), \
         patch("btran.translator._pi_json", translation_model):
        main()

    assert source.await_count == 1
    terminology_model.assert_called_once()
    translation_model.assert_awaited_once()
    assert "btran mode=translated" in capsys.readouterr().out
    chapter = _epub_text(output, ".xhtml")
    package = _epub_text(output, ".opf")
    assert "Island" in chapter and "島" not in chapter
    assert 'lang="en" xml:lang="en"' in chapter
    assert "<dc:language>en</dc:language>" in package


def test_cli_refresh_keeps_active_revision_and_retains_unactivated_candidate(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = input_dir / "page.png"
    _image(image)
    output, work = tmp_path / "book.epub", tmp_path / "work"
    source = AsyncMock(return_value=_extraction(1, image, language="en", text="source"))

    with patch.object(sys, "argv", _argv(input_dir, output, work)), \
         patch("btran.source_extractor.extract_page", source):
        main()
    revisions = RevisionStore(work)
    initial_id = next(item.name for item in revisions.revisions_dir.iterdir() if item.is_dir())
    revisions.activate(initial_id)

    with patch.object(sys, "argv", _argv(input_dir, output, work, "--refresh")), \
         patch("btran.source_extractor.extract_page", source):
        main()

    assert revisions.active_snapshot().revision_id == initial_id
    candidate_ids = {item.name for item in revisions.revisions_dir.iterdir() if item.is_dir()}
    assert len(candidate_ids) == 2
    refreshed_id = next(item for item in candidate_ids if item != initial_id)
    snapshot = revisions.snapshot(refreshed_id)
    from btran.artifacts import ArtifactStore
    artifacts = ArtifactStore(work)
    attempts = [artifacts.get(item) for item in snapshot.selected_artifact_ids
                if artifacts.get(item).kind == "RefreshAttempt"]
    assert len(attempts) == 1
    assert attempts[0].payload["reachable_artifact_ids"]
