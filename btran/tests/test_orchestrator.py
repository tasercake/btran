"""Task 13 core-executor contracts (finalization belongs to Task 14)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
import zipfile

import pytest
from PIL import Image

from btran.artifacts import ArtifactStore
from btran.config import Config
from btran.orchestrator import _atomic_write, _with_retries, orchestrator_run, run
from btran.schema import PageExtraction, SourceBlock


def _config(tmp_path: Path) -> Config:
    return Config(input_dir=tmp_path / "in", output_epub=tmp_path / "out.epub",
                  workspace=tmp_path / "work", target_lang=None, max_retries=1)


def _image(path: Path, color: tuple[int, int, int] = (1, 2, 3)) -> None:
    Image.new("RGB", (600, 600), color).save(path)


def _source(path: Path, number: int = 1) -> PageExtraction:
    return PageExtraction(number, str(path), "a" * 64, "b" * 16, "en", "vision",
                          blocks=[SourceBlock(f"page_{number}_block", "paragraph", "source text", 0)])


def test_atomic_write_replaces_complete_checkpoint_without_temp_residue(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    _atomic_write(path, "first")
    _atomic_write(path, "second")
    assert path.read_text() == "second"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_retry_count_excludes_the_initial_attempt():
    attempts = 0

    async def action() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("retry")
        return "done"

    assert await _with_retries(action, retries=2) == "done"
    assert attempts == 3


@pytest.mark.asyncio
async def test_zero_retries_still_makes_the_initial_attempt():
    action = AsyncMock(side_effect=RuntimeError("failed"))
    with pytest.raises(RuntimeError, match="failed"):
        await _with_retries(action, retries=0)
    action.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_executor_persists_stage_records_and_finalizes_in_fixed_order(tmp_path: Path):
    cfg = _config(tmp_path)
    cfg.input_dir.mkdir()
    _image(cfg.input_dir / "z.png", (2, 3, 4))
    _image(cfg.input_dir / "a.png", (1, 2, 3))
    sources = [_source(cfg.input_dir / "a.png", 1), _source(cfg.input_dir / "z.png", 2)]
    with patch("btran.source_extractor.extract_page", AsyncMock(side_effect=sources)):
        result = await orchestrator_run(cfg)

    assert result.status == "completed"
    assert result.errors == []
    assert result.report is not None
    assert [record.stage for record in result.report.stage_records] == [
        "discovery", "source_extraction", "corrections",
        "effective_source", "terminology", "target_materialization",
        "reconciliation", "validation", "rendering", "candidate_seal",
    ]
    assert all(record.stage_summary_finding_id in record.finding_ids for record in result.report.stage_records)
    assert result.report.final_epub_status == "completed"
    assert cfg.output_epub.exists()
    assert len(result.provenance) == 2
    assert all(item.source_artifact_id and item.effective_source_artifact_id and item.effective_target_artifact_id
               for item in result.provenance)
    # Stage records publish only after named immutable inputs/outputs exist.
    assert len(list((cfg.workspace / "artifacts").glob("*.json"))) >= len(result.report.stage_records)


@pytest.mark.asyncio
async def test_degraded_source_leaf_keeps_target_representation_and_still_renders(tmp_path: Path):
    cfg = _config(tmp_path)
    cfg.input_dir.mkdir()
    _image(cfg.input_dir / "a.png")
    cfg.output_epub.write_bytes(b"old")
    reported: list[tuple[int, str]] = []
    with patch("btran.source_extractor.extract_page", AsyncMock(side_effect=RuntimeError("model unavailable"))):
        result = await run(cfg, on_page_error=lambda page, message: reported.append((page, message)))

    assert result.status == "completed"
    assert result.target_run is not None
    assert result.target_run.leaves[0].degraded
    assert all(record.stage != "preflight" for record in result.report.stage_records)
    store = ArtifactStore(cfg.workspace)
    assert all(store.get(path.stem).kind != "PagePreflight" for path in (cfg.workspace / "artifacts").glob("*.json"))
    assert all(store.get_finding(path.stem).stage != "preflight" for path in (cfg.workspace / "findings").glob("*.json"))
    assert reported and reported[0][0] == 1
    assert cfg.output_epub.read_bytes() != b"old"
    # Fallback is materialized into same target-page/segment representation and
    # informational findings never gate final rendering.
    target_page = ArtifactStore(cfg.workspace).get(result.target_run.leaves[0].page_artifact_id)
    assert target_page.kind == "EffectiveTargetPage"
    assert result.report is not None
    assert result.report.final_epub_status == "completed"
    assert result.report.recoverable_failure_finding_ids


@pytest.mark.asyncio
async def test_explicit_target_keeps_unreadable_source_as_translated_und_diagnostic(tmp_path: Path):
    cfg = _config(tmp_path)
    cfg = Config(**{**cfg.__dict__, "target_lang": "fr"})
    cfg.input_dir.mkdir()
    (cfg.input_dir / "a-bad.png").write_bytes(b"not an image")
    _image(cfg.input_dir / "z-good.png")
    translated: list[str] = []

    async def extract(path, model, sha256, phash, page_number, **kwargs):
        if Path(path).read_bytes() == b"not an image":
            raise RuntimeError("vision rejected unreadable image")
        return _source(Path(path), page_number)

    async def translate(segment, **kwargs):
        translated.append(segment.segment_id)
        return "bonjour"

    with patch("btran.source_extractor.extract_page", new=extract), \
         patch("btran.translator.translate_segment", new=translate):
        result = await run(cfg)

    assert result.status == "completed_degraded"
    assert result.report is not None and result.report.final_epub_status == "completed_degraded"
    assert len(translated) == 1
    store = ArtifactStore(cfg.workspace)
    assert all(record.stage != "preflight" for record in result.report.stage_records)
    assert all(store.get(path.stem).kind != "PagePreflight" for path in (cfg.workspace / "artifacts").glob("*.json"))
    assert all(store.get_finding(path.stem).stage != "preflight" for path in (cfg.workspace / "findings").glob("*.json"))
    target_segments = [store.get(artifact_id).payload for leaf in result.target_run.leaves
                       for artifact_id in leaf.segment_artifact_ids]
    diagnostic = next(item for item in target_segments if item["source_lang"] is None)
    assert diagnostic["mode"] == "translated"
    assert diagnostic["render_lang"] == "und"
    assert diagnostic["translation_artifact_id"] is None
    with zipfile.ZipFile(cfg.output_epub) as archive:
        chapters = [archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith(".xhtml")]
        package = next(archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith(".opf"))
    assert any('lang="und" xml:lang="und"' in chapter and "source_extraction_failed" in chapter for chapter in chapters)
    assert any('lang="fr" xml:lang="fr"' in chapter and "bonjour" in chapter for chapter in chapters)
    assert "<dc:language>fr</dc:language>" in package
