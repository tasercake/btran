"""Duplicate raw inputs stay one logical workload but render every placement."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.artifacts import ArtifactStore, RevisionStore
from btran.config import Config
from btran.orchestrator import run
from btran.schema import PageExtraction, SourceBlock
from btran.terminology import PiConsolidationError


def _config(root: Path, *, target_lang: str | None = None) -> Config:
    return Config(input_dir=root / "input", output_epub=root / "book.epub", workspace=root / "work",
                  target_lang=target_lang, timeout=1, max_retries=0)


def _duplicate_inputs(root: Path) -> None:
    root.mkdir()
    image = root / "a.png"
    Image.new("RGB", (600, 600), (1, 2, 3)).save(image)
    shutil.copyfile(image, root / "b.png")


def _source(path: Path, number: int) -> PageExtraction:
    return PageExtraction(number, str(path), "a" * 64, "b" * 16, "en", "test",
                          blocks=[SourceBlock("model-block", "paragraph", "same source", 0)])


def _epub_placements(path: Path) -> tuple[list[str], dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        chapters = [archive.read(name).decode("utf-8") for name in archive.namelist()
                    if name.startswith("EPUB/text/page_") and name.endswith(".xhtml")]
        provenance = json.loads(archive.read("META-INF/btran-provenance.json"))
    return chapters, provenance


def _assert_unique_stage_sets(result) -> None:
    assert result.report is not None
    for record in result.report.stage_records:
        assert record.input_artifact_ids == tuple(sorted(set(record.input_artifact_ids)))
        assert record.output_artifact_ids == tuple(sorted(set(record.output_artifact_ids)))


def _stage_artifact_id(result, stage: str) -> str:
    assert result.report is not None
    record = next(record for record in result.report.stage_records if record.stage == stage)
    assert len(record.output_artifact_ids) == 1
    return record.output_artifact_ids[0]


@pytest.mark.asyncio
async def test_native_duplicate_placements_reuse_one_logical_page_and_survive_rename_reorder(tmp_path: Path):
    config = _config(tmp_path)
    _duplicate_inputs(config.input_dir)
    extraction = AsyncMock(side_effect=lambda path, _model, _sha, _phash, number, **_: _source(Path(path), number))
    with patch("btran.source_extractor.extract_page", extraction):
        first = await run(config)

    assert first.status == "completed"
    assert extraction.await_count == 1
    assert first.target_run is not None and len(first.target_run.leaves) == 1
    _assert_unique_stage_sets(first)
    chapters, provenance = _epub_placements(config.output_epub)
    assert len(chapters) == 2
    assert all('data-placement-id=' in chapter for chapter in chapters)
    assert len(provenance["placements"]) == 2
    assert first.report is not None and tuple(provenance["placements"]) == first.report.placement_provenance
    assert {item["page_id"] for item in provenance["placements"]} == {first.target_run.leaves[0].page_id}
    assert len({item["effective_page_artifact_id"] for item in provenance["placements"]}) == 1
    assert all(item["effective_segment_artifact_ids"] for item in provenance["placements"])

    RevisionStore(config.workspace).activate(first.candidate_revision_id)
    (config.input_dir / "a.png").rename(config.input_dir / "z.png")
    (config.input_dir / "b.png").rename(config.input_dir / "a.png")
    reused = AsyncMock(side_effect=AssertionError("rename/reorder must not re-extract"))
    with patch("btran.source_extractor.extract_page", reused):
        second = await run(config)

    assert second.status == "completed"
    assert reused.await_count == 0
    assert second.target_run is not None
    assert second.target_run.leaves[0].page_artifact_id == first.target_run.leaves[0].page_artifact_id
    assert [item.placement_id for item in second.placements] != [item.placement_id for item in first.placements]


@pytest.mark.asyncio
async def test_translated_duplicate_placements_translate_once_and_render_two_chapters(tmp_path: Path):
    config = _config(tmp_path, target_lang="fr")
    _duplicate_inputs(config.input_dir)
    extraction = AsyncMock(side_effect=lambda path, _model, _sha, _phash, number, **_: _source(Path(path), number))
    translation = AsyncMock(return_value="meme cible")
    with patch("btran.source_extractor.extract_page", extraction), \
         patch("btran.translator.translate_segment", translation):
        result = await run(config)

    assert result.status == "completed"
    assert extraction.await_count == 1
    assert translation.await_count == 1
    assert result.target_run is not None and len(result.target_run.leaves) == 1
    _assert_unique_stage_sets(result)
    chapters, provenance = _epub_placements(config.output_epub)
    assert len(chapters) == 2
    assert all("meme cible" in chapter and 'lang="fr"' in chapter for chapter in chapters)
    assert len(provenance["placements"]) == 2
    assert len(provenance["segments"]) == 1


@pytest.mark.asyncio
async def test_translated_duplicate_placement_rename_reorder_reuses_fallback_projection_and_translation(tmp_path: Path):
    config = _config(tmp_path, target_lang="fr")
    _duplicate_inputs(config.input_dir)
    extraction = AsyncMock(side_effect=lambda path, _model, _sha, _phash, number, **_: _source(Path(path), number))
    translated = AsyncMock(return_value="meme cible")

    def fallback_consolidation(_: str) -> str:
        raise PiConsolidationError("forced fallback")

    with patch("btran.source_extractor.extract_page", extraction), \
         patch("btran.translator.translate_segment", translated), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=fallback_consolidation):
        first = await run(config)

    assert first.status == "completed"
    assert extraction.await_count == translated.await_count == 1
    assert first.terminology_run is not None and first.target_run is not None
    first_projection_ids = first.terminology_run.projection_artifact_ids
    first_target_page_id = first.target_run.leaves[0].page_artifact_id
    first_reconciliation_id = _stage_artifact_id(first, "reconciliation")
    first_validation_id = _stage_artifact_id(first, "validation")
    store = ArtifactStore(config.workspace)
    assert first.report is not None
    first_target_reviews = tuple(
        finding_id for finding_id in first.report.review_finding_ids
        if store.get_finding(finding_id).stage == "target_materialization"
    )
    assert first_target_reviews
    assert any(store.get_finding(finding_id).evidence.get("base_revision_id") == "unsealed"
               for finding_id in first_target_reviews)
    # Exact base-revision review evidence remains report-visible, but not part
    # of immutable effective-target page content.
    assert not set(first_target_reviews) & set(store.get(first_target_page_id).finding_ids)
    RevisionStore(config.workspace).activate(first.candidate_revision_id)
    (config.input_dir / "a.png").rename(config.input_dir / "z.png")
    (config.input_dir / "b.png").rename(config.input_dir / "a.png")

    no_extract = AsyncMock(side_effect=AssertionError("rename/reorder must not re-extract"))
    no_translate = AsyncMock(side_effect=AssertionError("rename/reorder must not translate"))
    with patch("btran.source_extractor.extract_page", no_extract), \
         patch("btran.translator.translate_segment", no_translate), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=fallback_consolidation):
        second = await run(config)

    assert second.status == "completed"
    assert no_extract.await_count == no_translate.await_count == 0
    assert second.terminology_run is not None
    assert second.terminology_run.projection_artifact_ids == first_projection_ids
    assert second.target_run is not None and first.target_run is not None
    assert second.target_run.leaves[0].translation_artifact_ids == first.target_run.leaves[0].translation_artifact_ids
    # Activation/reordered duplicate placements alter review provenance only;
    # semantic target/reconciliation/validation closures must stay cache-stable.
    assert second.target_run.leaves[0].page_artifact_id == first_target_page_id
    assert _stage_artifact_id(second, "reconciliation") == first_reconciliation_id
    assert _stage_artifact_id(second, "validation") == first_validation_id
    assert second.report is not None
    second_target_reviews = tuple(
        finding_id for finding_id in second.report.review_finding_ids
        if store.get_finding(finding_id).stage == "target_materialization"
    )
    assert second_target_reviews
    assert any(store.get_finding(finding_id).evidence.get("base_revision_id") == first.candidate_revision_id
               for finding_id in second_target_reviews)
    assert [item.placement_id for item in second.placements] != [item.placement_id for item in first.placements]
