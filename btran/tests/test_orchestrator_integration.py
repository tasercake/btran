"""Task-14 final-executor integration coverage."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.artifacts import ArtifactStore, RevisionStore
from btran.config import Config
from btran.epub_builder import EpubInvocationError
from btran.orchestrator import run
from btran.schema import PageExtraction, SourceBlock


def _config(tmp_path: Path) -> Config:
    return Config(input_dir=tmp_path / "input", output_epub=tmp_path / "book.epub",
                  workspace=tmp_path / "work", target_lang=None, timeout=1)


def _image(path: Path) -> None:
    Image.new("RGB", (600, 600), (1, 2, 3)).save(path)


def _source(path: Path, text: str = "source text") -> PageExtraction:
    return PageExtraction(1, str(path), "a" * 64, "b" * 16, "en", "vision",
                          blocks=[SourceBlock("page_1_block", "paragraph", text, 0)])


@pytest.mark.asyncio
async def test_final_executor_orders_reconciliation_validation_render_and_unactivated_seal(tmp_path: Path):
    config = _config(tmp_path)
    config.input_dir.mkdir(); _image(config.input_dir / "page.png")
    with patch("btran.source_extractor.extract_page", AsyncMock(return_value=_source(config.input_dir / "page.png"))):
        result = await run(config)

    assert result.status == "completed"
    assert result.report is not None and result.candidate_revision_id
    assert [record.stage for record in result.report.stage_records][-4:] == [
        "reconciliation", "validation", "rendering", "candidate_seal",
    ]
    reconciliation = next(record for record in result.report.stage_records if record.stage == "reconciliation")
    validation = next(record for record in result.report.stage_records if record.stage == "validation")
    assert reconciliation.output_artifact_ids[0] in validation.input_artifact_ids
    assert config.output_epub.is_file()
    assert Path(result.report_path).is_file()
    assert (config.workspace / "revisions" / result.candidate_revision_id / "book.epub").is_file()
    assert not (config.workspace / "active-revision.json").exists()
    assert result.report.final_epub_status == "completed"
    assert len(result.provenance) == 1
    provenance = result.provenance[0]
    assert provenance.segment_id and provenance.page_id
    assert provenance.source_artifact_id == provenance.effective_source_artifact_id
    assert provenance.effective_target_artifact_id
    assert provenance.translation_artifact_id is None


def _selected_artifact(store: ArtifactStore, revision_id: str, kind: str):
    snapshot = RevisionStore(store.root).snapshot(revision_id)
    return next(store.get(artifact_id) for artifact_id in snapshot.selected_artifact_ids
                if store.get(artifact_id).kind == kind)


@pytest.mark.asyncio
async def test_clean_and_incremental_runs_keep_effective_and_render_hashes_and_sealed_closure(tmp_path: Path):
    async def execute(root: Path, *, activate: bool = False):
        config = Config(input_dir=root / "input", output_epub=root / "book.epub",
                        workspace=root / "work", target_lang=None, timeout=1)
        config.input_dir.mkdir(parents=True); _image(config.input_dir / "page.png")

        def source() -> PageExtraction:
            return _source(config.input_dir / "page.png")

        with patch("btran.source_extractor.extract_page", AsyncMock(side_effect=lambda *_, **__: source())):
            initial = await run(config)
        if not activate:
            return config, initial
        RevisionStore(config.workspace).activate(initial.candidate_revision_id)
        with patch("btran.source_extractor.extract_page", AsyncMock(side_effect=lambda *_, **__: source())):
            return config, await run(config)

    clean_config, clean = await execute(tmp_path / "clean")
    incremental_config, incremental = await execute(tmp_path / "incremental", activate=True)
    assert clean.status == incremental.status == "completed"

    clean_store, incremental_store = ArtifactStore(clean_config.workspace), ArtifactStore(incremental_config.workspace)
    for kind in ("EffectiveTargetPage", "SealedRenderInput"):
        left = _selected_artifact(clean_store, clean.candidate_revision_id, kind)
        right = _selected_artifact(incremental_store, incremental.candidate_revision_id, kind)
        assert left.artifact_id == right.artifact_id
        assert hashlib.sha256(left.to_json().encode()).hexdigest() == hashlib.sha256(right.to_json().encode()).hexdigest()

    revisions = RevisionStore(clean_config.workspace)
    snapshot = revisions.verify_bundle(clean.candidate_revision_id)
    bundle = clean_config.workspace / "revisions" / clean.candidate_revision_id
    assert (bundle / "book.epub").read_bytes() == clean_config.output_epub.read_bytes()
    render = _selected_artifact(clean_store, clean.candidate_revision_id, "SealedRenderInput")
    assert (bundle / "artifacts" / f"{render.artifact_id}.json").read_bytes() == (
        clean_config.workspace / "artifacts" / f"{render.artifact_id}.json"
    ).read_bytes()
    rendered = _selected_artifact(clean_store, clean.candidate_revision_id, "RenderedEpub")
    # Bundle remains reproducible after mutable cache history disappears.
    (clean_config.workspace / "artifacts" / f"{render.artifact_id}.json").unlink()
    assert revisions.verify_bundle(snapshot.revision_id) == snapshot

    reverse = revisions.selected_graph(clean.candidate_revision_id).reverse(clean.candidate_revision_id, rendered.artifact_id)
    assert any(edge.parent_artifact_id == render.artifact_id and edge.child_artifact_id == rendered.artifact_id
               for edge in reverse)

    report = clean.report
    assert report is not None
    categories = (report.content_finding_ids, report.uncertainty_finding_ids,
                  report.review_finding_ids, report.recoverable_failure_finding_ids)
    flattened = [finding_id for category in categories for finding_id in category]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(snapshot.selected_finding_ids)


@pytest.mark.asyncio
async def test_refresh_records_unactivated_attempt_and_preserves_active_revision(tmp_path: Path):
    config = _config(tmp_path)
    config.input_dir.mkdir(); _image(config.input_dir / "page.png")
    source = AsyncMock(side_effect=[
        _source(config.input_dir / "page.png", "old source"),
        _source(config.input_dir / "page.png", "refreshed source"),
    ])
    with patch("btran.source_extractor.extract_page", source):
        initial = await run(config)
        RevisionStore(config.workspace).activate(initial.candidate_revision_id)
        config.refresh = True
        refreshed = await run(config)

    assert refreshed.status == "completed"
    assert refreshed.report is not None
    assert refreshed.refresh_attempt_ids
    assert refreshed.report.refresh_attempt_ids == refreshed.refresh_attempt_ids
    assert RevisionStore(config.workspace).active_snapshot().revision_id == initial.candidate_revision_id
    snapshot = RevisionStore(config.workspace).snapshot(refreshed.candidate_revision_id)
    artifacts = ArtifactStore(config.workspace)
    attempts = [artifacts.get(artifact_id) for artifact_id in snapshot.selected_artifact_ids
                if artifacts.get(artifact_id).kind == "RefreshAttempt"]
    assert len(attempts) == 1
    assert attempts[0].payload["refresh_attempt_id"] == refreshed.refresh_attempt_ids[0]
    reachable = set(attempts[0].payload["reachable_artifact_ids"])
    assert reachable
    refresh_candidate = next(artifacts.get(artifact_id) for artifact_id in attempts[0].dependency_ids
                             if artifacts.get(artifact_id).kind == "RefreshCandidate")
    assert reachable.issubset(refresh_candidate.dependency_ids)
    assert set(refresh_candidate.dependency_ids) - reachable  # refreshed returned leaf IDs retained too.
    assert any(record.stage == "refresh" and record.status == "completed"
               for record in refreshed.report.stage_records)


@pytest.mark.asyncio
async def test_input_failure_terminates_and_persists_machine_readable_report(tmp_path: Path):
    config = _config(tmp_path)
    result = await run(config)

    assert result.status == "invocation_failed"
    assert result.report is not None
    assert result.report.invocation_failures[0]["code"] == "input_access"
    assert Path(result.report_path).is_file()


@pytest.mark.asyncio
async def test_output_access_failure_terminates_with_report(tmp_path: Path):
    config = _config(tmp_path)
    config.input_dir.mkdir(); _image(config.input_dir / "page.png")
    with patch("btran.source_extractor.extract_page", AsyncMock(return_value=_source(config.input_dir / "page.png"))), \
         patch("btran.orchestrator.build_epub", side_effect=EpubInvocationError("destination unavailable")):
        result = await run(config)

    assert result.status == "invocation_failed"
    assert result.report is not None
    assert result.report.invocation_failures[0]["code"] == "output_access"
    assert Path(result.report_path).is_file()
