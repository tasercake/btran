"""Legacy bridge does not let review/reconciliation/validation findings gate EPUB."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.artifacts import ArtifactStore, RevisionStore
from btran.config import Config
from btran.epub_builder import build_epub
from btran.orchestrator import run
from btran.reconciliation import reconcile_effective
from btran.schema import PageExtraction, SourceBlock
from btran.validators import validate_effective


def _config(tmp_path: Path) -> Config:
    return Config(input_dir=tmp_path / "in", output_epub=tmp_path / "out.epub", workspace=tmp_path / "work", target_lang=None, max_retries=1)


def _image(path: Path) -> None:
    Image.new("RGB", (600, 600), (1, 2, 3)).save(path)


def _source(path: Path) -> PageExtraction:
    return PageExtraction(1, str(path), "a" * 64, "1" * 16, "ja", "vision", blocks=[SourceBlock("page_1_block_0", "paragraph", "猫", 0)])


@pytest.mark.asyncio
async def test_validation_finding_does_not_block_epub(tmp_path: Path):
    cfg = _config(tmp_path); cfg.input_dir.mkdir(); _image(cfg.input_dir / "a.png")
    source = _source(cfg.input_dir / "a.png")

    def validate_with_informational_error(**kwargs):
        return validate_effective(**kwargs, rules={
            "informational": lambda _pages, _reconciliation, _mode: ("informational error",),
        })

    with patch("btran.source_extractor.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.reconcile_effective", wraps=reconcile_effective) as reconciliation, \
         patch("btran.orchestrator.validate_effective", side_effect=validate_with_informational_error) as validation, \
         patch("btran.orchestrator.build_epub", wraps=build_epub) as epub:
        result = await run(cfg)

    assert result.errors == []
    assert result.status == "completed"
    assert cfg.output_epub.is_file()
    assert result.report is not None and Path(result.report_path).is_file()
    reconciliation.assert_called_once()
    validation.assert_called_once()
    epub.assert_called_once()

    store = ArtifactStore(cfg.workspace)
    snapshot = RevisionStore(cfg.workspace).snapshot(result.candidate_revision_id)
    validation_artifact = next(
        store.get(artifact_id) for artifact_id in snapshot.selected_artifact_ids
        if store.get(artifact_id).kind == "ValidationArtifact"
    )
    assert validation_artifact.payload["rule_results"] == [{
        "rule": "informational", "errors": ["informational error"], "exception": None,
    }]
    finding = next(
        store.get_finding(finding_id) for finding_id in validation_artifact.finding_ids
        if store.get_finding(finding_id).kind == "validation_error"
    )
    assert finding.finding_id in result.report.content_finding_ids


def test_orchestrator_no_longer_imports_or_creates_review_workflow():
    import btran.orchestrator as orchestrator
    source = Path(orchestrator.__file__).read_text()
    assert "btran.review" not in source
    assert "needs_review" not in source
    assert "unresolved_items" not in source
