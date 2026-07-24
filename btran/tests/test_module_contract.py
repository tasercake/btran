"""Gate 1: public-module fan-in and the frozen orchestrator boundary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran import (
    epub_builder,
    eval_harness,
    manifest,
    preflight,
    reconciliation,
    source_extractor,
    terminology,
    translator,
    validators,
)
from btran.config import Config
from btran.orchestrator import RunResult as OrchestratorRunResult
from btran.orchestrator import orchestrator_run
from btran.orchestrator_contract import OrchestratorCallable, RunResult
from btran.schema import ErrorResult, Manifest, PageResult


@pytest.mark.asyncio
async def test_orchestrator_contract_supports_a_fake_with_streaming_failures():
    """WP-8 can use this boundary before WP-7 supplies its implementation."""
    reported: list[tuple[int, str]] = []

    async def fake_run(
        config: Config, on_page_error=None
    ) -> RunResult:
        assert config.target_lang == "fr"
        assert on_page_error is not None
        on_page_error(1, "translation failed")
        return RunResult(errors=["[btran] page 1 failed: translation failed"])

    runner: OrchestratorCallable = fake_run
    result = await runner(
        Config(target_lang="fr"),
        on_page_error=lambda page_number, message: reported.append((page_number, message)),
    )

    assert result.errors == ["[btran] page 1 failed: translation failed"]
    assert reported == [(1, "translation failed")]
    assert OrchestratorRunResult is RunResult
    assert callable(orchestrator_run)


@pytest.mark.asyncio
async def test_wave_one_modules_exchange_representative_typed_artifacts(tmp_path: Path):
    """Mock Pi results flow through extraction, terminology, translation, validation, and EPUB."""
    assert all(
        callable(public_api)
        for public_api in (
            manifest.generate_manifest,
            preflight.preflight_manifest,
            source_extractor.extract_page,
            terminology.consolidate_terminology,
            translator.translate_blocks,
            reconciliation.reconcile,
            validators.validate_page,
            eval_harness.run_corpus,
            epub_builder.build_epub,
        )
    )

    typed_manifest = Manifest(
        input_dir=str(tmp_path),
        pages=[{"filename": "page-001.png", "page_number": 1, "status": "pending"}],
        total_pages=1,
    )
    assert typed_manifest.total_pages == 1
    assert ErrorResult(page_number=1, error="failed").error == "failed"

    extraction_response = {
        "blocks": [
            {"id": "heading", "type": "heading", "text": "Chapter One", "reading_order": 0},
            {"id": "body", "type": "paragraph", "text": "An island appears.", "reading_order": 1},
        ],
        "term_mentions": [{"term": "island", "block_id": "body"}],
        "illustrations": [],
    }
    extraction_process = AsyncMock()
    extraction_process.communicate = AsyncMock(
        return_value=(json.dumps(extraction_response).encode(), b"")
    )
    extraction_process.returncode = 0
    with patch(
        "btran.source_extractor.asyncio.create_subprocess_exec",
        AsyncMock(return_value=extraction_process),
    ):
        extraction = await source_extractor.extract_page(
            image_path=tmp_path / "page-001.png",
            source_lang="en",
            model="vision-model",
            sha256="a" * 64,
            phash="b" * 16,
            page_number=1,
        )

    def consolidate_pi(_: str) -> str:
        return json.dumps(
            {
                "entries": [
                    {
                        "concept_id": "island",
                        "source_terms": ["island"],
                        "target_term": "island",
                        "provenance": ["page_1_block_1"],
                        "confidence": 1.0,
                    }
                ]
            }
        )

    glossary = terminology.consolidate_terminology(
        extraction.term_mentions,
        source_lang="en",
        target_lang="en",
        pi_call=consolidate_pi,
    )

    translation_response = {
        "blocks": [
            {"block_id": "page_1_block_0", "translated_text": "Chapter One"},
            {"block_id": "page_1_block_1", "translated_text": "The island appears."},
        ]
    }
    translation_process = AsyncMock()
    translation_process.communicate = AsyncMock(
        return_value=(json.dumps(translation_response).encode(), b"")
    )
    translation_process.returncode = 0
    with patch(
        "btran.translator.asyncio.create_subprocess_exec",
        AsyncMock(return_value=translation_process),
    ):
        translated_blocks = await translator.translate_blocks(
            extraction, glossary, model="text-model"
        )

    page_result = PageResult(
        page_number=extraction.page_number,
        sha256=extraction.sha256,
        phash=extraction.phash,
        image_path=extraction.image_path,
        source_lang=extraction.source_lang,
        target_lang=glossary.target_lang,
        page_text=source_extractor.legacy_page_text(extraction),
        translated_text="\n".join(block.translated_text for block in translated_blocks),
        blocks=extraction.blocks,
        translated_blocks=translated_blocks,
        term_mentions=extraction.term_mentions,
        illustrations=extraction.illustrations,
    )

    assert all(not errors for errors in validators.validate_page(extraction, page_result, glossary).values())
    assert reconciliation.reconcile(
        glossary=glossary,
        extractions=[extraction],
        translations={1: translated_blocks},
    ).affected_pages == []

    output_path = tmp_path / "book.epub"
    epub_builder.build_epub([page_result], output_path, target_lang="en")
    assert output_path.is_file()
