"""Wave 2 orchestration acceptance tests; all model leaves are mocked."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

from btran.config import Config
from btran.orchestrator import run
from btran.schema import PageExtraction, SourceBlock, TermMention, TerminologyEntry, TranslatedBlock
from btran.terminology import freeze_terminology


def image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (600, 600), color).save(path)


def config(tmp_path: Path, **changes: object) -> Config:
    values = dict(
        input_dir=tmp_path / "input", output_epub=tmp_path / "book.epub",
        intermediate_dir=tmp_path / "work", cache_db=tmp_path / "cache.sqlite",
        source_lang="ja", target_lang="en", max_retries=1, concurrency=2,
    )
    values.update(changes)
    return Config(**values)


def extraction(page: int, path: Path, term: str = "") -> PageExtraction:
    blocks = [SourceBlock(f"page_{page}_block_0", "paragraph", f"source {term}".strip(), 0)]
    mentions = [TermMention(term, blocks[0].id)] if term else []
    return PageExtraction(page, str(path), "a" * 64, f"{page:016x}", "ja", "vision", blocks=blocks, term_mentions=mentions)


def translations(source: PageExtraction, term: str = "") -> list[TranslatedBlock]:
    return [TranslatedBlock(block.id, f"translation {term}".strip()) for block in source.blocks]


@pytest.mark.asyncio
async def test_full_happy_path_writes_checkpointed_artifacts_and_epub(tmp_path: Path):
    cfg = config(tmp_path, epub_check=True, epub_check_path="test-epubcheck")
    cfg.input_dir.mkdir()
    image(cfg.input_dir / "a.png", (255, 0, 0))
    image(cfg.input_dir / "b.png", (0, 255, 0))
    sources = [extraction(1, cfg.input_dir / "a.png"), extraction(2, cfg.input_dir / "b.png")]

    with patch("btran.orchestrator.extract_page", AsyncMock(side_effect=sources)) as extract, \
         patch("btran.orchestrator.translate_blocks", AsyncMock(side_effect=[translations(s) for s in sources])) as translate, \
         patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg)

    assert result.errors == []
    assert extract.await_count == translate.await_count == 2
    assert epub.call_args.kwargs["epub_check"] is True
    assert epub.call_args.kwargs["epub_check_path"] == "test-epubcheck"
    manifest = json.loads((cfg.intermediate_dir / ".run_manifest.json").read_text())
    assert manifest["stages"]["epub"]["status"] == "succeeded"
    assert (cfg.intermediate_dir / "source" / "page_0001.json").is_file()
    assert (cfg.intermediate_dir / "page_0001.json").is_file()


@pytest.mark.asyncio
async def test_invalid_manifest_or_preflight_calls_no_pi_leaves(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.input_dir.mkdir()
    image(cfg.input_dir / "tiny.png", (1, 2, 3))
    # A syntactically valid manifest still must preflight the referenced page.
    (cfg.input_dir / "manifest.json").write_text(json.dumps({"input_dir": str(cfg.input_dir), "pages": [{"filename": "tiny.png", "page_number": 1, "status": "pending"}], "total_pages": 1}))
    Image.new("RGB", (10, 10)).save(cfg.input_dir / "tiny.png")
    with patch("btran.orchestrator.extract_page", AsyncMock()) as extract, \
         patch("btran.orchestrator.make_pi_consolidation_call") as consolidate, \
         patch("btran.orchestrator.translate_blocks", AsyncMock()) as translate:
        result = await run(cfg)
    assert result.errors
    extract.assert_not_awaited(); translate.assert_not_awaited(); consolidate.assert_not_called()


@pytest.mark.asyncio
async def test_source_failure_blocks_glossary_translation_and_epub_and_streams_immediately(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3))
    reported: list[tuple[int, str]] = []
    with patch("btran.orchestrator.extract_page", AsyncMock(side_effect=RuntimeError("bad OCR"))), \
         patch("btran.orchestrator.make_pi_consolidation_call") as consolidate, \
         patch("btran.orchestrator.translate_blocks", AsyncMock()) as translate, \
         patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg, on_page_error=lambda page, message: reported.append((page, message)))
    assert "page 1" in result.errors[0]
    assert reported and reported[0][0] == 1
    consolidate.assert_not_called(); translate.assert_not_awaited(); epub.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_source_checkpoint_blocks_glossary_and_translation(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3))
    source = extraction(1, cfg.input_dir / "a.png")
    def malformed_checkpoint(_source, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not source JSON")
    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.to_file", side_effect=malformed_checkpoint), \
         patch("btran.orchestrator.make_pi_consolidation_call") as consolidate, \
         patch("btran.orchestrator.translate_blocks", AsyncMock()) as translate:
        result = await run(cfg)
    assert result.errors and "source checkpoint" in result.errors[0]
    consolidate.assert_not_called(); translate.assert_not_awaited()


@pytest.mark.asyncio
async def test_glossary_is_frozen_before_translation_and_cache_key_changes_with_hash(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3))
    source = extraction(1, cfg.input_dir / "a.png", "島")
    glossary = freeze_terminology([TerminologyEntry("island", ["島"], "island", [source.blocks[0].id], 1.0)], source_lang="ja", target_lang="en")
    order: list[str] = []
    def pi_factory(**_: object):
        order.append("freeze")
        return lambda _: json.dumps({"entries": [{"concept_id": "island", "source_terms": ["島"], "target_term": "island", "provenance": [source.blocks[0].id], "confidence": 1.0}]})
    async def translate(extracted, frozen, **_: object):
        assert frozen.hash == glossary.hash
        assert order == ["freeze"]
        order.append("translate")
        return translations(extracted, "island")
    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.make_pi_consolidation_call", side_effect=pi_factory), \
         patch("btran.orchestrator.translate_blocks", side_effect=translate), \
         patch("btran.orchestrator.build_epub"):
        assert (await run(cfg)).errors == []
    assert order == ["freeze", "translate"]
    run_manifest = json.loads((cfg.intermediate_dir / ".run_manifest.json").read_text())
    assert run_manifest["glossary"]["hash"] == glossary.hash
    assert len(list((cfg.intermediate_dir / "translation_cache").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_reconciliation_retranslates_only_affected_pages_once(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3)); image(cfg.input_dir / "b.png", (4, 5, 6))
    sources = [extraction(1, cfg.input_dir / "a.png", "島"), extraction(2, cfg.input_dir / "b.png")]
    glossary = freeze_terminology([TerminologyEntry("island", ["島"], "island", [sources[0].blocks[0].id], 1)], source_lang="ja", target_lang="en")
    revised = freeze_terminology([TerminologyEntry("island", ["島"], "isle", [sources[0].blocks[0].id], 1)], source_lang="ja", target_lang="en", version="2")
    from btran.reconciliation import GlossaryChange, ReconciliationResult
    reconciliation = ReconciliationResult(revised, [GlossaryChange("island", "island", "isle")], [], [1])
    translated_pages: list[int] = []
    async def translate(source, glossary, **kwargs):
        translated_pages.append(source.page_number)
        return translations(source, glossary.entries[0].target_term if glossary.entries else "")
    with patch("btran.orchestrator.extract_page", AsyncMock(side_effect=sources)), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=lambda _: json.dumps({"entries": [{"concept_id": "island", "source_terms": ["島"], "target_term": "island", "provenance": [sources[0].blocks[0].id], "confidence": 1}]})), \
         patch("btran.orchestrator.translate_blocks", side_effect=translate), \
         patch("btran.orchestrator.reconcile", return_value=reconciliation), \
         patch("btran.orchestrator.build_epub"):
        assert (await run(cfg)).errors == []
    assert translated_pages == [1, 2, 1]


@pytest.mark.asyncio
async def test_unresolved_review_item_blocks_epub_and_epubcheck_is_wired(tmp_path: Path):
    cfg = config(tmp_path, epub_check=True)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3))
    source = extraction(1, cfg.input_dir / "a.png", "島")
    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=lambda _: json.dumps({"entries": [{"concept_id": "island", "source_terms": ["島"], "target_term": "island", "provenance": [source.blocks[0].id], "confidence": .2}]})), \
         patch("btran.orchestrator.translate_blocks", AsyncMock(return_value=translations(source, "island"))) as translate, \
         patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg)
    assert result.errors and "review" in result.errors[0]
    translate.assert_not_awaited(); epub.assert_not_called()
    items = list((cfg.intermediate_dir / "needs_review").glob("*.json"))
    assert items and json.loads(items[0].read_text())["image_path"].endswith("a.png")
