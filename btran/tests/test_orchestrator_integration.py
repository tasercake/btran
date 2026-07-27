"""Wave 2 orchestration acceptance tests; all model leaves are mocked."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

from btran.config import Config
from btran.orchestrator import _initial_review_items, run
from btran.schema import PageExtraction, SourceBlock, TermMention, TerminologyEntry, TerminologyMap, TranslatedBlock
from btran.terminology import _stable_concept_base, freeze_terminology


def image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (600, 600), color).save(path)


def config(tmp_path: Path, **changes: object) -> Config:
    values = dict(
        input_dir=tmp_path / "input", output_epub=tmp_path / "book.epub",
        intermediate_dir=tmp_path / "work",
        target_lang="en", max_retries=1, concurrency=2,
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
async def test_source_extraction_cache_reuses_valid_run_owned_artifact(tmp_path: Path):
    """A second run reuses only the source artifact bound to its page and image."""
    cfg = config(tmp_path)
    cfg.input_dir.mkdir()
    source_path = cfg.input_dir / "a.png"
    image(source_path, (1, 2, 3))

    async def extract(path, model, sha256, phash, page, *_):
        return PageExtraction(
            page, str(path), sha256, phash, "ja", model,
            blocks=[SourceBlock(f"page_{page}_block_0", "paragraph", "source", 0)],
        )

    async def translate(source, _glossary, **_):
        return translations(source)

    with patch("btran.orchestrator.extract_page", side_effect=extract) as leaf, \
         patch("btran.orchestrator.translate_blocks", side_effect=translate), \
         patch("btran.orchestrator.build_epub"):
        assert (await run(cfg)).errors == []
        assert (await run(cfg)).errors == []

    assert leaf.await_count == 1
    assert len(list((cfg.intermediate_dir / "source_cache").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_corrupt_source_cache_is_a_miss_and_is_replaced(tmp_path: Path):
    """Malformed cached extraction data never reaches downstream stages."""
    cfg = config(tmp_path)
    cfg.input_dir.mkdir()
    source_path = cfg.input_dir / "a.png"
    image(source_path, (1, 2, 3))
    from btran.source_extractor import extraction_cache_identity
    from btran.hasher import compute_sha256
    cache_path = cfg.intermediate_dir / "source_cache" / f"{extraction_cache_identity(compute_sha256(source_path), cfg.model)}.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not extraction JSON")

    async def extract(path, model, sha256, phash, page, *_):
        return PageExtraction(
            page, str(path), sha256, phash, "ja", model,
            blocks=[SourceBlock(f"page_{page}_block_0", "paragraph", "source", 0)],
        )

    with patch("btran.orchestrator.extract_page", side_effect=extract) as leaf, \
         patch("btran.orchestrator.translate_blocks", side_effect=lambda source, *_args, **_kwargs: translations(source)), \
         patch("btran.orchestrator.build_epub"):
        assert (await run(cfg)).errors == []

    assert leaf.await_count == 1
    assert json.loads(cache_path.read_text())["page_number"] == 1


@pytest.mark.asyncio
async def test_corrupt_translation_cache_is_a_miss_and_is_replaced(tmp_path: Path):
    """Malformed cached translations never block a recoverable resume or reach EPUB."""
    cfg = config(tmp_path)
    cfg.input_dir.mkdir()
    source_path = cfg.input_dir / "a.png"
    image(source_path, (1, 2, 3))

    async def extract(path, model, sha256, phash, page, *_):
        return PageExtraction(
            page, str(path), sha256, phash, "ja", model,
            blocks=[SourceBlock(f"page_{page}_block_0", "paragraph", "source", 0)],
        )

    async def translate(source, _glossary, **_):
        return translations(source)

    with patch("btran.orchestrator.extract_page", side_effect=extract), \
         patch("btran.orchestrator.translate_blocks", side_effect=translate) as leaf, \
         patch("btran.orchestrator.build_epub"):
        assert (await run(cfg)).errors == []
        cache_path = next((cfg.intermediate_dir / "translation_cache").glob("*.json"))
        cache_path.write_text("not translation JSON")
        assert (await run(cfg)).errors == []

    assert leaf.await_count == 2
    assert json.loads(cache_path.read_text())[0]["block_id"] == "page_1_block_0"


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
    stable_id = _stable_concept_base((("島",), (source.blocks[0].id,)))
    glossary = freeze_terminology(
        [TerminologyEntry(stable_id, ["島"], "island", [source.blocks[0].id], 1.0)],
        source_lang=source.source_lang, target_lang="en",
    )
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
    glossary = freeze_terminology(
        [TerminologyEntry("island", ["島"], "island", [sources[0].blocks[0].id], 1)],
        source_lang=sources[0].source_lang, target_lang="en",
    )
    revised = freeze_terminology(
        [TerminologyEntry("island", ["島"], "isle", [sources[0].blocks[0].id], 1)],
        source_lang=sources[0].source_lang, target_lang="en", version="2",
    )
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
async def test_source_failure_callback_arrives_before_another_page_finishes(tmp_path: Path):
    """A terminal page failure streams immediately; it does not wait for the extraction gate."""
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3)); image(cfg.input_dir / "b.png", (4, 5, 6))
    reported = asyncio.Event()
    failed_in_leaf = asyncio.Event()
    release_second_page = asyncio.Event()

    async def extract(_path, _model, _sha, _phash, page_number, _pi_bin, _timeout):
        if page_number == 1:
            failed_in_leaf.set()
            raise RuntimeError("bad OCR")
        await release_second_page.wait()
        return extraction(page_number, cfg.input_dir / "b.png")

    with patch("btran.orchestrator.extract_page", side_effect=extract):
        task = asyncio.create_task(run(cfg, on_page_error=lambda _page, _message: reported.set()))
        try:
            await asyncio.wait_for(failed_in_leaf.wait(), timeout=2)
            await asyncio.wait_for(reported.wait(), timeout=0.1)
        finally:
            release_second_page.set()
            result = await task

    assert result.errors and "page 1" in result.errors[0]


@pytest.mark.asyncio
async def test_reconciliation_exception_is_reported_without_replacing_existing_epub(tmp_path: Path):
    """A reconciliation leaf exception is a terminal gate failure, not an escaped task error."""
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3)); cfg.output_epub.write_bytes(b"old EPUB")
    source = extraction(1, cfg.input_dir / "a.png")
    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.translate_blocks", AsyncMock(return_value=translations(source))), \
         patch("btran.orchestrator.reconcile", side_effect=RuntimeError("reconcile crashed")), \
         patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg)

    assert result.errors == ["[btran] reconciliation failed: RuntimeError: reconcile crashed"]
    assert cfg.output_epub.read_bytes() == b"old EPUB"
    epub.assert_not_called()


@pytest.mark.asyncio
async def test_resolved_glossary_correction_freezes_v2_before_translation(tmp_path: Path):
    """A reviewed correction changes the frozen glossary and translation cache semantic input."""
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3))
    source = extraction(1, cfg.input_dir / "a.png", "島")
    from btran.orchestrator import _review_id
    from btran.review import ReviewItem, write_items
    stable_id = _stable_concept_base((("島",), (source.blocks[0].id,)))
    write_items(cfg.intermediate_dir / "needs_review", [ReviewItem(
        _review_id("low-confidence", stable_id, 1), "low_confidence", True,
        {"concept_id": stable_id}, status="resolved",
        resolution={"action": "correct", "correction": "isle"},
    )])

    async def translate(extracted, frozen, **_):
        assert frozen.version == "2"
        assert frozen.entries[0].target_term == "isle"
        return translations(extracted, "isle")

    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=lambda _: json.dumps({"entries": [{"concept_id": "island", "source_terms": ["島"], "target_term": "island", "provenance": [source.blocks[0].id], "confidence": .2}]})), \
         patch("btran.orchestrator.translate_blocks", side_effect=translate), \
         patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg)

    assert result.errors == []
    assert json.loads((cfg.intermediate_dir / "glossary.v2.json").read_text())["entries"][0]["target_term"] == "isle"
    epub.assert_called_once()


def test_initial_review_uses_exact_canonical_provenance_page():
    entry = TerminologyEntry(
        concept_id="stable", source_terms=["term"], target_term="term",
        provenance=["page_39_block_0"], confidence=.2,
    )
    glossary = TerminologyMap("1", "hash", "te", "en", [entry])
    paths = {3: Path("page_0003.jpg"), 39: Path("page_0039.jpg")}

    items = _initial_review_items(glossary, paths)

    assert len(items) == 1
    assert items[0].page_number == 39
    assert items[0].image_path == "page_0039.jpg"


@pytest.mark.asyncio
async def test_equivalent_regenerated_glossary_reuses_review_despite_model_id_change(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3))
    source = extraction(1, cfg.input_dir / "a.png", "島")
    responses = [
        lambda _: json.dumps({"entries": [{
            "concept_id": "island", "source_terms": ["島"],
            "target_term": "island", "provenance": [source.blocks[0].id],
            "confidence": .2,
        }]}),
        lambda _: json.dumps({"entries": [{
            "concept_id": "c086", "source_terms": ["島"],
            "target_term": "isle", "provenance": [source.blocks[0].id],
            "confidence": .3, "notes": "new wording",
        }]}),
    ]
    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.make_pi_consolidation_call", side_effect=responses), \
         patch("btran.orchestrator.translate_blocks", AsyncMock(return_value=translations(source, "isle"))), \
         patch("btran.orchestrator.build_epub") as epub:
        first = await run(cfg)
        assert first.errors == ["[btran] blocking glossary review items remain unresolved"]
        review_path = next((cfg.intermediate_dir / "needs_review").glob("*.json"))
        from btran.review import resolve_item
        resolved_id = resolve_item(review_path, "accept").item_id

        second = await run(cfg)

    assert second.errors == []
    assert (cfg.intermediate_dir / "needs_review" / "archive" / f"{resolved_id}.json").is_file()
    epub.assert_called_once()


@pytest.mark.asyncio
async def test_stale_resolved_correction_is_archived_before_current_glossary_review(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.input_dir.mkdir(); image(cfg.input_dir / "a.png", (1, 2, 3))
    source = extraction(1, cfg.input_dir / "a.png", "島")
    from btran.review import ReviewItem, write_items
    stale = ReviewItem(
        "stale-review", "low_confidence", True, {"concept_id": "obsolete"},
        status="resolved", resolution={"action": "correct", "correction": "old"},
    )
    write_items(cfg.intermediate_dir / "needs_review", [stale])

    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), \
         patch("btran.orchestrator.make_pi_consolidation_call", return_value=lambda _: json.dumps({"entries": [{
             "concept_id": "arbitrary", "source_terms": ["島"], "target_term": "island",
             "provenance": [source.blocks[0].id], "confidence": 1,
         }]})), \
         patch("btran.orchestrator.translate_blocks", AsyncMock(return_value=translations(source, "island"))), \
         patch("btran.orchestrator.build_epub"):
        result = await run(cfg)

    assert result.errors == []
    assert not (cfg.intermediate_dir / "needs_review" / "stale-review.json").exists()
    assert (cfg.intermediate_dir / "needs_review" / "archive" / "stale-review.json").is_file()


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
