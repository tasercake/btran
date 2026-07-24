"""Tests for glossary-aware, text-only block translation."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from btran.schema import (
    PageExtraction,
    SourceBlock,
    TerminologyEntry,
    TerminologyMap,
)


def _page() -> PageExtraction:
    return PageExtraction(
        page_number=7,
        image_path="page-007.jpg",
        sha256="source-artifact-sha",
        phash="unused",
        source_lang="ja",
        model="extractor",
        timestamp="2026-01-01T00:00:00Z",
        blocks=[
            SourceBlock(id="p7_b1", type="paragraph", text="猫が眠る。", reading_order=1),
            SourceBlock(id="p7_b2", type="paragraph", text="猫は起きた。", reading_order=2),
        ],
    )


def _glossary() -> TerminologyMap:
    return TerminologyMap(
        version="1",
        hash="glossary-v1-sha",
        source_lang="ja",
        target_lang="en",
        entries=[
            TerminologyEntry(
                concept_id="cat",
                source_terms=["猫"],
                target_term="cat",
                provenance=["p7_b1"],
                confidence=1.0,
            ),
            TerminologyEntry(
                concept_id="dog",
                source_terms=["犬"],
                target_term="dog",
                provenance=[],
                confidence=1.0,
            ),
        ],
        created_at="2026-01-01T00:00:00Z",
    )


def _proc(payload: dict):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(json.dumps(payload).encode(), b""))
    proc.returncode = 0
    return proc


@pytest.mark.asyncio
async def test_translate_blocks_preserves_ids_and_sends_text_only_glossary_context():
    """Every returned block corresponds exactly and prompt contains only relevant terms."""
    from btran.translator import translate_blocks

    proc = _proc(
        {
            "blocks": [
                {"block_id": "p7_b1", "translated_text": "The cat sleeps."},
                {"block_id": "p7_b2", "translated_text": "The cat woke up."},
            ]
        }
    )
    exec_mock = AsyncMock(return_value=proc)

    with patch("btran.translator.asyncio.create_subprocess_exec", exec_mock):
        result = await translate_blocks(_page(), _glossary(), model="text-model")

    assert [block.block_id for block in result] == ["p7_b1", "p7_b2"]
    assert [block.translated_text for block in result] == ["The cat sleeps.", "The cat woke up."]
    prompt = exec_mock.call_args.args[-1]
    assert "猫" in prompt
    assert '"target_term": "cat"' in prompt
    assert "犬" not in prompt
    assert "p7_b1" in prompt and "p7_b2" in prompt
    assert "@page-007.jpg" not in prompt
    assert "--model" in exec_mock.call_args.args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_ids", "error"),
    [(["p7_b1"], "missing"), (["p7_b1", "p7_b2", "p7_b3"], "extra")],
)
async def test_translate_blocks_rejects_missing_or_extra_block_ids(response_ids, error):
    """A model response cannot omit or invent source block identifiers."""
    from btran.translator import TranslationError, translate_blocks

    proc = _proc(
        {"blocks": [{"block_id": block_id, "translated_text": "text"} for block_id in response_ids]}
    )
    with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(TranslationError, match=error):
            await translate_blocks(_page(), _glossary(), model="text-model")


def test_translation_cache_identity_changes_for_source_artifact_or_glossary():
    """Text translations are never shared across source artifacts or glossary versions."""
    from btran.translator import translation_cache_identity

    baseline = translation_cache_identity(
        source_artifact_hash="source-a",
        glossary_hash="glossary-a",
        source_lang="ja",
        target_lang="en",
        model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-b",
        glossary_hash="glossary-a",
        source_lang="ja",
        target_lang="en",
        model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-a",
        glossary_hash="glossary-b",
        source_lang="ja",
        target_lang="en",
        model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-a",
        glossary_hash="glossary-a",
        source_lang="ko",
        target_lang="en",
        model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-a",
        glossary_hash="glossary-a",
        source_lang="ja",
        target_lang="fr",
        model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-a",
        glossary_hash="glossary-a",
        source_lang="ja",
        target_lang="en",
        model="other-text-model",
    )


def test_translation_cache_identity_binds_prompt_and_output_schema():
    """Prompt or response-schema changes invalidate text translation cache entries."""
    import btran.translator as translator

    baseline = translator.translation_cache_identity(
        source_artifact_hash="source-a",
        glossary_hash="glossary-a",
        source_lang="ja",
        target_lang="en",
        model="text-model",
    )
    with patch.object(translator, "TRANSLATION_PROMPT", "different prompt"):
        assert baseline != translator.translation_cache_identity(
            source_artifact_hash="source-a",
            glossary_hash="glossary-a",
            source_lang="ja",
            target_lang="en",
            model="text-model",
        )
    with patch.object(translator, "TRANSLATION_OUTPUT_SCHEMA", {"version": "different"}):
        assert baseline != translator.translation_cache_identity(
            source_artifact_hash="source-a",
            glossary_hash="glossary-a",
            source_lang="ja",
            target_lang="en",
            model="text-model",
        )


@pytest.mark.asyncio
async def test_translate_blocks_isolated_from_tools_and_project_configuration():
    """Untrusted source text reaches an ephemeral Pi subprocess with no capabilities."""
    from btran.translator import translate_blocks

    proc = _proc(
        {"blocks": [
            {"block_id": "p7_b1", "translated_text": "The cat sleeps."},
            {"block_id": "p7_b2", "translated_text": "The cat woke up."},
        ]}
    )
    exec_mock = AsyncMock(return_value=proc)
    with patch("btran.translator.asyncio.create_subprocess_exec", exec_mock):
        await translate_blocks(_page(), _glossary(), model="text-model")

    args = exec_mock.call_args.args
    for option in ("--no-session", "--no-tools", "--no-extensions", "--no-skills",
                   "--no-prompt-templates", "--no-context-files", "--no-approve"):
        assert option in args


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"blocks": [{"block_id": "p7_b1", "translated_text": "one"},
                {"block_id": "p7_b2", "translated_text": "two"}], "note": "ignore me"},
    {"blocks": [{"block_id": "p7_b1", "translated_text": "one", "extra": "ignore me"},
                {"block_id": "p7_b2", "translated_text": "two"}]},
    {"blocks": [{"block_id": "p7_b1", "translated_text": ["not text"]},
                {"block_id": "p7_b2", "translated_text": "two"}]},
])
async def test_translate_blocks_rejects_untrusted_non_schema_json(payload):
    """Only the exact response schema is accepted from Pi output."""
    from btran.translator import TranslationError, translate_blocks

    with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=_proc(payload))):
        with pytest.raises(TranslationError, match="schema"):
            await translate_blocks(_page(), _glossary(), model="text-model")


@pytest.mark.asyncio
async def test_translate_blocks_terminates_child_on_timeout():
    """A timed out request reaps its Pi child before reporting TranslationError."""
    from btran.translator import TranslationError, translate_blocks

    proc = Mock()
    proc.returncode = None
    proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    proc.wait = AsyncMock(return_value=None)
    with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(TranslationError, match="timed out"):
            await translate_blocks(_page(), _glossary(), model="text-model", timeout=1)

    proc.terminate.assert_called_once_with()
    proc.wait.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_translate_blocks_terminates_child_on_cancellation():
    """Cancellation reaps the child and propagates cancellation to the caller."""
    from btran.translator import translate_blocks

    started = asyncio.Event()

    async def never_finishes():
        started.set()
        await asyncio.Event().wait()

    proc = Mock(returncode=None)
    proc.communicate = never_finishes
    proc.wait = AsyncMock(return_value=None)
    with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        task = asyncio.create_task(translate_blocks(_page(), _glossary(), model="text-model"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    proc.terminate.assert_called_once_with()
    proc.wait.assert_awaited_once_with()
