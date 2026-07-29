"""Tests for glossary-aware, text-only block translation."""

import asyncio
import json
import os
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from btran.schema import (
    ConceptProjection,
    EffectivePage,
    EffectiveSegment,
    PageExtraction,
    SourceBlock,
    TerminologyEntry,
    TerminologyMap,
    tagged_sha256,
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
        source_artifact_hash="source-a", glossary_hash="glossary-a",
        target_lang="en", model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-b", glossary_hash="glossary-a",
        target_lang="en", model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-a", glossary_hash="glossary-b",
        target_lang="en", model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-a", glossary_hash="glossary-a",
        target_lang="fr", model="text-model",
    )
    assert baseline != translation_cache_identity(
        source_artifact_hash="source-a", glossary_hash="glossary-a",
        target_lang="en", model="other-text-model",
    )


def test_translation_context_uses_page_neighbors_and_slices_glossary_for_them():
    """Boundary context is the adjacent pages, and its terminology is included."""
    from btran.translator import _translation_context

    previous = PageExtraction(
        6, "page-006.jpg", "previous-sha", "unused", "ja", "extractor",
        blocks=[SourceBlock("p6_b1", "paragraph", "前の犬", 0)],
    )
    following = PageExtraction(
        8, "page-008.jpg", "next-sha", "unused", "ja", "extractor",
        blocks=[SourceBlock("p8_b1", "paragraph", "次の犬", 0)],
    )
    context = _translation_context(_page(), _glossary(), previous, following)

    assert context["adjacent_source_boundaries"] == {
        "previous_page_tail": {"page_number": 6, "block_id": "p6_b1", "text": "前の犬"},
        "next_page_head": {"page_number": 8, "block_id": "p8_b1", "text": "次の犬"},
    }
    assert {entry["concept_id"] for entry in context["glossary"]} == {"cat", "dog"}


def test_translation_cache_identity_binds_prompt_and_output_schema():
    """Prompt or response-schema changes invalidate text translation cache entries."""
    import btran.translator as translator

    baseline = translator.translation_cache_identity(
        source_artifact_hash="source-a", glossary_hash="glossary-a",
        target_lang="en", model="text-model",
    )
    with patch.object(translator, "TRANSLATION_PROMPT", "different prompt"):
        assert baseline != translator.translation_cache_identity(
            source_artifact_hash="source-a", glossary_hash="glossary-a",
            target_lang="en", model="text-model",
        )
    with patch.object(translator, "TRANSLATION_OUTPUT_SCHEMA", {"version": "different"}):
        assert baseline != translator.translation_cache_identity(
            source_artifact_hash="source-a", glossary_hash="glossary-a",
            target_lang="en", model="text-model",
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
    assert exec_mock.call_args.kwargs["stdin"] is asyncio.subprocess.DEVNULL
    assert exec_mock.call_args.kwargs["start_new_session"] is (os.name == "posix")


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
    proc.pid = 123
    proc.poll = None  # asyncio.subprocess.Process has no synchronous poll().
    proc.returncode = None
    proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    proc.wait = AsyncMock(return_value=None)
    from btran.process_cleanup import _ProcessRef
    with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), \
         patch("btran.process_cleanup._proc_ref", return_value=_ProcessRef(123, 1)), \
         patch("btran.process_cleanup._signal_group", return_value=False), \
         patch("btran.process_cleanup.os.kill"):
        with pytest.raises(TranslationError, match="timed out"):
            await translate_blocks(_page(), _glossary(), model="text-model", timeout=1)

    proc.terminate.assert_called_once_with()


@pytest.mark.asyncio
async def test_translate_blocks_terminates_child_on_cancellation():
    """Cancellation reaps the child and propagates cancellation to the caller."""
    from btran.translator import translate_blocks

    started = asyncio.Event()

    async def never_finishes():
        started.set()
        await asyncio.Event().wait()

    proc = Mock(returncode=None)
    proc.pid = 123
    proc.poll = None  # asyncio.subprocess.Process has no synchronous poll().
    proc.communicate = never_finishes
    proc.wait = AsyncMock(return_value=None)
    from btran.process_cleanup import _ProcessRef
    with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), \
         patch("btran.process_cleanup._proc_ref", return_value=_ProcessRef(123, 1)), \
         patch("btran.process_cleanup._signal_group", return_value=False), \
         patch("btran.process_cleanup.os.kill"):
        task = asyncio.create_task(translate_blocks(_page(), _glossary(), model="text-model"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    proc.terminate.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="requires /proc POSIX process cleanup")
async def test_translation_timeout_kills_detached_setsid_pipe_holder(tmp_path):
    """Translation cleanup tracks Pi child that escaped original process group."""
    from btran.translator import TranslationError, translate_blocks

    pid_path = tmp_path / "escaped.pid"
    child_source = (
        "import os, signal, time\n"
        "os.setsid()\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True: time.sleep(.1)\n"
    )
    worker = tmp_path / "fake-pi"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import signal, subprocess, sys, time\n"
        f"child=subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        f"open({str(pid_path)!r}, 'w').write(str(child.pid))\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True: time.sleep(.1)\n",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    child_pid: int | None = None
    try:
        with pytest.raises(TranslationError, match="timed out"):
            await translate_blocks(_page(), _glossary(), model="model", pi_bin=str(worker), timeout=1)
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(.01)
        else:
            pytest.fail("detached translation pipe holder survived cleanup")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1, 0, 3601, 1.5, True])
async def test_translate_blocks_rejects_invalid_timeout_before_spawning_pi(timeout):
    from btran.translator import TranslationError, translate_blocks

    exec_mock = AsyncMock()
    with patch("btran.translator.asyncio.create_subprocess_exec", exec_mock):
        with pytest.raises(TranslationError, match="between 1 and 3600"):
            await translate_blocks(_page(), _glossary(), model="text-model", timeout=timeout)

    exec_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_target_materialization_rejects_zero_timeout_before_fallback_page_work(tmp_path):
    """Outer Task-10 fallback path cannot turn zero into an unbounded leaf."""
    from btran.translator import TranslationError, materialize_effective_target

    store, graph, source, _ = _task10_source(tmp_path, ["source"])
    model_called = False

    async def model(_):
        nonlocal model_called
        model_called = True
        return {"translated_text": "unexpected"}

    with pytest.raises(TranslationError, match="between 1 and 3600"):
        await materialize_effective_target(
            source, store=store, graph=graph, mode="translated", target_lang="en",
            timeout=0, max_retries=0, translation_call=model,
        )

    assert not model_called


def _task10_source(tmp_path, texts, *, diagnostic_indexes=()):
    """Minimal closed Task-7 source closure for target-materializer tests."""
    from btran.artifacts import ArtifactStore, DependencyGraph
    from btran.source_extractor import EffectiveSourceLeaf, EffectiveSourceRun

    store = ArtifactStore(tmp_path / "state")
    graph = DependencyGraph(tmp_path / "state")
    segments = []
    for index, text in enumerate(texts):
        diagnostic = index in diagnostic_indexes
        segment_id = f"segment-{index}"
        effective = EffectiveSegment(
            effective_segment_id=tagged_sha256("test-source", str(index).encode()),
            segment_id=segment_id, source_lang=None if diagnostic else "ja", source_text=text,
            effective_text=text, render_lang="und" if diagnostic else "ja", mode="native",
            finding_ids=(tagged_sha256("test-finding", str(index).encode()),) if diagnostic else (),
        )
        # Diagnostic test artifacts retain a real finding because store closure
        # validates all named Finding IDs.
        findings = ()
        if diagnostic:
            from btran.schema import Finding
            finding = Finding(kind="diagnostic", severity="warning", stage="test", subject_refs=(segment_id,), evidence={}, message="diagnostic")
            store.put_finding(finding)
            effective.finding_ids = (finding.finding_id,)
            findings = effective.finding_ids
        artifact = store.put("DiagnosticEffectiveSourceSegment" if diagnostic else "EffectiveSourceSegment", effective.to_dict(),
                             finding_ids=findings, semantic_key=f"source-{index}")
        segments.append((artifact, effective))
    page = EffectivePage(
        effective_page_id=tagged_sha256("test-page", b"one"), page_id="page-1",
        effective_segment_ids=tuple(item.effective_segment_id for _, item in segments),
        source_langs=("ja",), display_metadata={},
    )
    page_artifact = store.put("EffectiveSourcePage", page.to_dict(),
                              dependency_ids=tuple(item.artifact_id for item, _ in segments), semantic_key="source-page")
    leaf = EffectiveSourceLeaf("page-1", page_artifact.artifact_id, tuple(item.artifact_id for item, _ in segments), (), (), (), bool(diagnostic_indexes))
    return store, graph, EffectiveSourceRun((leaf,), "summary", "degraded" if diagnostic_indexes else "completed", ()), segments


@pytest.mark.asyncio
async def test_task10_native_materializes_unchanged_source_without_any_model_call(tmp_path):
    from btran.translator import materialize_effective_target

    store, graph, source, segments = _task10_source(tmp_path, ["一", "二"])
    called = False

    async def model(_):
        nonlocal called
        called = True
        raise AssertionError("native mode must not translate")

    result = await materialize_effective_target(source, store=store, graph=graph, mode="native", translation_call=model)

    assert result.status == "completed"
    output = [store.get(item).payload for item in result.leaves[0].segment_artifact_ids]
    assert [item["effective_text"] for item in output] == ["一", "二"]
    assert [item["render_lang"] for item in output] == ["ja", "ja"]
    assert not called


@pytest.mark.asyncio
async def test_task10_translated_uses_only_immediate_non_diagnostic_context_and_records_finding(tmp_path):
    from btran.translator import materialize_effective_target

    store, graph, source, _ = _task10_source(tmp_path, ["first", "bad", "third"], diagnostic_indexes=(1,))
    contexts = []

    async def model(context):
        contexts.append(context)
        return {"translated_text": "T:" + context["focal_source"]["text"], "confidence": .5}

    result = await materialize_effective_target(source, store=store, graph=graph, mode="translated", target_lang="en", translation_call=model)

    assert result.status == "degraded"  # diagnostic source is retained, not skipped.
    assert [item["focal_source"]["segment_id"] for item in contexts] == ["segment-0", "segment-2"]
    assert all(item["immediate_source_neighbors"]["preceding"] is None or item["immediate_source_neighbors"]["following"] is None for item in contexts)
    findings = [store.get_finding(item) for item in result.leaves[0].finding_ids]
    assert any(item.kind == "context_neighbor_diagnostic" for item in findings)
    assert any(item.kind == "review_request" and item.requires_action is False for item in findings)
    outputs = [store.get(item) for item in result.leaves[0].segment_artifact_ids]
    diagnostic = next(item for item in outputs if item.kind == "DiagnosticEffectiveTargetSegment")
    assert diagnostic.payload["mode"] == "translated"
    assert diagnostic.payload["source_lang"] is None
    assert diagnostic.payload["source_text"] == diagnostic.payload["effective_text"] == "bad"
    assert diagnostic.payload["render_lang"] == "und"
    assert diagnostic.payload["translation_artifact_id"] is None
    assert diagnostic.payload["finding_ids"]
    page = store.get(result.leaves[0].page_artifact_id)
    assert page.payload["display_metadata"] == {"target_lang": "en"}


@pytest.mark.asyncio
async def test_task10_local_target_segment_overlay_bypasses_model_and_reaches_target(tmp_path):
    from btran.corrections import OverlayInput
    from btran.translator import materialize_effective_target

    store, graph, source, segments = _task10_source(tmp_path, ["source"])
    base = store.put("TranslationArtifact", {"segment_id": "segment-0", "translated_text": "old", "mappings": []}, semantic_key="old-translation")
    overlay = OverlayInput("correction-1", "target_segment", "segment-0", "local replacement", (base.artifact_id,),
                           {"segment_id": "segment-0", "expected_target_text": "old"})

    async def model(_):
        raise AssertionError("local target overlay must bypass translation model")

    result = await materialize_effective_target(source, store=store, graph=graph, mode="translated", target_lang="en",
                                                target_overlays=(overlay,), translation_call=model)

    segment = store.get(result.leaves[0].segment_artifact_ids[0]).payload
    assert segment["effective_text"] == "local replacement"
    assert segment["target_overlay_artifact_id"] is not None
    edges = [graph.get(item) for item in result.graph_edge_ids]
    assert any(item.edge_kind == "target_overlay_to_effective_target" for item in edges)


def test_task10_refresh_preserves_old_and_new_leaves_in_unactivated_candidate(tmp_path):
    from btran.artifacts import ArtifactStore
    from btran.translator import refresh_reachable_model_leaves

    store = ArtifactStore(tmp_path / "state")
    old = store.put("ModelLeaf", {"version": "old"}, semantic_key="old")
    fresh = store.put("ModelLeaf", {"version": "fresh"}, semantic_key="fresh")

    refreshed = refresh_reachable_model_leaves(store=store, base_revision_id="a" * 64,
                                                reachable_artifact_ids=(old.artifact_id,),
                                                refreshed_artifact_ids=(fresh.artifact_id,))

    candidate = store.get(refreshed.candidate_artifact_id)
    assert set(candidate.dependency_ids) == {old.artifact_id, fresh.artifact_id}
    assert refreshed.attempt.base_revision_id == "a" * 64
    assert not (tmp_path / "state" / "active-revision.json").exists()
