"""Tests for typed source extraction from page images."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from btran.schema import PageExtraction, SourceBlock, TermMention


def _make_mock_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.returncode = returncode
    return proc


_VALID_OUTPUT = json.dumps(
    {
        "source_lang": "en",
        "blocks": [
            {"id": "scan-heading", "type": "heading", "text": "Chapter One", "reading_order": 0},
            {"id": "scan-body", "type": "paragraph", "text": "The first paragraph.", "reading_order": 1},
            {"id": "scan-figure", "type": "illustration", "text": "A map of the island.", "reading_order": 2},
        ],
        "term_mentions": [{"term": "island", "block_id": "scan-body"}],
        "illustrations": ["A map of the island."],
    }
)


def _assert_exact_segment_review_provenance(store, finding_ids, *, page_id, segment_artifacts):
    """Review requests must target and base exactly one correction-ready segment."""
    requests = [store.get_finding(finding_id) for finding_id in finding_ids]
    requests = [finding for finding in requests if finding.kind == "review_request"]
    assert requests
    for request in requests:
        subject_ids = request.evidence["applicable_subject_ids"]
        assert request.evidence["scope"] == "segment"
        assert subject_ids == list(request.subject_refs)
        assert len(subject_ids) == 1
        subject_id = subject_ids[0]
        assert subject_id != page_id
        assert subject_id in segment_artifacts
        assert request.evidence["base_artifact_ids"] == [segment_artifacts[subject_id]]
        assert request.dependency_ids == (segment_artifacts[subject_id],)


class TestExtractPage:
    @pytest.mark.asyncio
    async def test_parses_typed_blocks_term_mentions_and_legacy_text(self):
        """A valid Pi response becomes a typed PageExtraction in reading order."""
        from btran.source_extractor import extract_page, legacy_page_text

        proc = _make_mock_proc(stdout=_VALID_OUTPUT)
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ):
            extraction = await extract_page(
                image_path=Path("page-001.png"),
                model="vision-model",
                sha256="a" * 64,
                phash="b" * 16,
                page_number=1,
            )

        assert isinstance(extraction, PageExtraction)
        assert extraction.blocks == [
            SourceBlock("page_1_block_0", "heading", "Chapter One", 0),
            SourceBlock("page_1_block_1", "paragraph", "The first paragraph.", 1),
            SourceBlock("page_1_block_2", "illustration", "A map of the island.", 2),
        ]
        assert extraction.term_mentions == [
            TermMention(term="island", block_id="page_1_block_1")
        ]
        assert extraction.illustrations == ["A map of the island."]
        assert legacy_page_text(extraction) == "Chapter One\n\nThe first paragraph."
        assert extraction.page_number == 1
        assert extraction.sha256 == "a" * 64
        assert extraction.phash == "b" * 16
        assert extraction.source_lang == "en"
        assert extraction.model == "vision-model"

    @pytest.mark.asyncio
    async def test_block_ids_are_deterministic_not_model_supplied(self):
        """The same page and reading order retain canonical IDs despite Pi IDs changing."""
        from btran.source_extractor import extract_page

        first = json.dumps({
            "source_lang": "en",
            "blocks": [{"id": "arbitrary-a", "type": "paragraph", "text": "One", "reading_order": 7}],
            "term_mentions": [], "illustrations": [],
        })
        second = json.dumps({
            "source_lang": "en",
            "blocks": [{"id": "arbitrary-b", "type": "paragraph", "text": "One", "reading_order": 7}],
            "term_mentions": [], "illustrations": [],
        })
        procs = [_make_mock_proc(stdout=first), _make_mock_proc(stdout=second)]
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=procs),
        ):
            kwargs = dict(
                image_path=Path("page.png"), model="model",
                sha256="a" * 64, phash="b" * 16, page_number=9,
            )
            one = await extract_page(**kwargs)
            two = await extract_page(**kwargs)

        assert one.blocks[0].id == "page_9_block_7"
        assert two.blocks[0].id == one.blocks[0].id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "output, error",
        [
            ({"source_lang": "en", "blocks": [{"id": "a", "type": "unknown", "text": "x", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "type"),
            ({"source_lang": "en", "blocks": [{"id": "a", "type": "paragraph", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "text"),
            ({"source_lang": "en", "blocks": [{"id": "a", "type": "paragraph", "text": "x", "reading_order": 0}, {"id": "b", "type": "heading", "text": "y", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "reading_order"),
            ({"source_lang": "en", "blocks": [{"type": "paragraph", "text": "x", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "id"),
        ],
    )
    async def test_rejects_invalid_source_blocks(self, output, error):
        """Every block must carry a source ID, supported type, text, and unique order."""
        from btran.source_extractor import ExtractionError, extract_page

        proc = _make_mock_proc(stdout=json.dumps(output))
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ):
            with pytest.raises(ExtractionError, match=error):
                await extract_page(
                    Path("page.png"), "model", "a" * 64, "b" * 16, 1
                )

    @pytest.mark.asyncio
    async def test_rejects_term_mention_for_unknown_block(self):
        """Term mentions must refer to a source block from the same response."""
        from btran.source_extractor import ExtractionError, extract_page

        output = json.dumps({
            "source_lang": "en",
            "blocks": [{"id": "known", "type": "paragraph", "text": "text", "reading_order": 0}],
            "term_mentions": [{"term": "term", "block_id": "missing"}],
            "illustrations": [],
        })
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(return_value=_make_mock_proc(stdout=output)),
        ):
            with pytest.raises(ExtractionError, match="block_id"):
                await extract_page(Path("page.png"), "model", "a" * 64, "b" * 16, 1)

    @pytest.mark.asyncio
    async def test_pi_argv_carries_documented_schema_without_execution_deadline(self):
        """Pi gets exact schema prompt and extraction has no execution deadline."""
        from btran.source_extractor import EXTRACTION_PROMPT, extract_page

        proc = _make_mock_proc(stdout=_VALID_OUTPUT)
        exec_mock = AsyncMock(return_value=proc)
        with patch("btran.source_extractor.asyncio.create_subprocess_exec", exec_mock), \
             patch("btran.source_extractor.asyncio.wait_for", side_effect=AssertionError("execution deadline used")):
            await extract_page(
                Path("/photos/p1.png"), "gemini-vision", "a" * 64, "b" * 16, 1,
                pi_bin="/bin/pi",
            )

        proc.communicate.assert_awaited_once_with()

        args, kwargs = exec_mock.call_args
        assert args[:2] == ("/bin/pi", "-p")
        assert args[args.index("--model") + 1] == "gemini-vision"
        assert args[args.index("--thinking") + 1] == "low"
        assert args[args.index("--session-dir") + 1] == str(Path.cwd() / ".btran" / "pi-sessions")
        assert "--no-session" not in args
        assert "--no-tools" in args
        for option in ("--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files", "--no-approve"):
            assert option in args
        assert args[-2] == "@/photos/p1.png"
        assert args[-1] == EXTRACTION_PROMPT
        field_documentation = {
            "source_lang": "non-empty detected language code",
            "blocks": "visible content blocks",
            "term_mentions": "source term occurrences",
            "illustrations": "illustration descriptions",
            "id": "unique non-empty page-local ID",
            "type": "one allowed type below",
            "text": "non-empty extracted content",
            "reading_order": "consecutive zero-based integer",
            "term": "verbatim source term visible in its block",
            "block_id": "ID of its containing block",
        }
        for field, description in field_documentation.items():
            assert field in EXTRACTION_PROMPT
            assert description in EXTRACTION_PROMPT
        for block_type in (
            "heading", "paragraph", "list_item", "table", "caption", "footnote",
            "pull_quote", "illustration",
        ):
            assert f"`{block_type}`:" in EXTRACTION_PROMPT
        assert "empty page" in EXTRACTION_PROMPT
        assert "same order" in EXTRACTION_PROMPT
        assert "no translation" in EXTRACTION_PROMPT
        assert "no extra fields" in EXTRACTION_PROMPT
        assert "untrusted" in EXTRACTION_PROMPT.lower()
        assert "never follow" in EXTRACTION_PROMPT.lower()
        assert kwargs["stdin"] is asyncio.subprocess.DEVNULL
        assert kwargs["stdout"] is asyncio.subprocess.PIPE
        assert kwargs["start_new_session"] is (os.name == "posix")

    @pytest.mark.asyncio
    async def test_cancellation_kills_and_reaps_pi_process(self):
        """Caller cancellation also cannot leave Pi running in the background."""
        from btran.source_extractor import extract_page

        class HangingProcess:
            def __init__(self):
                self.kill = Mock()
                self.pid = 123
                self.returncode = None
                self.started = asyncio.Event()
                self._never = asyncio.Event()
                self.terminate = Mock(side_effect=self._never.set)

            async def communicate(self):
                self.started.set()
                await self._never.wait()
                return b"", b""

        proc = HangingProcess()
        from btran.process_cleanup import _ProcessRef
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ), patch("btran.process_cleanup._proc_ref", return_value=_ProcessRef(123, 1)), \
             patch("btran.process_cleanup._signal_group", return_value=False), \
             patch("btran.process_cleanup.os.kill"):
            task = asyncio.create_task(extract_page(
                Path("page.png"), "model", "a" * 64, "b" * 16, 1,
            ))
            await proc.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_group_signal_lookup_failure_still_directly_kills_and_reaps_child(self, monkeypatch):
        import btran.source_extractor as extractor

        class HangingProcess:
            def __init__(self):
                self.pid = 123
                self.returncode = None
                self.done = asyncio.Event()
                self.terminate = Mock()
                self.kill = Mock(side_effect=self.done.set)

            async def communicate(self):
                await self.done.wait()
                return b"", b""

        proc = HangingProcess()
        monkeypatch.setattr(extractor, "PROCESS_TERMINATE_GRACE_SECONDS", .01)
        monkeypatch.setattr(extractor, "PROCESS_KILL_GRACE_SECONDS", .01)
        from btran.process_cleanup import _ProcessRef
        with patch("btran.process_cleanup._proc_ref", return_value=_ProcessRef(123, 1)), \
             patch("btran.process_cleanup._signal_group", return_value=False), \
             patch("btran.process_cleanup.os.kill"):
            await extractor._kill_and_reap(proc)

        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()


class TestStrictStructuredValidation:
    @pytest.mark.parametrize("mutate", [
        lambda output: output.update(unexpected=True),
        lambda output: output["blocks"][0].update(unexpected=True),
        lambda output: output["term_mentions"][0].update(unexpected=True),
    ])
    def test_rejects_unknown_fields_at_every_structured_level(self, mutate):
        """Pi output must exactly match the extraction schema, not merely contain it."""
        from btran.source_extractor import ExtractionError, parse_extraction

        output = json.loads(_VALID_OUTPUT)
        mutate(output)

        with pytest.raises(ExtractionError, match="unexpected fields"):
            parse_extraction(
                output,
                image_path=Path("page.png"), model="model",
                sha256="a" * 64, phash="b" * 16, page_number=1,
            )

    def test_rejects_missing_detected_source_language(self):
        """A source artifact must record the language detected by the model."""
        from btran.source_extractor import ExtractionError, parse_extraction

        output = json.loads(_VALID_OUTPUT)
        output.pop("source_lang")
        with pytest.raises(ExtractionError, match="missing required fields"):
            parse_extraction(
                output, image_path=Path("page.png"), model="model",
                sha256="a" * 64, phash="b" * 16, page_number=1,
            )

    @pytest.mark.parametrize("mutate", [
        lambda output: output["blocks"][0].update(text="   "),
        lambda output: output.update(illustrations=["   "]),
        lambda output: output.update(illustrations=["different description"]),
    ])
    def test_rejects_empty_or_inconsistent_source_content(self, mutate):
        """Descriptions and source text must be meaningful and internally consistent."""
        from btran.source_extractor import ExtractionError, parse_extraction

        output = json.loads(_VALID_OUTPUT)
        mutate(output)

        with pytest.raises(ExtractionError):
            parse_extraction(
                output,
                image_path=Path("page.png"), model="model",
                sha256="a" * 64, phash="b" * 16, page_number=1,
            )


class TestExtractionArtifacts:
    def test_to_file_writes_extraction_atomically(self, tmp_path):
        """An existing artifact is atomically replaced by valid extraction JSON."""
        from btran.source_extractor import to_file

        extraction = PageExtraction(
            page_number=2, image_path="p2.png", sha256="a" * 64, phash="b" * 16,
            source_lang="en", model="model",
            blocks=[SourceBlock("page_2_block_0", "paragraph", "Text", 0)],
        )
        path = tmp_path / "nested" / "page-2.json"
        path.parent.mkdir()
        path.write_text("old artifact")

        to_file(extraction, path)

        assert PageExtraction.from_file(path) == extraction
        assert not list(path.parent.glob(".page-2.json.*.tmp"))

    def test_extraction_cache_identity_is_namespaced_and_semantic(self, monkeypatch):
        """Extraction keys include model, prompt, and schema independently of translations."""
        import btran.source_extractor as extractor

        first = extractor.extraction_cache_identity("a" * 64, "vision-a")
        assert first == extractor.extraction_cache_identity("a" * 64, "vision-a")
        assert first.startswith("extraction:")
        assert first != extractor.extraction_cache_identity("a" * 64, "vision-b")
        assert first != extractor.extraction_cache_identity("a" * 64, "vision-a", "high")

        monkeypatch.setattr(extractor, "EXTRACTION_SCHEMA_VERSION", "changed")
        assert first != extractor.extraction_cache_identity("a" * 64, "vision-a")

    def test_prompt_bytes_naturally_change_source_extraction_cache_identity(self):
        """Prompt bytes already participate in source-extraction semantic identity."""
        from btran.artifacts import source_extraction_semantic_key
        from btran.source_extractor import EXTRACTION_PROMPT, EXTRACTION_SCHEMA_VERSION

        inputs = dict(
            extraction_schema=EXTRACTION_SCHEMA_VERSION,
            model_executable_identity="pi-bin:pi", model_id="vision-a", raw_bytes=b"page",
        )
        first = source_extraction_semantic_key(
            prompt_bytes=EXTRACTION_PROMPT.encode("utf-8"), **inputs,
        )
        changed = source_extraction_semantic_key(
            prompt_bytes=(EXTRACTION_PROMPT + "\nchanged").encode("utf-8"), **inputs,
        )
        assert first != changed


class TestPersistedRawLeaves:
    @pytest.mark.asyncio
    async def test_model_rejection_of_unreadable_input_is_raw_diagnostic_fallback(self, tmp_path):
        from btran.artifacts import ArtifactStore
        from btran.identity import page_id_for_raw_sha256
        from btran.source_extractor import RawPageInput, extract_raw_pages

        image = tmp_path / "accepted.png"
        image.write_bytes(b"not a decodable image")
        digest = __import__("hashlib").sha256(image.read_bytes()).hexdigest()
        page_id = page_id_for_raw_sha256(digest)
        store = ArtifactStore(tmp_path / "state")

        model = AsyncMock(side_effect=RuntimeError("unreadable image"))
        with patch("btran.source_extractor.extract_page", model):
            result = await extract_raw_pages(
                [RawPageInput(page_id, image, digest)], store=store,
                workspace=tmp_path / "state", model="vision", base_revision_id="base-rev",
            )

        model.assert_awaited_once()
        leaf = result.leaves[0]
        artifact = store.get(leaf.page_artifact_id)
        assert result.status == "degraded"
        assert artifact.kind == "DiagnosticSourceFallback"
        assert artifact.payload["source_lang"] is None
        assert "effective_content_id" not in artifact.payload
        assert "[btran diagnostic: source_extraction_failed:" in artifact.payload["segment"]["source_text"]
        from btran.artifacts import source_extraction_semantic_key
        from btran.source_extractor import EXTRACTION_PROMPT, EXTRACTION_SCHEMA_VERSION
        assert artifact.semantic_key == source_extraction_semantic_key(
            extraction_schema=EXTRACTION_SCHEMA_VERSION,
            prompt_bytes=EXTRACTION_PROMPT.encode("utf-8"),
            model_executable_identity="pi-bin:pi", model_id="vision",
            raw_bytes=image.read_bytes(),
        )
        requests = [store.get_finding(item) for item in leaf.finding_ids]
        _assert_exact_segment_review_provenance(
            store, leaf.finding_ids, page_id=page_id,
            segment_artifacts={artifact.payload["segment"]["segment_id"]: artifact.artifact_id},
        )
        request = next(item for item in requests if item.kind == "review_request")
        assert request.requires_action is False
        assert request.evidence == {
            "trigger": "degraded_unknown_confidence",
            "suggested_correction_kind": "source_text",
            "applicable_subject_ids": [artifact.payload["segment"]["segment_id"]],
            "base_revision_id": "base-rev",
            "base_artifact_ids": [artifact.artifact_id],
            "scope": "segment",
        }

    @pytest.mark.asyncio
    async def test_hash_mismatch_fallback_key_uses_actual_raw_bytes(self, tmp_path):
        from hashlib import sha256

        from btran.artifacts import ArtifactStore, source_extraction_semantic_key
        from btran.identity import page_id_for_raw_sha256
        from btran.source_extractor import (
            EXTRACTION_PROMPT,
            EXTRACTION_SCHEMA_VERSION,
            RawPageInput,
            extract_raw_pages,
        )

        actual_raw = b"raw bytes retained despite claimed digest mismatch"
        claimed_digest = sha256(b"different accepted bytes").hexdigest()
        store = ArtifactStore(tmp_path / "state")
        result = await extract_raw_pages([
            RawPageInput(page_id_for_raw_sha256(claimed_digest), tmp_path / "missing.png",
                         claimed_digest, raw_bytes=actual_raw)
        ], store=store, workspace=tmp_path / "state", model="vision")

        artifact = store.get(result.leaves[0].page_artifact_id)
        assert result.status == "degraded"
        assert artifact.semantic_key == source_extraction_semantic_key(
            extraction_schema=EXTRACTION_SCHEMA_VERSION,
            prompt_bytes=EXTRACTION_PROMPT.encode("utf-8"),
            model_executable_identity="pi-bin:pi",
            model_id="vision",
            raw_bytes=actual_raw,
        )

    @pytest.mark.asyncio
    async def test_invalid_retry_bound_rejects_before_decode_failure_fallback(self, tmp_path):
        from hashlib import sha256

        from btran.artifacts import ArtifactStore
        from btran.identity import page_id_for_raw_sha256
        from btran.source_extractor import ExtractionError, RawPageInput, extract_raw_pages

        raw = b"undecodable accepted input"
        digest = sha256(raw).hexdigest()
        store = ArtifactStore(tmp_path / "state")
        with pytest.raises(ExtractionError, match="max_retries must be an integer between 0 and 5"):
            await extract_raw_pages([
                RawPageInput(page_id_for_raw_sha256(digest), tmp_path / "missing.png", digest,
                             raw_bytes=raw)
            ], store=store, workspace=tmp_path / "state", model="vision", max_retries=6)

        assert not list(store.artifacts_dir.iterdir())
        assert not list(store.findings_dir.iterdir())

    @pytest.mark.asyncio
    async def test_low_confidence_page_aggregation_keeps_segment_review_provenance(self, tmp_path):
        from btran.artifacts import ArtifactStore, DependencyGraph
        from btran.identity import page_id_for_raw_sha256
        from btran.source_extractor import RawPageInput, extract_raw_pages, materialize_effective_source
        from PIL import Image

        image = tmp_path / "page.png"
        Image.new("RGB", (600, 600), "white").save(image)
        digest = __import__("hashlib").sha256(image.read_bytes()).hexdigest()
        page_id = page_id_for_raw_sha256(digest)
        store = ArtifactStore(tmp_path / "state")
        proc = _make_mock_proc(stdout=_VALID_OUTPUT)
        with patch("btran.source_extractor.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await extract_raw_pages(
                [RawPageInput(page_id, image, digest, confidence=.5)], store=store,
                workspace=tmp_path / "state", model="vision", base_revision_id="base-rev",
            )

        findings = [store.get_finding(item) for item in result.leaves[0].finding_ids]
        requests = [item for item in findings if item.kind == "review_request"]
        assert requests
        assert all(item.requires_action is False for item in requests)
        assert all(item.evidence["trigger"] == "low_confidence" for item in requests)
        assert all(item.evidence["base_revision_id"] == "base-rev" for item in requests)
        assert all(item.evidence["base_artifact_ids"] for item in requests)
        raw_segments = {
            store.get(artifact_id).payload["segment_id"]: artifact_id
            for artifact_id in result.leaves[0].segment_artifact_ids
        }
        _assert_exact_segment_review_provenance(
            store, result.leaves[0].finding_ids, page_id=page_id, segment_artifacts=raw_segments,
        )

        effective = materialize_effective_source(
            result, store=store, graph=DependencyGraph(tmp_path / "state"), base_revision_id="base-rev",
        )
        effective_segments = {
            store.get(artifact_id).payload["segment_id"]: artifact_id
            for artifact_id in effective.leaves[0].segment_artifact_ids
        }
        _assert_exact_segment_review_provenance(
            store, effective.leaves[0].finding_ids, page_id=page_id,
            segment_artifacts=effective_segments,
        )


class TestEffectiveSourceMaterialization:
    @pytest.mark.asyncio
    async def test_native_effective_source_preserves_language_and_applies_selected_overlay(self, tmp_path):
        from btran.artifacts import ArtifactStore, DependencyGraph
        from btran.corrections import OverlayInput
        from btran.identity import page_id_for_raw_sha256
        from btran.source_extractor import RawPageInput, extract_raw_pages, materialize_effective_source
        from PIL import Image

        image = tmp_path / "page.png"
        Image.new("RGB", (600, 600), "white").save(image)
        digest = __import__("hashlib").sha256(image.read_bytes()).hexdigest()
        page_id = page_id_for_raw_sha256(digest)
        store = ArtifactStore(tmp_path / "state")
        graph = DependencyGraph(tmp_path / "state")
        with patch("btran.source_extractor.asyncio.create_subprocess_exec", AsyncMock(return_value=_make_mock_proc(stdout=_VALID_OUTPUT))):
            raw = await extract_raw_pages(
                [RawPageInput(page_id, image, digest)], store=store, workspace=tmp_path / "state",
                model="vision", base_revision_id="base-revision",
            )

        raw_segment_id = raw.leaves[0].segment_artifact_ids[0]
        segment = store.get(raw_segment_id).payload
        overlay = OverlayInput(
            correction_id="source-correction", kind="source_text", subject_id=segment["segment_id"],
            replacement="Corrected source text", base_artifact_ids=(raw_segment_id,),
            scope={"segment_id": segment["segment_id"]},
        )
        result = materialize_effective_source(
            raw, store=store, graph=graph, source_overlays=(overlay,), base_revision_id="base-revision",
        )

        assert result.status == "completed"
        assert len(result.leaves) == 1
        leaf = result.leaves[0]
        page = store.get(leaf.page_artifact_id)
        assert page.kind == "EffectiveSourcePage"
        assert page.payload["source_langs"] == ["en"]
        effective = [store.get(artifact_id) for artifact_id in leaf.segment_artifact_ids]
        corrected = next(item for item in effective if item.payload["segment_id"] == segment["segment_id"])
        assert corrected.payload["source_text"] == "Corrected source text"
        assert corrected.payload["effective_text"] == "Corrected source text"
        assert corrected.payload["source_lang"] == corrected.payload["render_lang"] == "en"
        assert corrected.payload["translation_artifact_id"] is None
        assert corrected.payload["target_overlay_artifact_id"] is None
        assert corrected.payload["correction_ids"] == ["source-correction"]
        overlay_id = corrected.payload["source_overlay_artifact_id"]
        assert store.get(overlay_id).kind == "SourceTextOverlay"
        edges = [graph.get(edge_id) for edge_id in result.graph_edge_ids]
        assert {(edge.parent_artifact_id, edge.child_artifact_id, edge.edge_kind) for edge in edges} >= {
            (raw_segment_id, corrected.artifact_id, "raw_extraction_to_effective_source"),
            (overlay_id, corrected.artifact_id, "source_overlay_to_effective_source"),
        }

    @pytest.mark.asyncio
    async def test_raw_diagnostic_fallback_becomes_only_und_effective_diagnostic_with_review(self, tmp_path):
        from btran.artifacts import ArtifactStore, DependencyGraph
        from btran.identity import page_id_for_raw_sha256
        from btran.source_extractor import RawPageInput, extract_raw_pages, materialize_effective_source

        image = tmp_path / "bad.png"
        image.write_bytes(b"not an image")
        digest = __import__("hashlib").sha256(image.read_bytes()).hexdigest()
        store = ArtifactStore(tmp_path / "state")
        raw = await extract_raw_pages(
            [RawPageInput(page_id_for_raw_sha256(digest), image, digest)], store=store,
            workspace=tmp_path / "state", model="vision", base_revision_id="base-revision",
        )
        graph = DependencyGraph(tmp_path / "state")
        result = materialize_effective_source(raw, store=store, graph=graph, base_revision_id="base-revision")

        assert result.status == "degraded"
        leaf = result.leaves[0]
        effective = store.get(leaf.segment_artifact_ids[0])
        assert effective.kind == "DiagnosticEffectiveSourceSegment"
        assert effective.payload["source_lang"] is None
        assert effective.payload["render_lang"] == "und"
        assert effective.payload["effective_text"] == effective.payload["source_text"]
        raw_fallback = store.get(raw.leaves[0].page_artifact_id)
        _assert_exact_segment_review_provenance(
            store, raw.leaves[0].finding_ids, page_id=raw.leaves[0].page_id,
            segment_artifacts={raw_fallback.payload["segment"]["segment_id"]: raw_fallback.artifact_id},
        )
        _assert_exact_segment_review_provenance(
            store, leaf.finding_ids, page_id=leaf.page_id,
            segment_artifacts={effective.payload["segment_id"]: effective.artifact_id},
        )
        requests = [store.get_finding(item) for item in leaf.finding_ids]
        assert {request.evidence["trigger"] for request in requests if request.kind == "review_request"} == {
            "degraded_unknown_confidence"
        }
        edges = [graph.get(edge_id) for edge_id in result.graph_edge_ids]
        assert any(
            edge.edge_kind == "raw_fallback_to_effective_source"
            and edge.parent_artifact_id == raw.leaves[0].page_artifact_id
            and edge.child_artifact_id == effective.artifact_id
            for edge in edges
        )

    @pytest.mark.asyncio
    async def test_task10_native_target_keeps_task7_text_and_makes_no_translation_call(self, tmp_path):
        from btran.artifacts import ArtifactStore, DependencyGraph
        from btran.identity import page_id_for_raw_sha256
        from btran.source_extractor import RawPageInput, extract_raw_pages, materialize_effective_source
        from btran.translator import materialize_effective_target
        from PIL import Image

        image = tmp_path / "page.png"
        Image.new("RGB", (600, 600), "white").save(image)
        digest = __import__("hashlib").sha256(image.read_bytes()).hexdigest()
        store = ArtifactStore(tmp_path / "state")
        graph = DependencyGraph(tmp_path / "state")
        with patch("btran.source_extractor.asyncio.create_subprocess_exec", AsyncMock(return_value=_make_mock_proc(stdout=_VALID_OUTPUT))):
            raw = await extract_raw_pages([RawPageInput(page_id_for_raw_sha256(digest), image, digest)],
                                          store=store, workspace=tmp_path / "state", model="vision")
        source = materialize_effective_source(raw, store=store, graph=graph)

        async def forbidden_model(_):
            raise AssertionError("native target materialization must not call translation")

        target = await materialize_effective_target(source, store=store, graph=graph, mode="native",
                                                    translation_call=forbidden_model)
        source_texts = [store.get(item).payload["effective_text"] for item in source.leaves[0].segment_artifact_ids]
        target_segments = [store.get(item).payload for item in target.leaves[0].segment_artifact_ids]
        assert [item["effective_text"] for item in target_segments] == source_texts
        assert all(item["render_lang"] == item["source_lang"] for item in target_segments)
