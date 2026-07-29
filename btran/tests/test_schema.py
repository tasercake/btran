"""Tests for strict canonical records and legacy migration readers."""

import json
from datetime import datetime, timezone

import pytest

from btran.schema import (
    ArtifactEnvelope,
    CorrectionImpact,
    EffectivePage,
    EffectiveSegment,
    ErrorResult,
    Finding,
    Manifest,
    PageExtraction,
    PageResult,
    RunReport,
    SchemaError,
    SourceBlock,
    StageRecord,
    TermMention,
    TerminologyEntry,
    TerminologyMap,
    TranslatedBlock,
    canonical_json,
)


def _page_result_data(**overrides):
    data = {
        "page_number": 3,
        "image_path": "scans/chapter1/page_003.png",
        "sha256": "a" * 64,
        "phash": "b" * 16,
        "source_lang": "ja",
        "target_lang": "en",
        "page_text": "こんにちは世界",
        "translated_text": "Hello world",
        "image_descriptions": ["A book page with Japanese text"],
        "model": "gpt-4o",
        "timestamp": "2025-01-15T10:30:00+00:00",
        "retry_count": 0,
        "blocks": [],
        "translated_blocks": [],
        "term_mentions": [],
        "illustrations": [],
    }
    data.update(overrides)
    return data


class TestLegacyMigrationReaders:
    @pytest.mark.parametrize("record", (Manifest, PageExtraction, TranslatedBlock, PageResult))
    def test_read_only_records_expose_no_writers(self, record):
        assert not hasattr(record, "to_dict")
        assert not hasattr(record, "to_file")

    def test_page_result_from_dict_preserves_legacy_values(self):
        result = PageResult.from_dict(_page_result_data())
        assert result.page_text == "こんにちは世界"
        assert result.translated_text == "Hello world"
        assert result.image_descriptions == ["A book page with Japanese text"]

    def test_page_result_from_file_reads_legacy_json(self, tmp_path):
        path = tmp_path / "result.json"
        path.write_text(json.dumps(_page_result_data()), encoding="utf-8")
        assert PageResult.from_file(path).page_number == 3

    def test_page_result_missing_fields_keep_legacy_defaults(self):
        result = PageResult.from_dict({
            "page_number": 1, "image_path": "img.jpg", "sha256": "e" * 64,
            "phash": "f" * 16, "source_lang": "en", "target_lang": "es",
            "page_text": "Hello", "translated_text": "Hola",
        })
        assert result.image_descriptions == []
        assert result.model == ""
        assert result.retry_count == 0
        assert result.timestamp != ""

    def test_page_result_timestamp_auto_populates(self):
        result = PageResult(page_number=1, sha256="a" * 64, phash="b" * 16)
        timestamp = datetime.fromisoformat(result.timestamp)
        assert timestamp.tzinfo is not None
        assert abs((datetime.now(timezone.utc) - timestamp).total_seconds()) < 5

    def test_page_extraction_from_dict_and_file_read_legacy_json(self, tmp_path):
        data = {
            "page_number": 1, "image_path": "test.jpg", "sha256": "a" * 64,
            "phash": "b" * 16, "source_lang": "en", "model": "test-model",
            "timestamp": "2026-01-01T00:00:00Z",
            "blocks": [{"id": "p1_b0", "type": "paragraph", "text": "Hello", "reading_order": 0}],
            "term_mentions": [{"term": "hello", "block_id": "p1_b0"}],
            "illustrations": [],
        }
        extraction = PageExtraction.from_dict(data)
        assert extraction.blocks == [SourceBlock(id="p1_b0", type="paragraph", text="Hello", reading_order=0)]
        assert extraction.term_mentions == [TermMention(term="hello", block_id="p1_b0")]
        path = tmp_path / "extraction.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert PageExtraction.from_file(path) == extraction

    def test_page_extraction_legacy_defaults_and_timestamp(self):
        extraction = PageExtraction.from_dict({
            "page_number": 1, "image_path": "page.jpg", "sha256": "a" * 64,
            "phash": "b" * 16, "source_lang": "en", "model": "model",
        })
        assert extraction.blocks == []
        assert extraction.term_mentions == []
        assert extraction.illustrations == []
        assert datetime.fromisoformat(extraction.timestamp).tzinfo is not None

    def test_manifest_and_translated_block_read_legacy_json(self, tmp_path):
        manifest_data = {
            "input_dir": "/tmp/books",
            "pages": [{"filename": "page_001.jpg", "page_number": 1}],
            "total_pages": 1,
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        assert Manifest.from_file(manifest_path) == Manifest.from_dict(manifest_data)
        assert TranslatedBlock.from_dict({"block_id": "p1_b0", "translated_text": "Bonjour"}).translated_text == "Bonjour"


class TestErrorResult:
    def test_minimal_construction(self):
        result = ErrorResult(page_number=5, image_path="bad_scan.png", error="OCR failed: unreadable image")
        assert result.retry_count == 0
        assert result.model == ""

    def test_round_trip_to_from_dict(self):
        result = ErrorResult(page_number=99, image_path="missing.png", error="File not found", retry_count=3, model="gpt-4o")
        assert ErrorResult.from_dict(result.to_dict()) == result

    def test_from_dict_missing_fields_use_defaults(self):
        result = ErrorResult.from_dict({"page_number": 1, "image_path": "a.png", "error": "boom"})
        assert result.retry_count == 0
        assert result.model == ""


class TestSourceBlock:
    def test_construction_and_equality(self, sample_source_block):
        assert sample_source_block == SourceBlock(id="page_1_block_0", type="heading", text="Chapter 1", reading_order=0)

    def test_to_dict_contains_all_fields(self, sample_source_block):
        assert sample_source_block.to_dict() == {
            "id": "page_1_block_0", "type": "heading", "text": "Chapter 1", "reading_order": 0,
        }


class TestTerminologyMigrationRecords:
    def test_term_mention_construction(self):
        assert TermMention(term="continuation", block_id="page_4_block_2").term == "continuation"

    def test_terminology_entry_default_notes(self):
        entry = TerminologyEntry(
            concept_id="c1", source_terms=["hello"], target_term="bonjour",
            provenance=["hello"], confidence=0.95,
        )
        assert entry.notes == ""

    def test_terminology_map_round_trip(self, sample_terminology_map, tmp_path):
        assert TerminologyMap.from_dict(sample_terminology_map.to_dict()) == sample_terminology_map
        path = tmp_path / "glossary.json"
        sample_terminology_map.to_file(path)
        assert TerminologyMap.from_file(path) == sample_terminology_map


class TestCanonicalSchemas:
    def test_canonical_json_is_nfc_sorted_and_compact(self):
        assert canonical_json({"z": "e\u0301", "a": ["x"]}) == '{"a":["x"],"z":"é"}'

    @pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
    def test_artifact_envelope_rejects_non_finite_structured_values(self, value):
        envelope = ArtifactEnvelope(artifact_id="artifact-1", kind="test", payload={}, semantic_key="key-1")
        data = envelope.to_dict()
        data["payload"] = {"nested": [value]}
        with pytest.raises(SchemaError, match="non-finite"):
            ArtifactEnvelope.from_dict(data)

    def test_finding_is_deterministic_non_gating_and_strict(self):
        finding = Finding(
            kind="stage_summary", severity="info", stage="extract",
            subject_refs=("page-1",), evidence={"counts": {"ok": 1}},
            message="done", dependency_ids=("artifact-1",),
        )
        restored = Finding.from_json(finding.to_json())
        assert restored == finding
        assert restored.requires_action is False
        with pytest.raises(SchemaError):
            Finding.from_dict({**finding.to_dict(), "unknown": True})
        with pytest.raises(SchemaError):
            Finding(kind="uncertainty", severity="warning", stage="extract", requires_action=True)

    def test_run_report_preserves_nested_stage_records(self):
        stage = StageRecord(
            stage="extract", finding_ids=("finding-summary",),
            stage_summary_finding_id="finding-summary",
        )
        report = RunReport(
            run_id="run-1", final_epub_status="completed", stage_records=(stage,),
        )
        restored = RunReport.from_json(report.to_json())
        assert restored == report
        assert isinstance(restored.stage_records[0], StageRecord)
        assert restored.stage_records == (stage,)

    def test_correction_impact_regenerated_is_empty_until_execution_phase(self):
        entry = {"stage": "translation", "subject_id": "segment-1", "base_artifact_id": "artifact-1"}
        correction = CorrectionImpact(
            base_revision_id="revision-1", projection_plan_id="plan-1",
            projected_universe=(entry,), affected=(entry,),
        )
        assert correction.phase == "correction"
        assert correction.to_dict()["regenerated"] == []

        persisted = correction.to_dict()
        persisted["regenerated"] = [entry]
        with pytest.raises(SchemaError, match="non-executing.*regenerated.*empty"):
            CorrectionImpact.from_dict(persisted)

        execution = CorrectionImpact(
            phase="execution", base_revision_id="revision-1", projection_plan_id="plan-1",
            projected_universe=(entry,), affected=(entry,), regenerated=(entry,),
        )
        assert execution.regenerated == (entry,)

    def test_native_and_translated_effective_segment_invariants(self):
        native = EffectiveSegment(
            effective_segment_id="effective-1", segment_id="segment-1", source_lang="ja",
            source_text="本文", effective_text="本文", render_lang="ja", mode="native",
        )
        assert native.to_dict()["effective_text"] == "本文"
        with pytest.raises(SchemaError):
            EffectiveSegment(
                effective_segment_id="effective-1", segment_id="segment-1", source_lang="ja",
                source_text="本文", effective_text="changed", render_lang="ja", mode="native",
            )
        translated = EffectiveSegment(
            effective_segment_id="effective-2", segment_id="segment-1", source_lang="ja",
            source_text="本文", effective_text="text", render_lang="en", mode="translated",
            translation_artifact_id="translation-1",
        )
        assert translated.render_lang == "en"

    def test_effective_page_keeps_ordered_segments_and_sorted_languages(self):
        page = EffectivePage(
            effective_page_id="effective-page", page_id="page", effective_segment_ids=("s2", "s1"),
            source_langs=("en", "ja"), display_metadata={},
        )
        assert page.effective_segment_ids == ("s2", "s1")
        with pytest.raises(SchemaError):
            EffectivePage(
                effective_page_id="effective-page", page_id="page", effective_segment_ids=("s1",),
                source_langs=("ja", "en"), display_metadata={},
            )
