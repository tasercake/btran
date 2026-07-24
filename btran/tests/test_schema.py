"""Tests for btran.schema — intermediate translation result data model."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from btran.schema import (
    ErrorResult,
    Manifest,
    PageExtraction,
    PageResult,
    SourceBlock,
    TermMention,
    TerminologyEntry,
    TerminologyMap,
    TranslatedBlock,
)


# ---------------------------------------------------------------------------
# PageResult tests
# ---------------------------------------------------------------------------

class TestPageResult:
    def test_round_trip_to_from_dict(self):
        """Create → to_dict → from_dict → equality."""
        pr = PageResult(
            page_number=3,
            image_path="scans/chapter1/page_003.png",
            sha256="a" * 64,
            phash="b" * 16,
            source_lang="ja",
            target_lang="en",
            page_text="こんにちは世界",
            translated_text="Hello world",
            image_descriptions=["A book page with Japanese text"],
            model="gpt-4o",
            timestamp="2025-01-15T10:30:00+00:00",
            retry_count=0,
        )
        d = pr.to_dict()
        pr2 = PageResult.from_dict(d)
        assert pr == pr2

    def test_file_io_round_trip(self):
        """to_file → from_file round-trips faithfully."""
        pr = PageResult(
            page_number=7,
            image_path="scans/ch2/page_007.png",
            sha256="c" * 64,
            phash="d" * 16,
            source_lang="fr",
            target_lang="de",
            page_text="Bonjour le monde",
            translated_text="Hallo Welt",
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp = Path(f.name)

        try:
            pr.to_file(tmp)
            pr2 = PageResult.from_file(tmp)
            assert pr == pr2
        finally:
            tmp.unlink(missing_ok=True)

    def test_from_dict_missing_fields_use_defaults(self):
        """from_dict must be forgiving — missing fields → defaults."""
        d: dict = {
            "page_number": 1,
            "image_path": "img.jpg",
            "sha256": "e" * 64,
            "phash": "f" * 16,
            "source_lang": "en",
            "target_lang": "es",
            "page_text": "Hello",
            "translated_text": "Hola",
        }
        pr = PageResult.from_dict(d)
        assert pr.image_descriptions == []
        assert pr.model == ""
        assert pr.retry_count == 0
        # timestamp auto-populated (or empty if from_dict doesn't trigger __post_init__)
        # __post_init__ runs on construction; from_dict creates via cls(**...) so it will run
        assert pr.timestamp != ""

    def test_timestamp_auto_populated_on_construction(self):
        """If timestamp is empty, __post_init__ sets it to now (UTC ISO 8601)."""
        pr = PageResult(
            page_number=1,
            image_path="x.png",
            sha256="a" * 64,
            phash="b" * 16,
            source_lang="en",
            target_lang="fr",
            page_text="Hi",
            translated_text="Salut",
        )
        # Should have been auto-populated
        ts = datetime.fromisoformat(pr.timestamp)
        assert ts.tzinfo is not None  # timezone-aware
        now = datetime.now(timezone.utc)
        # Should be within a few seconds of now
        delta = abs((now - ts).total_seconds())
        assert delta < 5

    def test_to_dict_excludes_internal_fields(self):
        """to_dict returns a plain dict with no underscored keys."""
        pr = PageResult(
            page_number=2,
            image_path="a.png",
            sha256="a" * 64,
            phash="b" * 16,
            source_lang="en",
            target_lang="ja",
            page_text="Hello",
            translated_text="こんにちは",
        )
        d = pr.to_dict()
        # All keys should be plain field names — no __dataclass_fields__ etc.
        assert all(not k.startswith("_") for k in d)
        assert "page_number" in d
        assert "image_path" in d
        assert "sha256" in d
        assert "phash" in d

    def test_to_dict_contains_all_required_fields(self):
        """Every dataclass field appears in the dict."""
        pr = PageResult(
            page_number=1,
            image_path="img.png",
            sha256="a" * 64,
            phash="b" * 16,
            source_lang="en",
            target_lang="es",
            page_text="text",
            translated_text="texto",
        )
        d = pr.to_dict()
        expected_keys = {
            "page_number", "image_path", "sha256", "phash",
            "source_lang", "target_lang", "page_text", "translated_text",
            "image_descriptions", "model", "timestamp", "retry_count",
            "blocks", "translated_blocks", "term_mentions", "illustrations",
        }
        assert set(d.keys()) == expected_keys

    def test_to_file_writes_pretty_json(self):
        """to_file writes readable, indented JSON."""
        pr = PageResult(
            page_number=42,
            image_path="scans/last.png",
            sha256="f" * 64,
            phash="0" * 16,
            source_lang="ko",
            target_lang="en",
            page_text="안녕하세요",
            translated_text="Hello",
            image_descriptions=["A page of Korean text", "Novel page"],
            model="claude-sonnet-4-20250514",
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp = Path(f.name)

        try:
            pr.to_file(tmp)
            raw = tmp.read_text()
            assert "page_number" in raw
            # It must be valid JSON
            parsed = json.loads(raw)
            assert parsed["page_number"] == 42
            # Pretty-printed: should contain newlines and indentation
            assert "\n" in raw
            assert "  " in raw
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ErrorResult tests
# ---------------------------------------------------------------------------

class TestErrorResult:
    def test_minimal_construction(self):
        er = ErrorResult(
            page_number=5,
            image_path="bad_scan.png",
            error="OCR failed: unreadable image",
        )
        assert er.page_number == 5
        assert er.image_path == "bad_scan.png"
        assert er.error == "OCR failed: unreadable image"
        assert er.retry_count == 0
        assert er.model == ""

    def test_round_trip_to_from_dict(self):
        er = ErrorResult(
            page_number=99,
            image_path="missing.png",
            error="File not found",
            retry_count=3,
            model="gpt-4o",
        )
        d = er.to_dict()
        er2 = ErrorResult.from_dict(d)
        assert er == er2

    def test_from_dict_missing_fields_use_defaults(self):
        d = {
            "page_number": 1,
            "image_path": "a.png",
            "error": "boom",
        }
        er = ErrorResult.from_dict(d)
        assert er.retry_count == 0
        assert er.model == ""


# ---------------------------------------------------------------------------
# Gate 0 structural-contract tests
# ---------------------------------------------------------------------------

class TestSourceBlock:
    def test_construction_and_equality(self, sample_source_block):
        assert sample_source_block == SourceBlock(
            id="page_1_block_0", type="heading", text="Chapter 1", reading_order=0
        )

    def test_to_dict_contains_all_fields(self, sample_source_block):
        assert sample_source_block.to_dict() == {
            "id": "page_1_block_0", "type": "heading", "text": "Chapter 1", "reading_order": 0
        }


class TestTermMention:
    def test_construction_and_field_access(self):
        mention = TermMention(term="continuation", block_id="page_4_block_2")
        assert mention.term == "continuation"
        assert mention.block_id == "page_4_block_2"


class TestPageExtraction:
    def test_round_trip_to_from_dict(self, sample_page_extraction):
        assert PageExtraction.from_dict(sample_page_extraction.to_dict()) == sample_page_extraction

    def test_file_io_round_trip(self, sample_page_extraction, tmp_path):
        path = tmp_path / "nested" / "extraction.json"
        sample_page_extraction.to_file(path)
        assert PageExtraction.from_file(path) == sample_page_extraction

    def test_defaults_are_empty_collections(self):
        extraction = PageExtraction(
            page_number=1, image_path="page.jpg", sha256="a" * 64, phash="b" * 16,
            source_lang="en", model="model", timestamp="",
        )
        assert extraction.blocks == []
        assert extraction.term_mentions == []
        assert extraction.illustrations == []

    def test_timestamp_is_auto_populated(self):
        extraction = PageExtraction(
            page_number=1, image_path="page.jpg", sha256="a" * 64, phash="b" * 16,
            source_lang="en", model="model", timestamp="",
        )
        assert datetime.fromisoformat(extraction.timestamp).tzinfo is not None


class TestTranslatedBlock:
    def test_construction_preserves_matching_block_id(self, sample_translated_block):
        assert sample_translated_block.block_id == "p1_b0"
        assert sample_translated_block.translated_text == "Bonjour le monde"


class TestTerminologyEntry:
    def test_construction_and_default_notes(self):
        entry = TerminologyEntry(
            concept_id="c1", source_terms=["hello"], target_term="bonjour",
            provenance=["hello"], confidence=0.95,
        )
        assert entry.notes == ""
        assert 0.0 <= entry.confidence <= 1.0


class TestTerminologyMap:
    def test_round_trip_to_from_dict(self, sample_terminology_map):
        assert TerminologyMap.from_dict(sample_terminology_map.to_dict()) == sample_terminology_map

    def test_file_io_round_trip(self, sample_terminology_map, tmp_path):
        path = tmp_path / "nested" / "glossary.json"
        sample_terminology_map.to_file(path)
        assert TerminologyMap.from_file(path) == sample_terminology_map

    def test_hash_is_stable_through_serialization(self, sample_terminology_map):
        assert TerminologyMap.from_dict(sample_terminology_map.to_dict()).hash == "abc123"

    def test_timestamp_is_auto_populated(self):
        glossary = TerminologyMap(
            version="1.0.0", hash="abc", source_lang="en", target_lang="fr", entries=[], created_at=""
        )
        assert datetime.fromisoformat(glossary.created_at).tzinfo is not None


class TestManifest:
    def test_round_trip_to_from_dict(self, sample_manifest):
        assert Manifest.from_dict(sample_manifest.to_dict()) == sample_manifest

    def test_file_io_round_trip(self, sample_manifest, tmp_path):
        path = tmp_path / "nested" / "manifest.json"
        sample_manifest.to_file(path)
        assert Manifest.from_file(path) == sample_manifest

    def test_total_pages_matches_pages_length(self, sample_manifest):
        assert sample_manifest.total_pages == len(sample_manifest.pages)


class TestPageResultExtended:
    def test_new_fields_default_to_empty_lists(self):
        result = PageResult(page_number=1, sha256="a" * 64, phash="b" * 16)
        assert result.blocks == []
        assert result.translated_blocks == []
        assert result.term_mentions == []
        assert result.illustrations == []

    def test_from_dict_without_new_fields_remains_backward_compatible(self):
        result = PageResult.from_dict({
            "page_number": 1, "sha256": "a" * 64, "phash": "b" * 16,
        })
        assert result.blocks == []
        assert result.translated_blocks == []
        assert result.term_mentions == []
        assert result.illustrations == []
