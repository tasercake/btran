"""Focused tests for the frozen FC8 baseline fixture and harness."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import btran.benchmark_baseline as baseline

from btran.benchmark_baseline import (
    BASELINE_PATH,
    CORPUS_DIR,
    FIXTURE_VERSION,
    PNG_BYTES,
    _ROWS,
    _TERMINOLOGY,
    canonical_quality_bytes,
    consolidation_callback,
    expected_dirty_set,
    fixture_manifest,
    load_corpus,
    page_extraction,
    state_measure,
    translation_callback,
)


def test_frozen_corpus_images_and_manifest():
    corpus = load_corpus()
    assert corpus.page_ids == ("p01", "p02", "p03", "p04", "p05", "p06")
    assert all((CORPUS_DIR / f"page-{number:02d}.png").read_bytes() == PNG_BYTES for number in range(1, 7))
    manifest = fixture_manifest()
    assert manifest["version"] == FIXTURE_VERSION
    assert len(manifest["files"]) == 10
    assert all(item["bytes"] == len(PNG_BYTES) for item in manifest["files"] if item["path"].endswith(".png"))
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_frozen_baseline_has_canonical_quality_and_oracles():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["version"] == FIXTURE_VERSION
    assert baseline["fixture_sha256_manifest"] == fixture_manifest()
    assert set(baseline["quality"]) == {"native", "translated"}
    for mode in baseline["quality"]:
        data = baseline["quality"][mode]["canonical_json"].encode("utf-8")
        assert data == canonical_quality_bytes(mode)
        assert hashlib.sha256(data).hexdigest() == baseline["quality"][mode]["sha256"]
    assert baseline["oracles"]["correction"]["block_id"] == "p05:2"
    assert baseline["oracles"]["correction"]["start"] == 3
    assert baseline["oracles"]["correction"]["end"] == 9
    assert baseline["oracles"]["excluded_term"] == "the"


def test_page_extraction_freezes_declared_categories_ids_and_unmapped_illustration():
    for number, row in enumerate(_ROWS, 1):
        extraction = page_extraction(number, CORPUS_DIR / f"page-{number:02d}.png")
        assert [block.id for block in extraction.blocks] == [f"{row['page_id']}:{n}" for n in (1, 2, 3)]
        assert [(mention.term, mention.block_id, mention.category) for mention in extraction.term_mentions] == [
            (term, f"{row['page_id']}:1" if term in row["source"][0] else f"{row['page_id']}:2", category)
            for term, category in row["mentions"]
        ]
        assert extraction.illustrations == [{"block_id": f"{row['page_id']}:4", "path": f"page-{number:02d}.png", "alt": f"benchmark illustration {number:02d}"}]


def test_extraction_and_consolidation_callbacks_have_frozen_waits(monkeypatch):
    waits: list[float] = []

    async def fake_async_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("btran.benchmark_baseline.asyncio.sleep", fake_async_sleep)
    asyncio.run(baseline.async_extraction_callback(str(CORPUS_DIR / "page-01.png"), "fc8-vision", "", "", 1))
    assert waits == [10.244]
    sync_waits: list[float] = []
    monkeypatch.setattr("btran.benchmark_baseline.time.sleep", sync_waits.append)
    consolidation_callback("prefix\n" + json.dumps({"items": []}))
    assert sync_waits == [0.131]


def test_callbacks_return_every_frozen_translation_and_exact_page_waits(monkeypatch):
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("btran.benchmark_baseline.asyncio.sleep", fake_sleep)
    # Three segments per page must still produce one wait per page.
    for row in _ROWS:
        for text in row["source"]:
            segment = type("Segment", (), {"source_text": text})()
            assert asyncio.run(translation_callback(segment=segment)) == row["target"][row["source"].index(text)]
    assert waits == [value / 1000 for value in (137, 139, 149, 151, 157, 163)]


def test_consolidation_is_the_fixed_terminology_set_and_never_the():
    prompt = json.dumps({"items": [{"source_terms": ["the", "Analytical Engine"]}]})
    data = json.loads(consolidation_callback("prefix\n" + prompt))
    assert {entry["concept_id"] for entry in data["entries"]} == set(_TERMINOLOGY)
    assert "the" not in {term for entry in data["entries"] for term in entry["source_terms"]}
    assert any(entry["concept_id"] == "Analytical Engine" for entry in data["entries"])


def test_correction_oracle_uses_actual_span_and_stable_occurrence_identity():
    first = expected_dirty_set()
    second = expected_dirty_set()
    assert first == second
    assert first["occurrence_id"]
    assert first["dirty_segment_ids"] == ["p05:2"]
    assert expected_dirty_set("other-segment")["occurrence_id"] != first["occurrence_id"]


def test_harness_executes_warm_cache_corruption_probe_and_sibling_json(tmp_path: Path, monkeypatch):
    def fake_run(config, mode, timing, stats):
        stats["extraction"] = 1 if config.base_revision is None else 0
        (config.workspace / "record.json").write_text("immutable", encoding="utf-8")
        return SimpleNamespace(status="completed", report=SimpleNamespace(non_actionable_finding_count=0), candidate_revision_id="sealed-revision")

    monkeypatch.setattr(baseline, "_run_once", fake_run)
    result = baseline.run_benchmark_case("native", tmp_path)
    assert result["cache_oracle"]["warm_cache_reuse"] is True
    assert result["corruption_oracle"]["corruption_observable"] is True
    assert result["corruption_oracle"]["original_hashes_mtimes_unchanged"] is True
    output_json = tmp_path / "native-baseline.json"
    assert output_json.is_file()
    assert json.loads(output_json.read_text(encoding="utf-8"))["mode"] == "native"


def test_benchmark_rejects_non_empty_measured_workspace(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stale-state").write_text("stale", encoding="utf-8")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("benchmark must reject stale state before running")

    monkeypatch.setattr(baseline, "_run_once", unexpected_run)
    with pytest.raises(AssertionError, match="measured workspace must be empty"):
        baseline.run_benchmark_case("native", tmp_path)


def test_benchmark_fails_cache_and_corruption_oracles(tmp_path: Path, monkeypatch):
    def fake_run(config, mode, timing, stats):
        stats["extraction"] = 1 if config.base_revision is None else 1
        (config.workspace / "record.json").write_text("immutable", encoding="utf-8")
        return SimpleNamespace(status="completed", report=SimpleNamespace(non_actionable_finding_count=0), candidate_revision_id="sealed-revision")

    monkeypatch.setattr(baseline, "_run_once", fake_run)
    with pytest.raises(AssertionError, match="cache-reuse oracle failed"):
        baseline.run_benchmark_case("native", tmp_path / "cache-failure")

    def cached_run(config, mode, timing, stats):
        stats["extraction"] = 1 if config.base_revision is None else 0
        (config.workspace / "record.json").write_text("immutable", encoding="utf-8")
        return SimpleNamespace(status="completed", report=SimpleNamespace(non_actionable_finding_count=0), candidate_revision_id="sealed-revision")

    monkeypatch.setattr(baseline, "_run_once", cached_run)
    monkeypatch.setattr(baseline, "_corruption_probe", lambda *_args: {
        "corruption_observable": False,
        "original_hashes_mtimes_unchanged": True,
    })
    with pytest.raises(AssertionError, match="corruption oracle failed"):
        baseline.run_benchmark_case("native", tmp_path / "corruption-failure")


def test_state_measure_excludes_directories_and_symlinks(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "record").write_bytes(b"abc")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "link").symlink_to(tmp_path / "nested" / "record")
    assert state_measure(tmp_path) == {"files": ["nested/record"], "file_count": 1, "bytes": 3}
