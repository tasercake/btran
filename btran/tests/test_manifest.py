"""Task 4 raw-hash BookRecord discovery tests."""

import json
import os
from pathlib import Path

import pytest

from btran.manifest import DISCOVERY_FILENAME, discover_book, generate_manifest, write_manifest


def test_discovery_accepts_supported_undecodable_file(tmp_path):
    (tmp_path / "page.png").write_bytes(b"definitely not a png")
    result = discover_book(tmp_path)
    assert result.succeeded
    assert result.book is not None
    assert len(result.pages) == 1
    assert result.pages[0].page.raw_file_sha256


def test_discovery_uses_raw_hash_not_filename_or_order(tmp_path):
    workspace = tmp_path / "state"
    (tmp_path / "z.png").write_bytes(b"same bytes")
    first = discover_book(tmp_path, workspace)
    first_page_id = first.pages[0].page.page_id
    (tmp_path / "z.png").rename(tmp_path / "a.png")
    second = discover_book(tmp_path, workspace)
    assert second.pages[0].page.page_id == first_page_id
    assert second.pages[0].reconciliation == "reused"


def test_discovery_keeps_duplicate_raw_files_as_distinct_ordered_placements(tmp_path):
    (tmp_path / "a.png").write_bytes(b"same bytes")
    (tmp_path / "b.png").write_bytes(b"same bytes")

    result = discover_book(tmp_path)

    assert result.book is not None
    assert len(result.pages) == 2
    assert len({item.page.page_id for item in result.pages}) == 1
    assert [item.placement.relative_path for item in result.pages] == ["a.png", "b.png"]
    assert len({item.placement.placement_id for item in result.pages}) == 2
    assert len(result.book.page_ids) == 1


def test_discovery_reuses_raw_id_across_rename_reorder_and_timestamp_change(tmp_path):
    workspace = tmp_path / "state"
    first_path, second_path = tmp_path / "a.png", tmp_path / "b.png"
    first_path.write_bytes(b"first page")
    second_path.write_bytes(b"second page")
    first = discover_book(tmp_path, workspace)
    first_ids = {item.page.page_id for item in first.pages}

    first_path.rename(tmp_path / "z.png")
    second_path.rename(tmp_path / "a.png")  # Reverse deterministic placement order.
    os.utime(tmp_path / "z.png", (1_700_000_000, 1_700_000_000))
    os.utime(tmp_path / "a.png", (1_800_000_000, 1_800_000_000))
    second = discover_book(tmp_path, workspace)

    assert {item.page.page_id for item in second.pages} == first_ids
    assert [item.reconciliation for item in second.pages] == ["reused", "reused"]


def test_missing_page_is_persisted_finding_and_history_remains(tmp_path):
    workspace = tmp_path / "state"
    page = tmp_path / "page.jpg"
    page.write_bytes(b"bytes")
    first = discover_book(tmp_path, workspace)
    page.unlink()
    second = discover_book(tmp_path, workspace)
    assert [finding.kind for finding in second.findings] == ["page_missing"]
    snapshot = json.loads((workspace / DISCOVERY_FILENAME).read_text())
    assert snapshot["known_pages"][0]["page_id"] == first.pages[0].page.page_id
    assert (workspace / "findings" / f"{second.findings[0].finding_id}.json").exists()


def test_missing_input_becomes_nonleaking_invocation_failure(tmp_path):
    missing = tmp_path / "missing"
    result = discover_book(missing)
    assert result.book is None
    assert result.invocation_failure is not None
    assert result.invocation_failure.code == "input_access"
    assert result.invocation_failure.exception_type in {"FileNotFoundError", "NotADirectoryError"}


def test_discovery_stat_error_is_typed_input_access_failure(tmp_path, monkeypatch):
    page = tmp_path / "page.png"
    page.write_bytes(b"raw page bytes")
    original_stat = Path.stat

    def inaccessible_stat(path, *args, **kwargs):
        if path == page:
            raise PermissionError("page stat denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", inaccessible_stat)
    result = discover_book(tmp_path)

    assert result.invocation_failure is not None
    assert result.invocation_failure.code == "input_access"
    assert result.invocation_failure.path == str(page)
    assert result.invocation_failure.exception_type == "PermissionError"


def test_legacy_manifest_stat_error_is_not_silently_skipped(tmp_path, monkeypatch):
    page = tmp_path / "page.png"
    page.write_bytes(b"raw page bytes")
    original_stat = Path.stat

    def inaccessible_stat(path, *args, **kwargs):
        if path == page:
            raise PermissionError("page stat denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", inaccessible_stat)

    with pytest.raises(PermissionError, match="page stat denied"):
        generate_manifest(tmp_path)


def test_legacy_manifest_serialization_remains_narrow_migration_format(tmp_path):
    (tmp_path / "page.png").write_bytes(b"bytes")
    manifest = generate_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert set(json.loads(path.read_text())) == {"input_dir", "pages", "total_pages"}
