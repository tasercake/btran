"""Task 4 raw-hash BookRecord discovery tests."""

import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from btran.artifacts import ArtifactStore, LegacyArtifactStore, LegacyRevisionStore, RevisionStore
from btran.manifest import (
    DISCOVERY_FILENAME,
    SelectedClosure,
    SelectedClosureError,
    discover_book,
    generate_manifest,
    load_selected_closure,
    write_manifest,
)
from btran.schema import EffectivePage, EffectiveSegment, Finding, RevisionSnapshot


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
    # New workspaces retain findings in compact v2 SQLite state.
    from btran.storage import Storage
    assert Storage(workspace).finding_bytes(second.findings[0].finding_id)


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


def _sealed_effective_page(store, *, semantic_key="page-key", page_id="page-1", page_number=None,
                           segment_id="segment-1", effective_segment_id="effective-segment-1",
                           page_kind="EffectiveSourcePage", segment_kind="EffectiveSourceSegment",
                           extra_dependency_ids=()):
    segment = EffectiveSegment(
        effective_segment_id=effective_segment_id, segment_id=segment_id,
        source_lang="en", source_text="One", effective_text="One", render_lang="en",
    )
    metadata = {} if page_number is None else {"page_number": page_number}
    page = EffectivePage(
        effective_page_id=f"effective-{page_id}", page_id=page_id,
        effective_segment_ids=(segment.effective_segment_id,), source_langs=("en",),
        display_metadata=metadata,
    )
    segment_record = store.put(segment_kind, segment.to_dict(), semantic_key=f"{segment_id}-key")
    page_record = store.put(page_kind, page.to_dict(),
                            dependency_ids=(segment_record.artifact_id, *extra_dependency_ids), semantic_key=semantic_key)
    return page_record, segment_record


def test_selected_closure_loads_archive_once_and_preserves_declared_segment_order(tmp_path):
    store = ArtifactStore(tmp_path)
    page_record, segment_record = _sealed_effective_page(store)
    finding = Finding(
        kind="closure_test", stage="test", message="closure finding",
        evidence={"nested": {"value": "before"}},
    )
    store.put_finding(finding)
    revisions = RevisionStore(tmp_path)
    edge = revisions.graph.edge(
        stable_subject_id="closure-edge", parent_artifact_id=page_record.artifact_id,
        child_artifact_id=segment_record.artifact_id, stage="test", edge_kind="child",
    )
    revisions.graph.put(edge)
    snapshot = RevisionSnapshot(
        revision_id="revision-1", selected_artifact_ids=(page_record.artifact_id,),
        selected_finding_ids=(finding.finding_id,),
    )
    revisions.seal_bundle(snapshot, {"run": "one"}, b"", edge_ids=(edge.edge_id,))

    verify_calls = 0
    verify_revision = revisions.storage.verify_revision

    def counted_verify(revision_id):
        nonlocal verify_calls
        verify_calls += 1
        return verify_revision(revision_id)

    revisions.storage.verify_revision = counted_verify
    closure = load_selected_closure(revisions, "revision-1")
    assert isinstance(closure, SelectedClosure)
    assert closure.revision_id == "revision-1"
    assert tuple(closure.records) == tuple(sorted(closure.records))
    assert closure.ordered_effective_pages[0].segments[0].effective_segment_id == "effective-segment-1"
    assert closure.ordered_effective_segments[0].effective_segment_id == "effective-segment-1"
    assert closure.selected_effective_content.pages == closure.ordered_effective_pages
    with pytest.raises(TypeError):
        closure.ordered_effective_pages[0].page.display_metadata["page_number"] = 2
    with pytest.raises(TypeError):
        closure.ordered_effective_pages[0].page.page_id = "changed"
    with pytest.raises(TypeError):
        closure.ordered_effective_pages[0].segments[0].effective_text = "changed"
    with pytest.raises(TypeError):
        closure.snapshot.revision_id = "changed"
    with pytest.raises(TypeError):
        closure.finding(finding.finding_id).message = "changed"
    with pytest.raises(TypeError):
        closure.finding(finding.finding_id).evidence["nested"]["value"] = "changed"
    with pytest.raises(TypeError):
        closure.edge(edge.edge_id).edge_kind = "changed"
    with pytest.raises(AttributeError):
        closure.selected_effective_content.pages = ()
    assert closure.provenance == {}
    assert closure.final_finding_ids == (finding.finding_id,)
    with pytest.raises(TypeError):
        closure.records["new"] = page_record
    with pytest.raises(TypeError):
        closure.provenance["new"] = True
    with pytest.raises(AttributeError):
        closure.revision_id = "changed"
    # Loading validates the archive once; all later views use the in-memory
    # SelectedClosure rather than reopening or rescanning the selected ZIP.
    assert verify_calls == 1


def test_selected_closure_freezes_nested_record_data(tmp_path):
    store = ArtifactStore(tmp_path)
    record = store.put(
        "NestedPayload",
        {"outer": {"items": [{"value": "before"}]}},
        semantic_key="nested-payload",
    )
    snapshot = RevisionSnapshot(
        revision_id="deep-immutable",
        selected_artifact_ids=(record.artifact_id,),
    )
    RevisionStore(tmp_path).seal_bundle(
        snapshot, {"outer": {"items": ["before"]}}, b"",
    )

    closure = load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)
    selected = closure.record(record.artifact_id)

    with pytest.raises(TypeError):
        selected.payload["outer"]["items"][0]["value"] = "after"
    with pytest.raises(TypeError):
        selected.payload["outer"]["items"].append({"value": "after"})
    with pytest.raises(TypeError):
        selected.payload = {}
    with pytest.raises(TypeError):
        selected.kind = "mutated"
    assert selected.payload["outer"]["items"][0]["value"] == "before"


def test_selected_closure_accepts_translated_closure_with_source_and_target_pages(tmp_path):
    store = ArtifactStore(tmp_path)
    source_page, _ = _sealed_effective_page(store, semantic_key="source-page")
    target_page, _ = _sealed_effective_page(
        store, semantic_key="target-page", page_kind="EffectiveTargetPage",
        segment_kind="EffectiveTargetSegment",
    )
    snapshot = RevisionSnapshot(
        revision_id="translated-with-source-cache",
        selected_artifact_ids=(source_page.artifact_id, target_page.artifact_id),
    )
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")

    closure = load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)

    assert len(closure.ordered_effective_pages) == 1
    assert closure.ordered_effective_pages[0].page.page_id == "page-1"
    assert closure.ordered_effective_pages[0].segments[0].effective_segment_id == "effective-segment-1"
    assert closure.ordered_effective_pages[0].page.__class__ is EffectivePage
    assert closure.selected_effective_content.pages == closure.ordered_effective_pages


def test_selected_closure_rejects_duplicate_stable_identity(tmp_path):
    store = ArtifactStore(tmp_path)
    first, _ = _sealed_effective_page(store, semantic_key="first")
    # A second page with the same stable page ID has a different canonical
    # artifact identity and is therefore valid storage but invalid selection.
    duplicate_payload = EffectivePage(
        effective_page_id="effective-page-2", page_id="page-1",
        effective_segment_ids=(), source_langs=("en",),
    )
    duplicate = store.put("EffectiveSourcePage", duplicate_payload.to_dict(), semantic_key="second")
    snapshot = RevisionSnapshot(
        revision_id="revision-2", selected_artifact_ids=(first.artifact_id, duplicate.artifact_id),
    )
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    with pytest.raises(SelectedClosureError, match="stable identity"):
        load_selected_closure(RevisionStore(tmp_path), "revision-2")


@pytest.mark.parametrize("kind", ["RawSourceExtraction", "DiagnosticSourceFallback"])
def test_selected_closure_rejects_duplicate_source_page_cache_leaf_identity(tmp_path, kind):
    store = ArtifactStore(tmp_path)
    payload = {"page_id": "duplicate-source-page"}
    first = store.put(kind, payload, semantic_key="first")
    duplicate = store.put(kind, {**payload, "variant": 2}, semantic_key="second")
    snapshot = RevisionSnapshot(
        revision_id=f"duplicate-{kind}",
        selected_artifact_ids=tuple(sorted((first.artifact_id, duplicate.artifact_id))),
    )
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    with pytest.raises(SelectedClosureError, match="stable identity"):
        load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)


def test_selected_closure_includes_diagnostic_translation_cache_leaf(tmp_path):
    store = ArtifactStore(tmp_path)
    fallback = store.put(
        "DiagnosticTranslationFallback",
        {"translation_artifact_id": "translation-1", "segment_id": "segment-1"},
        semantic_key="fallback",
    )
    snapshot = RevisionSnapshot(
        revision_id="diagnostic-translation-leaf",
        selected_artifact_ids=(fallback.artifact_id,),
    )
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")

    closure = load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)

    assert closure.translation_segment_cache_leaves == (fallback,)
    assert closure.translation_segment_cache_leaf_map == {"segment-1": fallback}


@pytest.mark.parametrize("kind", ["TranslationArtifact", "DiagnosticTranslationFallback"])
def test_selected_closure_rejects_duplicate_translation_cache_leaf_identity(tmp_path, kind):
    store = ArtifactStore(tmp_path)
    payload = {"translation_artifact_id": "duplicate-translation", "segment_id": "segment-1"}
    first = store.put(kind, payload, semantic_key="first")
    duplicate = store.put(kind, {**payload, "variant": 2}, semantic_key="second")
    snapshot = RevisionSnapshot(
        revision_id=f"duplicate-{kind}",
        selected_artifact_ids=tuple(sorted((first.artifact_id, duplicate.artifact_id))),
    )
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    with pytest.raises(SelectedClosureError, match="stable identity"):
        load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)


@pytest.mark.parametrize("first_kind,second_kind", [
    ("RawSourceExtraction", "DiagnosticSourceFallback"),
    ("RawSourceExtraction", "EffectiveSourcePage"),
    ("RawSourceExtraction", "DiagnosticEffectiveSourcePage"),
    ("DiagnosticSourceFallback", "EffectiveSourcePage"),
    ("DiagnosticSourceFallback", "DiagnosticEffectiveSourcePage"),
    ("EffectiveSourcePage", "DiagnosticEffectiveSourcePage"),
])
def test_selected_closure_rejects_cross_kind_source_page_cache_collision(
    tmp_path, first_kind, second_kind,
):
    store = ArtifactStore(tmp_path)
    first = store.put(first_kind, {"page_id": "shared-page"}, semantic_key="first")
    second = store.put(second_kind, {"page_id": "shared-page"}, semantic_key="second")
    snapshot = RevisionSnapshot(
        revision_id="cross-kind-source-page",
        selected_artifact_ids=tuple(sorted((first.artifact_id, second.artifact_id))),
    )
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")

    with pytest.raises(SelectedClosureError, match="source page cache identity"):
        load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)


@pytest.mark.parametrize("first_kind,second_kind", [
    ("TranslationArtifact", "DiagnosticTranslationFallback"),
    ("TranslationArtifact", "EffectiveTargetSegment"),
    ("TranslationArtifact", "DiagnosticEffectiveTargetSegment"),
    ("DiagnosticTranslationFallback", "EffectiveTargetSegment"),
    ("DiagnosticTranslationFallback", "DiagnosticEffectiveTargetSegment"),
    ("EffectiveTargetSegment", "DiagnosticEffectiveTargetSegment"),
])
def test_selected_closure_rejects_cross_kind_translation_segment_cache_collision(
    tmp_path, first_kind, second_kind,
):
    store = ArtifactStore(tmp_path)
    first = store.put(first_kind, {"segment_id": "shared-segment"}, semantic_key="first")
    second = store.put(second_kind, {"segment_id": "shared-segment"}, semantic_key="second")
    snapshot = RevisionSnapshot(
        revision_id="cross-kind-translation-segment",
        selected_artifact_ids=tuple(sorted((first.artifact_id, second.artifact_id))),
    )
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")

    with pytest.raises(SelectedClosureError, match="translation segment cache identity"):
        load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)


def test_selected_closure_rejects_missing_declared_page_child(tmp_path):
    store = ArtifactStore(tmp_path)
    page = EffectivePage(
        effective_page_id="effective-page-missing", page_id="page-missing",
        effective_segment_ids=("missing-segment",), source_langs=("en",),
    )
    page_record = store.put("EffectiveSourcePage", page.to_dict(), semantic_key="missing-page")
    snapshot = RevisionSnapshot(revision_id="missing-child", selected_artifact_ids=(page_record.artifact_id,))
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    with pytest.raises(SelectedClosureError, match="missing child"):
        load_selected_closure(RevisionStore(tmp_path), "missing-child")


def test_selected_closure_rejects_extra_page_dependency(tmp_path):
    store = ArtifactStore(tmp_path)
    first, first_segment = _sealed_effective_page(store, page_id="page-extra", segment_id="segment-extra",
                                                   effective_segment_id="effective-extra")
    extra_segment = EffectiveSegment(
        effective_segment_id="effective-unlisted", segment_id="segment-unlisted",
        source_lang="en", source_text="Two", effective_text="Two", render_lang="en",
    )
    extra = store.put("EffectiveSourceSegment", extra_segment.to_dict(), semantic_key="unlisted")
    page = EffectivePage(
        effective_page_id="effective-page-extra", page_id="page-extra",
        effective_segment_ids=("effective-extra",), source_langs=("en",),
    )
    malformed = store.put("EffectiveSourcePage", page.to_dict(),
                          dependency_ids=(first_segment.artifact_id, extra.artifact_id), semantic_key="extra-page")
    # The malformed page is the only root; its dependencies are the complete
    # selected closure, including the undeclared child.
    snapshot = RevisionSnapshot(revision_id="extra-child", selected_artifact_ids=(malformed.artifact_id,))
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    with pytest.raises(SelectedClosureError, match="child relationship"):
        load_selected_closure(RevisionStore(tmp_path), "extra-child")


def test_selected_closure_uses_declared_page_order_not_artifact_or_page_id_order(tmp_path):
    store = ArtifactStore(tmp_path)
    first, _ = _sealed_effective_page(store, page_id="page-z", page_number=2, segment_id="segment-z",
                                      effective_segment_id="effective-z")
    second, _ = _sealed_effective_page(store, page_id="page-a", page_number=1, segment_id="segment-a",
                                       effective_segment_id="effective-a")
    snapshot = RevisionSnapshot(revision_id="declared-order", selected_artifact_ids=tuple(sorted((first.artifact_id, second.artifact_id))))
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    closure = load_selected_closure(RevisionStore(tmp_path), "declared-order")
    assert closure.selected_effective_content.ordered_page_ids == ("page-a", "page-z")


@pytest.mark.parametrize("kind,field", [("CorrectionRecord", "correction_id"), ("TermOccurrence", "occurrence_id")])
def test_selected_closure_rejects_duplicate_non_page_stable_identity(tmp_path, kind, field):
    store = ArtifactStore(tmp_path)
    first = store.put(kind, {field: "duplicate", "variant": 1}, semantic_key=f"{kind}-one",
                      dependency_ids=())
    second = store.put(kind, {field: "duplicate", "variant": 2}, semantic_key=f"{kind}-two",
                       dependency_ids=())
    snapshot = RevisionSnapshot(revision_id=f"duplicate-{kind}", selected_artifact_ids=(first.artifact_id, second.artifact_id))
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    with pytest.raises(SelectedClosureError, match="stable identity"):
        load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)


def test_selected_closure_accessors_do_not_reparse_records(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    page_record, _ = _sealed_effective_page(store)
    snapshot = RevisionSnapshot(revision_id="no-reparse", selected_artifact_ids=(page_record.artifact_id,))
    RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    closure = load_selected_closure(RevisionStore(tmp_path), "no-reparse")

    def fail(*args, **kwargs):
        raise AssertionError("selected closure accessor reparsed a record")

    monkeypatch.setattr(EffectivePage, "from_dict", classmethod(fail))
    monkeypatch.setattr(EffectiveSegment, "from_dict", classmethod(fail))
    assert closure.ordered_effective_pages[0].page.page_id == "page-1"
    assert closure.selected_effective_content.ordered_page_ids == ("page-1",)


def test_legacy_discovery_read_does_not_create_or_mutate_state(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    page = input_dir / "page.png"
    page.write_bytes(b"legacy discovery bytes")
    discovered = discover_book(input_dir)
    workspace = tmp_path / "legacy-workspace"
    workspace.mkdir()
    # This is the old loose discovery snapshot without a v2 marker.  It is
    # readable migration input, not permission to publish a new snapshot.
    (workspace / DISCOVERY_FILENAME).write_text(json.dumps({
        "schema_version": "book-discovery-v1",
        "book": discovered.book.to_dict(),
        "known_pages": [item.page.to_dict() for item in discovered.pages],
        "placements": [{
            "page_id": item.page.page_id,
            "raw_file_sha256": item.page.raw_file_sha256,
            "relative_path": item.placement.relative_path,
            "placement_id": item.placement.placement_id,
            "reconciliation": item.reconciliation,
        } for item in discovered.pages],
        "finding_ids": [],
    }))

    def snapshot(root):
        return {
            path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()
        }

    before_workspace = snapshot(workspace)
    before_input = snapshot(input_dir)
    result = discover_book(input_dir, workspace)
    assert result.succeeded
    assert snapshot(workspace) == before_workspace
    assert snapshot(input_dir) == before_input
    assert not (workspace / "state-v2.sqlite3").exists()


def test_corrupt_legacy_selected_closure_fails_closed_without_mutating_bytes_or_mtime(tmp_path):
    store = LegacyArtifactStore(tmp_path)
    page_record, _ = _sealed_effective_page(store)
    snapshot = RevisionSnapshot(revision_id="legacy-corrupt", selected_artifact_ids=(page_record.artifact_id,))
    epub = io.BytesIO()
    with zipfile.ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/btran-provenance.json", json.dumps({}, separators=(",", ":")))
    LegacyRevisionStore(tmp_path, store).seal_bundle(snapshot, {}, epub.getvalue())

    artifact = tmp_path / "revisions" / snapshot.revision_id / "artifacts" / f"{page_record.artifact_id}.json"
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")

    def file_snapshot(root):
        return {
            path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()
        }

    before = file_snapshot(tmp_path)
    with pytest.raises(SelectedClosureError):
        load_selected_closure(RevisionStore(tmp_path), snapshot.revision_id)
    assert file_snapshot(tmp_path) == before
    assert not (tmp_path / "state-v2.sqlite3").exists()


def test_legacy_selected_closure_read_does_not_create_or_mutate_state(tmp_path, monkeypatch):
    store = LegacyArtifactStore(tmp_path)
    page_record, _ = _sealed_effective_page(store)
    snapshot = RevisionSnapshot(revision_id="legacy-1", selected_artifact_ids=(page_record.artifact_id,))
    epub = io.BytesIO()
    with zipfile.ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/btran-provenance.json", json.dumps({"legacy": True}, separators=(",", ":")))
    LegacyRevisionStore(tmp_path, store).seal_bundle(snapshot, {"legacy": True}, epub.getvalue())
    paths = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in tmp_path.rglob("*") if path.is_file()}

    def fail_verify(*args, **kwargs):
        raise AssertionError("legacy loader performed a second verification traversal")

    monkeypatch.setattr(LegacyRevisionStore, "verify_bundle", fail_verify)
    closure = load_selected_closure(RevisionStore(tmp_path), "legacy-1")
    assert closure.snapshot.revision_id == "legacy-1"
    assert closure.ordered_effective_pages
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in tmp_path.rglob("*") if path.is_file()}
    assert after == paths
    assert not (tmp_path / "state-v2.sqlite3").exists()
