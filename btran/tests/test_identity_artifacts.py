"""Focused ``identity-v1`` identity and exact raw-hash reconciliation tests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import zipfile
from pathlib import Path

import pytest

from btran.artifacts import (
    ArtifactError,
    ArtifactStore,
    LegacyArtifactStore,
    LegacyRevisionStore,
    CacheValidator,
    DependencyGraph,
    RevisionStore,
    concept_membership_semantic_key,
    correction_semantic_key,
    finding_semantic_key,
    occurrence_shard_semantic_key,
    projection_semantic_key,
    reconciliation_semantic_key,
    render_input_semantic_key,
    source_extraction_semantic_key,
    translation_semantic_key,
    validation_semantic_key,
)
from btran.identity import (
    IdentityError,
    PagePlacement,
    book_id_for_page_ids,
    book_record_for_pages,
    canonical_root_segments,
    canonical_source_text,
    concept_for,
    occurrence_for,
    page_id_for_raw_sha256,
    page_record_for_bytes,
    placement_id_for,
    reconcile_book_pages,
    reconcile_raw_hash,
    segment_for,
    structural_anchor_for,
)
from btran.schema import Finding, RevisionSnapshot, canonical_json_bytes, tagged_sha256
from btran.storage import Storage, StorageError


def test_source_text_and_raw_byte_page_identity_are_exact():
    raw = b"not decodable \xff\x00"
    digest = hashlib.sha256(raw).hexdigest()
    assert canonical_source_text(" e\u0301\r\n X\r") == " é\n X\n"
    assert page_id_for_raw_sha256(digest) == tagged_sha256("page-v1", bytes.fromhex(digest))
    assert page_record_for_bytes(raw).raw_file_sha256 == digest

    # Whitespace/case/punctuation are semantic source content, never erased.
    assert canonical_source_text("A  B!") != canonical_source_text("a b")


def test_book_and_placement_keep_logical_identity_separate_from_path():
    first = page_record_for_bytes(b"same")
    second = page_record_for_bytes(b"other")
    book = book_record_for_pages((second, first, first))
    assert book.page_ids == tuple(sorted({first.page_id, second.page_id}))
    assert book.book_id == book_id_for_page_ids((first.page_id, second.page_id))

    first_path = PagePlacement.create(first.page_id, first.raw_file_sha256, "chapter/a.png")
    moved_path = PagePlacement.create(first.page_id, first.raw_file_sha256, "renamed/a.png")
    assert first_path.page_id == moved_path.page_id == first.page_id
    assert first_path.placement_id != moved_path.placement_id
    assert first_path.placement_id == placement_id_for(first.page_id, first.raw_file_sha256, "chapter/a.png")
    assert first_path.evidence == {
        "duplicate_discriminator": "same-raw-bytes",
        "raw_file_sha256": first.raw_file_sha256,
    }
    with pytest.raises(IdentityError):
        PagePlacement.create(first.page_id, first.raw_file_sha256, "../a.png")
    with pytest.raises(IdentityError, match="page_id"):
        PagePlacement.create(second.page_id, first.raw_file_sha256, "chapter/a.png")


@pytest.mark.parametrize("path", ("./a/b.png", "a/./b.png"))
def test_placement_rejects_textually_noncanonical_paths(path):
    page = page_record_for_bytes(b"page")
    with pytest.raises(IdentityError, match="relative path"):
        PagePlacement.create(page.page_id, page.raw_file_sha256, path)


def test_structural_segment_anchor_uses_only_declared_canonical_inputs():
    page = page_record_for_bytes(b"page")
    source = "e\u0301\r\ntext"
    fields = {"bbox": [1, 2, 3, 4], "style": "body"}
    anchor = structural_anchor_for("paragraph", 2, source, fields)
    expected = tagged_sha256(
        "anchor-v1", canonical_json_bytes(["paragraph", 2, "é\ntext", fields])
    )
    assert anchor == expected
    segment = segment_for(page.page_id, "paragraph", 2, source, "fr", fields)
    assert segment.structural_anchor == anchor
    assert segment.source_text == "é\ntext"
    assert segment.segment_id == tagged_sha256(
        "segment-v1", page.page_id.encode("utf-8"), anchor.encode("utf-8")
    )
    assert structural_anchor_for("paragraph", 2, source, {"style": "heading"}) != anchor


def test_invalid_root_orders_produce_diagnostic_placeholder_and_finding():
    page = page_record_for_bytes(b"page")
    result = canonical_root_segments(page.page_id, [
        {"kind": "paragraph", "source_text": "first", "source_lang": "en", "reading_order": 1},
        {"kind": "paragraph", "source_text": "second", "source_lang": "en", "reading_order": 1},
    ])
    assert len(result.segments) == len(result.placements) == 1
    assert result.segments[0].kind == "diagnostic_placeholder"
    assert result.segments[0].source_lang is None
    assert result.findings[0].kind == "invalid_root_sequence"


def test_same_order_anchor_with_different_source_language_is_invalid_root_sequence():
    page = page_record_for_bytes(b"page")
    result = canonical_root_segments(page.page_id, [
        {"kind": "paragraph", "source_text": "same", "source_lang": "en", "reading_order": 1},
        {"kind": "paragraph", "source_text": "same", "source_lang": "fr", "reading_order": 1},
    ])
    assert len(result.segments) == len(result.placements) == 1
    assert result.segments[0].kind == "diagnostic_placeholder"
    assert [finding.kind for finding in result.findings] == ["invalid_root_sequence"]


def test_exact_duplicate_block_keeps_one_segment_and_references_it_from_each_placement():
    page = page_record_for_bytes(b"page")
    duplicate = {"kind": "paragraph", "source_text": "same", "source_lang": "en", "reading_order": 1}
    result = canonical_root_segments(page.page_id, [duplicate, duplicate])
    assert len(result.segments) == 1
    assert len(result.placements) == 2
    assert {placement.segment_id for placement in result.placements} == {result.segments[0].segment_id}
    assert [finding.kind for finding in result.findings] == ["duplicate_segment_identity"]
    with_bbox = canonical_root_segments(page.page_id, [{
        **duplicate, "reading_order": 2, "bbox": [1, 2, 3, 4],
    }])
    assert with_bbox.segments[0].structural_anchor != result.segments[0].structural_anchor


def test_occurrence_uses_python_code_point_offsets_and_exact_surface_slice():
    page = page_record_for_bytes(b"page")
    # Python indexes U+1F4A9 as one code point, unlike UTF-8 byte offsets.
    segment = segment_for(page.page_id, "paragraph", 1, "A💩é", "en")
    occurrence = occurrence_for(segment, 1, 3)
    assert occurrence.surface == "💩é"
    assert occurrence.occurrence_id == tagged_sha256(
        "occurrence-v1", segment.segment_id.encode("utf-8"), b"1", b"3", "💩é".encode("utf-8")
    )
    with pytest.raises(IdentityError, match="exactly slice"):
        occurrence_for(segment, 1, 3, "💩")


def test_concept_identity_sorts_occurrences_and_canonicalizes_source_form():
    page = page_record_for_bytes(b"page")
    segment = segment_for(page.page_id, "paragraph", 1, "é é", "fr")
    one = occurrence_for(segment, 0, 1)
    two = occurrence_for(segment, 2, 3)
    concept = concept_for("fr", "e\u0301", (two.occurrence_id, one.occurrence_id))
    assert concept.canonical_source_form == "é"
    assert concept.occurrence_ids == tuple(sorted((one.occurrence_id, two.occurrence_id)))
    assert concept.concept_id == tagged_sha256(
        "concept-v1", canonical_json_bytes(["fr", "é", list(concept.occurrence_ids)])
    )


def test_exact_raw_hash_reconciliation_has_zero_one_many_results():
    existing = page_record_for_bytes(b"same")
    other = page_record_for_bytes(b"other")
    duplicate_legacy = {"page_id": "legacy-distinct-page", "raw_file_sha256": existing.raw_file_sha256}

    zero = reconcile_raw_hash("a" * 64, (existing, other))
    assert (zero.status, zero.page_id, zero.finding) == ("new", None, None)

    one = reconcile_raw_hash(existing.raw_file_sha256, (existing, other))
    assert (one.status, one.page_id, one.candidate_page_ids) == ("reused", existing.page_id, (existing.page_id,))

    many = reconcile_raw_hash(existing.raw_file_sha256, (existing, duplicate_legacy))
    assert many.status == "ambiguous"
    assert many.page_id is None
    assert many.candidate_page_ids == tuple(sorted((existing.page_id, "legacy-distinct-page")))
    assert many.finding is not None
    assert many.finding.kind == "duplicate_identity_ambiguous"

    all_results = reconcile_book_pages((existing.raw_file_sha256, "a" * 64), (existing, other))
    assert [item.status for item in all_results] == ["reused", "new"]


def _store_artifact(store, *, payload=None, semantic_key="semantic"):
    finding = Finding(kind="stage_summary", severity="info", stage="test", message="done")
    store.put_finding(finding)
    return store.put("test", payload or {"value": 1}, finding_ids=(finding.finding_id,), semantic_key=semantic_key)


def test_v2_store_uses_full_durability_schema_and_exact_relations(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    assert (tmp_path / "state-v2.sqlite3").is_file()
    connection = __import__("sqlite3").connect(tmp_path / "state-v2.sqlite3")
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"records", "findings", "record_dependencies", "record_findings", "edges", "attestations", "semantic_index", "revisions", "active_revision"} <= tables
    assert store.get(artifact.artifact_id) == artifact


def test_v2_sealed_zip_is_deterministic_self_contained_and_active_pointer_is_db(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(artifact.artifact_id,), selected_finding_ids=artifact.finding_ids)
    revisions = RevisionStore(tmp_path)
    bundle = revisions.seal_bundle(snapshot, {}, b"")
    first = bundle.read_bytes()
    assert revisions.seal_bundle(snapshot, {}, b"").read_bytes() == first
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert names[-1] == "manifest.json" and names[:-1] == sorted(names[:-1])
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3 and info.create_version == 20 and info.extract_version == 20
            assert info.external_attr == (0o100444 << 16) and info.compress_type == zipfile.ZIP_STORED
    assert revisions.snapshot("revision") == snapshot
    revisions.activate("revision")
    assert revisions.active_snapshot() == snapshot


def test_v2_exact_key_attestation_and_index_are_inspectable(tmp_path):
    store = ArtifactStore(tmp_path)
    first = _store_artifact(store, payload={"value": 1}, semantic_key="same")
    second = _store_artifact(store, payload={"value": 2}, semantic_key="same")
    assert store.indexed_ids("test", "same") == tuple(sorted((first.artifact_id, second.artifact_id)))
    validator = CacheValidator(store)
    snapshot = RevisionSnapshot(revision_id="selected", selected_artifact_ids=(second.artifact_id,), selected_cache_attestation_ids=(store.attestation_id_for(second.artifact_id, "test", "same"),))
    assert validator.select(snapshot, requested_artifact_id=second.artifact_id, kind="test", key_constructor=lambda *, value: value, value="same") == second
    assert validator.select(snapshot, requested_artifact_id=first.artifact_id, kind="test", key_constructor=lambda *, value: value, value="same") is None


def test_legacy_workspace_is_not_migrated_or_mutated_on_read(tmp_path):
    legacy = tmp_path / "artifacts"; legacy.mkdir(); marker = legacy / "old.json"; marker.write_text("not-json")
    before = (marker.stat().st_size, marker.stat().st_mtime_ns)
    with pytest.raises(Exception): ArtifactStore(tmp_path).get("old")
    assert not (tmp_path / "state-v2.sqlite3").exists()
    assert marker.exists() and (marker.stat().st_size, marker.stat().st_mtime_ns) == before


def test_every_semantic_key_constructor_mutates_declared_inputs_and_rejects_metadata():
    cases = (
        (source_extraction_semantic_key, dict(extraction_schema="schema", prompt_bytes=b"prompt", model_executable_identity="pi", model_id="model", raw_bytes=b"raw"), {"extraction_schema": "schema-2", "prompt_bytes": b"prompt-2", "model_executable_identity": "pi-2", "model_id": "model-2", "raw_bytes": b"raw-2"}),
        (occurrence_shard_semantic_key, dict(effective_source_segment={"id": "segment", "text": "source"}, evidence_rules_version="rules"), {"effective_source_segment": {"id": "segment", "text": "changed"}, "evidence_rules_version": "rules-2"}),
        (concept_membership_semantic_key, dict(concept_id="concept", occurrence_ids=("occurrence",), evidence_shard_ids=("shard",), membership_rules_version="rules"), {"concept_id": "concept-2", "occurrence_ids": ("occurrence-2",), "evidence_shard_ids": ("shard-2",), "membership_rules_version": "rules-2"}),
        (projection_semantic_key, dict(concept_id="concept", occurrence_scope_selector={"kind": "all"}, membership_id="membership", target_form="term", active_terminology_corrections=("correction",), model_executable_identity="pi", model_id="model", prompt_bytes=b"prompt", consolidation_schema="schema", algorithm_version="algorithm"), {"concept_id": "concept-2", "occurrence_scope_selector": {"kind": "subset"}, "membership_id": "membership-2", "target_form": "term-2", "active_terminology_corrections": ("correction-2",), "model_executable_identity": "pi-2", "model_id": "model-2", "prompt_bytes": b"prompt-2", "consolidation_schema": "schema-2", "algorithm_version": "algorithm-2"}),
        (translation_semantic_key, dict(source_artifact_id="source", preceding_source_artifact_id="before", following_source_artifact_id="after", projection_ids=("projection",), model_executable_identity="pi", model_id="model", prompt_bytes=b"prompt", target_lang="en"), {"source_artifact_id": "source-2", "preceding_source_artifact_id": None, "following_source_artifact_id": None, "projection_ids": ("projection-2",), "model_executable_identity": "pi-2", "model_id": "model-2", "prompt_bytes": b"prompt-2", "target_lang": "fr"}),
        (reconciliation_semantic_key, dict(effective_page_ids=("page",), effective_page_text_hashes=("hash",), projection_ids=("projection",)), {"effective_page_ids": ("page-2",), "effective_page_text_hashes": ("hash-2",), "projection_ids": ("projection-2",)}),
        (validation_semantic_key, dict(rules_version="rules", mode="translated", target_lang="en", effective_page_ids=("page",), reconciliation_artifact_id="reconciliation", projection_ids=("projection",)), {"rules_version": "rules-2", "mode": "native", "target_lang": None, "effective_page_ids": ("page-2",), "reconciliation_artifact_id": "reconciliation-2", "projection_ids": ("projection-2",)}),
        (render_input_semantic_key, dict(renderer_version="renderer", template_version="template", mode="translated", language="en", ordered_page_display_metadata=({"title": "one"},), effective_page_ids=("page",), validation_artifact_id="validation", correction_ids=("correction",), finding_ids=("finding",)), {"renderer_version": "renderer-2", "template_version": "template-2", "mode": "native", "language": None, "ordered_page_display_metadata": ({"title": "two"},), "effective_page_ids": ("page-2",), "validation_artifact_id": "validation-2", "correction_ids": ("correction-2",), "finding_ids": ("finding-2",)}),
        (correction_semantic_key, dict(kind="source_text", revision_id="revision", subjects=("segment",), base_hashes={"source": "hash"}, replacement="replacement", explicit_scope={"kind": "segment"}, supersedes_id="old"), {"kind": "target_segment", "revision_id": "revision-2", "subjects": ("segment-2",), "base_hashes": {"source": "hash-2"}, "replacement": "replacement-2", "explicit_scope": {"kind": "occurrence"}, "supersedes_id": None}),
    )
    for constructor, inputs, changes in cases:
        baseline = constructor(**inputs)
        for field, value in changes.items():
            changed = dict(inputs)
            changed[field] = value
            assert constructor(**changed) != baseline, f"{constructor.__name__} omitted {field}"
        # Excluded execution metadata cannot quietly become a key input.
        with pytest.raises(TypeError):
            constructor(**inputs, execution_metadata={"run_id": "different"})


def test_finding_semantic_key_requires_mapping_schema_and_excludes_nonsemantic_metadata():
    finding = Finding(kind="issue", severity="warning", stage="validate", subject_refs=("subject",), evidence={"rule": "r"}, message="first")
    body = finding.to_dict()
    baseline = finding_semantic_key(body)
    for field, value in (("schema_version", "schema-v2"), ("stage", "render"), ("kind", "other"), ("severity", "error"), ("subject_refs", ["subject-2"]), ("evidence", {"rule": "other"})):
        changed = dict(body)
        changed[field] = value
        assert finding_semantic_key(changed) != baseline
    with pytest.raises(ArtifactError, match="schema_version"):
        finding_semantic_key({key: value for key, value in body.items() if key != "schema_version"})
    assert finding_semantic_key({**body, "message": "transport copy", "dependency_ids": ["other"], "execution_metadata": {"run_id": "other"}}) == baseline


def test_snapshot_only_selection_and_same_key_ambiguity_never_use_index_order(tmp_path):
    store = ArtifactStore(tmp_path)
    first = _store_artifact(store, payload={"value": 1}, semantic_key="same")
    second = _store_artifact(store, payload={"value": 2}, semantic_key="same")
    validator = CacheValidator(store)
    selected = RevisionSnapshot(revision_id="selected", selected_artifact_ids=(second.artifact_id,), selected_cache_attestation_ids=(store.attestation_id_for(second.artifact_id, "test", "same"),))
    key_constructor = lambda *, value: value
    assert validator.select(selected, requested_artifact_id=second.artifact_id, kind="test", key_constructor=key_constructor, value="same") == second
    assert validator.select(selected, requested_artifact_id=first.artifact_id, kind="test", key_constructor=key_constructor, value="same") is None
    finding = validator.ambiguity("test", "same")
    assert finding is not None and set(finding.subject_refs) == {first.artifact_id, second.artifact_id}


def test_sealed_revision_is_self_contained_and_graph_is_selected_revision_only(tmp_path):
    store = ArtifactStore(tmp_path)
    first = _store_artifact(store, payload={"value": 1})
    second = store.put("render", {"value": 2}, dependency_ids=(first.artifact_id,), semantic_key="render")
    graph = DependencyGraph(tmp_path)
    edge = graph.edge(stable_subject_id="segment", parent_artifact_id=first.artifact_id, child_artifact_id=second.artifact_id, stage="render", edge_kind="input")
    graph.put(edge)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(second.artifact_id,))
    revisions = RevisionStore(tmp_path, store, graph)
    bundle = revisions.seal_bundle(snapshot, {}, b"", edge_ids=(edge.edge_id,))
    with zipfile.ZipFile(bundle) as archive:
        assert f"records/{first.artifact_id}.json" in archive.namelist()
        assert f"records/{second.artifact_id}.json" in archive.namelist()
        assert f"edges/{edge.edge_id}.json" in archive.namelist()
    assert revisions.selected_graph("revision").forward("revision", first.artifact_id) == (edge,)


def test_standalone_verification_rejects_closure_relationship_mutation(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(artifact.artifact_id,))
    bundle = RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    values = Storage(tmp_path).verify_revision("revision")
    raw = bytearray(bundle.read_bytes())
    # The repository row must detect archive bytes before any mutable index is
    # consulted.  This also covers the immutable SHA comparison boundary.
    connection = sqlite3.connect(tmp_path / "state-v2.sqlite3")
    connection.execute("UPDATE revisions SET zip_sha256=? WHERE revision_id=?", ("0" * 64, "revision"))
    connection.commit(); connection.close()
    with pytest.raises(ArtifactError, match="SHA"):
        RevisionStore(tmp_path).verify_bundle("revision")
    assert values["snapshot.json"]


def test_resealing_revision_id_with_changed_snapshot_does_not_reuse_old_zip(tmp_path):
    store = ArtifactStore(tmp_path)
    first = _store_artifact(store, payload={"value": 1})
    second = _store_artifact(store, payload={"value": 2})
    revisions = RevisionStore(tmp_path)
    revisions.seal_bundle(RevisionSnapshot(revision_id="revision", selected_artifact_ids=(first.artifact_id,)), {}, b"")
    with pytest.raises(ArtifactError, match="conflicting ZIP"):
        revisions.seal_bundle(RevisionSnapshot(revision_id="revision", selected_artifact_ids=(second.artifact_id,)), {}, b"")


def test_existing_matching_revision_revalidates_before_idempotent_return(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(artifact.artifact_id,))
    revisions = RevisionStore(tmp_path)
    bundle = revisions.seal_bundle(snapshot, {}, b"")
    raw = bytearray(bundle.read_bytes())
    raw[-1] ^= 1
    bundle.write_bytes(raw)
    with pytest.raises(ArtifactError):
        revisions.seal_bundle(snapshot, {}, b"")


def _rewrite_zip(path: Path, values: dict[str, bytes], order: tuple[str, ...], *, changed_name: str | None = None, changed_metadata: bool = False) -> None:
    """Rewrite a test archive with FC3 metadata, optionally changing one field."""
    temporary = path.with_suffix(".rewrite")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = 0o100444 << 16
            info.extra = b""
            info.comment = b""
            info.flag_bits = 0
            info.compress_type = zipfile.ZIP_STORED
            if name == changed_name:
                info.date_time = (1980, 1, 1, 0, 0, 2)
            if changed_metadata and name == "snapshot.json":
                info.external_attr ^= 1
            archive.writestr(info, values[name])
    os.replace(temporary, path)


def test_v2_writer_uses_wal_full_checkpoint_and_fsync(tmp_path, monkeypatch):
    fsynced: list[str] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            target = ""
        fsynced.append(target)
        real_fsync(fd)

    monkeypatch.setattr("btran.storage.os.fsync", record_fsync)
    store = Storage(tmp_path)
    finding = Finding(kind="stage_summary", severity="info", stage="test", message="done")
    store.put_finding(finding.finding_id, finding.to_json().encode())
    connection = store._connect()
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
    connection.close()
    assert any(target.endswith("state-v2.sqlite3") for target in fsynced)


def test_revision_zip_is_fsynced_after_zip_finalization(tmp_path, monkeypatch):
    fsynced_archives: list[bool] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            target = ""
        if target.endswith("revision.zip.tmp"):
            fsynced_archives.append(zipfile.is_zipfile(target))
        real_fsync(fd)

    monkeypatch.setattr("btran.storage.os.fsync", record_fsync)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=())
    Storage(tmp_path).seal_revision("revision", snapshot.to_json().encode(), {})
    assert fsynced_archives == [True]


def test_empty_v2_workspaces_produce_equal_zip_bytes(tmp_path):
    snapshot = RevisionSnapshot(revision_id="empty", selected_artifact_ids=())
    first_root, second_root = tmp_path / "one", tmp_path / "two"
    first = Storage(first_root).seal_revision("empty", snapshot.to_json().encode(), {})
    second = Storage(second_root).seal_revision("empty", snapshot.to_json().encode(), {})
    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()


@pytest.mark.parametrize("corruption", ("timestamp", "metadata", "order", "member", "manifest", "snapshot"))
def test_v2_standalone_verification_rejects_archive_corruption(tmp_path, corruption):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(artifact.artifact_id,))
    bundle = RevisionStore(tmp_path).seal_bundle(snapshot, {}, b"")
    with zipfile.ZipFile(bundle) as archive:
        values = {name: archive.read(name) for name in archive.namelist()}
    names = tuple(values)
    if corruption == "timestamp":
        _rewrite_zip(bundle, values, names, changed_name="snapshot.json")
    elif corruption == "metadata":
        _rewrite_zip(bundle, values, names, changed_metadata=True)
    elif corruption == "order":
        _rewrite_zip(bundle, values, ("manifest.json",) + names[:-1], changed_name=None)
    elif corruption == "member":
        values["provenance.json"] = b"{}"
        _rewrite_zip(bundle, values, tuple(sorted(names[:-1] + ("provenance.json",), key=lambda name: name.encode())) + ("manifest.json",))
    elif corruption == "manifest":
        manifest = json.loads(values["manifest.json"])
        manifest["members"]["snapshot.json"] = "0" * 64
        values["manifest.json"] = canonical_json_bytes(manifest)
        _rewrite_zip(bundle, values, names)
    else:
        changed = RevisionSnapshot(revision_id="other", selected_artifact_ids=(artifact.artifact_id,))
        values["snapshot.json"] = changed.to_json().encode()
        _rewrite_zip(bundle, values, names)
    with pytest.raises(StorageError):
        Storage(tmp_path).verify_zip(bundle, revision_id="revision")


def test_v2_standalone_verification_rejects_unrelated_valid_closure_members(tmp_path):
    store = ArtifactStore(tmp_path)
    selected = _store_artifact(store, payload={"value": "selected"})
    selected_snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(selected.artifact_id,))
    bundle = RevisionStore(tmp_path).seal_bundle(selected_snapshot, {}, b"")

    unrelated_finding = Finding(kind="unrelated", severity="info", stage="other", message="not selected")
    store.put_finding(unrelated_finding)
    unrelated = store.put("unrelated", {"value": "unselected"}, finding_ids=(unrelated_finding.finding_id,), semantic_key="unrelated")
    graph = DependencyGraph(tmp_path)
    unrelated_edge = graph.edge(stable_subject_id="unrelated", parent_artifact_id=selected.artifact_id,
                                child_artifact_id=unrelated.artifact_id, stage="other", edge_kind="unrelated")
    graph.put(unrelated_edge)
    unrelated_attestation_id = store.attestation_id_for(unrelated.artifact_id, unrelated.kind, unrelated.semantic_key)

    with zipfile.ZipFile(bundle) as archive:
        values = {name: archive.read(name) for name in archive.namelist()}
    extras = {
        f"records/{unrelated.artifact_id}.json": unrelated.to_json().encode(),
        f"findings/{unrelated_finding.finding_id}.json": unrelated_finding.to_json().encode(),
        f"edges/{unrelated_edge.edge_id}.json": unrelated_edge.to_json().encode(),
        f"attestations/{unrelated_attestation_id}.json": canonical_json_bytes(store.get_semantic_attestation(unrelated_attestation_id)),
    }
    manifest = json.loads(values["manifest.json"])
    for name, data in extras.items():
        values[name] = data
        manifest["members"][name] = hashlib.sha256(data).hexdigest()
    values["manifest.json"] = canonical_json_bytes(manifest)
    order = tuple(sorted((name for name in values if name != "manifest.json"), key=lambda name: name.encode())) + ("manifest.json",)
    _rewrite_zip(bundle, values, order)

    with pytest.raises(StorageError, match="outside selected closure"):
        Storage(tmp_path).verify_zip(bundle, revision_id="revision")


def _tree_state(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


def _legacy_epub(provenance: dict[str, object]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("META-INF/btran-provenance.json", canonical_json_bytes(provenance))
    return output.getvalue()


def test_valid_legacy_revision_verification_does_not_mutate_bytes_or_mtimes(tmp_path):
    legacy = LegacyArtifactStore(tmp_path)
    artifact = legacy.put("test", {"value": 1}, semantic_key="semantic")
    snapshot = RevisionSnapshot(revision_id="legacy", selected_artifact_ids=(artifact.artifact_id,))
    LegacyRevisionStore(tmp_path, legacy).seal_bundle(snapshot, {}, _legacy_epub({}))
    before = _tree_state(tmp_path)
    assert RevisionStore(tmp_path).verify_bundle("legacy") == snapshot
    assert _tree_state(tmp_path) == before


def test_corrupt_legacy_revision_verification_does_not_mutate_bytes_or_mtimes(tmp_path):
    legacy = LegacyArtifactStore(tmp_path)
    artifact = legacy.put("test", {"value": 1}, semantic_key="semantic")
    snapshot = RevisionSnapshot(revision_id="legacy", selected_artifact_ids=(artifact.artifact_id,))
    LegacyRevisionStore(tmp_path, legacy).seal_bundle(snapshot, {}, _legacy_epub({}))
    snapshot_path = tmp_path / "revisions" / "legacy" / "snapshot.json"
    snapshot_path.write_bytes(b"not-json")
    before = _tree_state(tmp_path)
    with pytest.raises(ArtifactError):
        RevisionStore(tmp_path).verify_bundle("legacy")
    assert _tree_state(tmp_path) == before
