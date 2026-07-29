"""Focused ``identity-v1`` identity and exact raw-hash reconciliation tests."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from btran.artifacts import (
    ArtifactError,
    ArtifactStore,
    CacheValidator,
    DependencyGraph,
    RevisionStore,
    concept_membership_semantic_key,
    correction_semantic_key,
    finding_semantic_key,
    occurrence_shard_semantic_key,
    preflight_semantic_key,
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


def test_artifact_store_is_content_addressed_closed_and_quarantines_bad_cache(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    assert store.get(artifact.artifact_id) == artifact
    assert store.closure((artifact.artifact_id,))[0] == (artifact,)

    # Malformed cache copy must fail closed, move only mutable cache data, and
    # leave a durable informational diagnostic.
    (tmp_path / "artifacts" / f"{artifact.artifact_id}.json").write_text("not-json")
    with pytest.raises(Exception):
        store.get(artifact.artifact_id)
    assert not (tmp_path / "artifacts" / f"{artifact.artifact_id}.json").exists()
    assert list((tmp_path / "quarantine").glob("artifact-*.json"))
    assert any(Finding.from_file(path).kind == "cache_artifact_invalid" for path in (tmp_path / "findings").glob("*.json"))


def test_missing_or_malformed_finding_invalidates_its_parent_cache_artifact(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    finding_id = artifact.finding_ids[0]
    (tmp_path / "findings" / f"{finding_id}.json").unlink()
    with pytest.raises(Exception):
        store.get(artifact.artifact_id)
    assert not (tmp_path / "artifacts" / f"{artifact.artifact_id}.json").exists()


def test_corrupt_dependency_is_quarantined_during_parent_closure_validation(tmp_path):
    store = ArtifactStore(tmp_path)
    child = _store_artifact(store, payload={"value": "child"})
    parent = store.put("parent", {"value": "parent"}, dependency_ids=(child.artifact_id,), semantic_key="parent")
    (tmp_path / "artifacts" / f"{child.artifact_id}.json").write_text("not-json")

    with pytest.raises(Exception):
        store.get(parent.artifact_id)

    assert not (tmp_path / "artifacts" / f"{child.artifact_id}.json").exists()
    assert list((tmp_path / "quarantine").glob(f"artifact-{child.artifact_id}-*.json"))
    # Parent closure became invalid too, so no stale parent can be reused.
    assert not (tmp_path / "artifacts" / f"{parent.artifact_id}.json").exists()


def test_cache_validator_recomputes_current_requested_key_before_snapshot_reuse(tmp_path):
    store = ArtifactStore(tmp_path)
    current = dict(algorithm_version="a1", image_library_version="image-lib", configuration={"dpi": 300}, raw_bytes=b"raw", normalized_image_bytes=b"normalized")
    key = preflight_semantic_key(**current)
    artifact = _store_artifact(store, semantic_key=key)
    snapshot = RevisionSnapshot(
        revision_id="revision", selected_artifact_ids=(artifact.artifact_id,),
        selected_cache_attestation_ids=(store.attestation_id_for(artifact.artifact_id, "test", key),),
    )
    validator = CacheValidator(store)
    assert validator.select(snapshot, requested_artifact_id=artifact.artifact_id, kind="test", key_constructor=preflight_semantic_key, current_inputs=current) == artifact

    # Every changed requested input creates a cache miss even while stored hash
    # and closure remain valid.
    variants = (
        {**current, "algorithm_version": "a2"},
        {**current, "image_library_version": "image-lib-2"},
        {**current, "configuration": {"dpi": 301}},
        {**current, "raw_bytes": b"raw-2"},
        {**current, "normalized_image_bytes": b"normalized-2"},
    )
    for changed in variants:
        assert validator.select(snapshot, requested_artifact_id=artifact.artifact_id, kind="test", key_constructor=preflight_semantic_key, current_inputs=changed) is None

    # Reuse requires both matching envelope key and matching index membership.
    store._index_path("test", key).unlink()
    assert validator.select(snapshot, requested_artifact_id=artifact.artifact_id, kind="test", key_constructor=preflight_semantic_key, current_inputs=current) is None


def test_cache_validator_empty_or_invalid_selected_attestation_is_a_miss(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store, semantic_key="key")
    validator = CacheValidator(store)
    key_constructor = lambda *, value: value
    empty = RevisionSnapshot(revision_id="empty", selected_artifact_ids=(artifact.artifact_id,))
    assert validator.select(empty, requested_artifact_id=artifact.artifact_id,
                            kind="test", key_constructor=key_constructor, value="key") is None

    attestation_id = store.attestation_id_for(artifact.artifact_id, "test", "key")
    selected = RevisionSnapshot(
        revision_id="selected", selected_artifact_ids=(artifact.artifact_id,),
        selected_cache_attestation_ids=(attestation_id,),
    )
    store._attestation_path(attestation_id).write_text("not-json")
    assert validator.select(selected, requested_artifact_id=artifact.artifact_id,
                            kind="test", key_constructor=key_constructor, value="key") is None


def test_every_semantic_key_constructor_mutates_declared_inputs_and_rejects_metadata():
    cases = (
        (preflight_semantic_key, dict(algorithm_version="a", image_library_version="lib", configuration={"dpi": 300}, raw_bytes=b"raw", normalized_image_bytes=b"normal"), {"algorithm_version": "b", "image_library_version": "lib-2", "configuration": {"dpi": 301}, "raw_bytes": b"raw-2", "normalized_image_bytes": b"normal-2"}),
        (source_extraction_semantic_key, dict(extraction_schema="schema", prompt_bytes=b"prompt", model_executable_identity="pi", model_id="model", raw_bytes=b"raw", normalization_bytes=b"normal"), {"extraction_schema": "schema-2", "prompt_bytes": b"prompt-2", "model_executable_identity": "pi-2", "model_id": "model-2", "raw_bytes": b"raw-2", "normalization_bytes": b"normal-2"}),
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
    selected = RevisionSnapshot(
        revision_id="selected", selected_artifact_ids=(second.artifact_id,),
        selected_cache_attestation_ids=(store.attestation_id_for(second.artifact_id, "test", "same"),),
    )
    key_constructor = lambda *, value: value
    assert validator.select(selected, requested_artifact_id=second.artifact_id, kind="test", key_constructor=key_constructor, value="same") == second
    assert validator.select(selected, requested_artifact_id=first.artifact_id, kind="test", key_constructor=key_constructor, value="same") is None
    assert validator.select(selected, requested_artifact_id=None, kind="test", key_constructor=key_constructor, value="same") is None
    assert any(Finding.from_file(path).kind == "cache_key_ambiguous" for path in (tmp_path / "findings").glob("*.json"))


def test_sealed_revision_is_self_contained_and_graph_is_selected_revision_only(tmp_path):
    store = ArtifactStore(tmp_path)
    first = _store_artifact(store, payload={"value": 1})
    second_finding = Finding(kind="stage_summary", severity="info", stage="test", message="dependent")
    store.put_finding(second_finding)
    second = store.put("render", {"value": 2}, dependency_ids=(first.artifact_id,), finding_ids=(second_finding.finding_id,), semantic_key="render")
    graph = DependencyGraph(tmp_path)
    edge = graph.edge(stable_subject_id="segment", parent_artifact_id=first.artifact_id, child_artifact_id=second.artifact_id, stage="render", edge_kind="input")
    graph.put(edge)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(second.artifact_id,))
    provenance = {"revision_id": "revision", "render_input": second.artifact_id}
    epub = io.BytesIO()
    with zipfile.ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/btran-provenance.json", canonical_json_bytes(provenance))
    revisions = RevisionStore(tmp_path, store, graph)
    bundle = revisions.seal_bundle(snapshot, provenance, epub.getvalue(), render_input_artifact_id=second.artifact_id, edge_ids=(edge.edge_id,), expected_embedded_provenance=provenance)
    assert (bundle / "artifacts" / f"{first.artifact_id}.json").exists()
    assert (bundle / "artifacts" / f"{second.artifact_id}.json").exists()
    assert (bundle / "findings" / f"{second_finding.finding_id}.json").exists()
    revisions.activate("revision")
    assert revisions.active_snapshot() == snapshot

    # Global graph cache can change/corrupt after sealing. Selected traversal
    # must exclusively read immutable copied graph bytes in sealed bundle.
    (tmp_path / "graph" / "edges" / f"{edge.edge_id}.json").write_text("not-json")
    assert revisions.selected_graph("revision").forward("revision", first.artifact_id) == (edge,)


def test_existing_matching_bundle_revalidates_before_idempotent_return(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = _store_artifact(store)
    snapshot = RevisionSnapshot(revision_id="revision", selected_artifact_ids=(artifact.artifact_id,))
    graph = DependencyGraph(tmp_path)
    revisions = RevisionStore(tmp_path, store, graph)
    provenance = {"revision_id": "revision", "render_input": artifact.artifact_id}
    epub = io.BytesIO()
    with zipfile.ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/btran-provenance.json", canonical_json_bytes(provenance))
    bundle = revisions.seal_bundle(snapshot, provenance, epub.getvalue(), render_input_artifact_id=artifact.artifact_id, expected_embedded_provenance=provenance)
    with (bundle / "book.epub").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ArtifactError, match="EPUB hash"):
        revisions.seal_bundle(snapshot, provenance, epub.getvalue(), render_input_artifact_id=artifact.artifact_id, expected_embedded_provenance=provenance)
