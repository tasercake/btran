"""Task 5 correction record/set storage and selection substrate."""

from __future__ import annotations

from dataclasses import replace
import io
import json
import sys
import zipfile

import pytest

from btran.artifacts import ArtifactStore, RevisionStore
from btran.cli import main as cli_main
from btran.corrections import (
    CorrectionError,
    CorrectionSet,
    CorrectionStore,
    base_hash_for_artifact,
    correction_event_for,
    correction_event_id_for,
    correction_record_for,
    correction_transition,
    parse_correction_json,
    resolve_correction_set,
    resolve_selected_overlays,
)
from btran.schema import ArtifactEnvelope, CorrectionEvent, CorrectionImpact, RevisionSnapshot, canonical_json_bytes


def _artifact(identifier: str, payload: dict, *, kind: str = "test") -> ArtifactEnvelope:
    return ArtifactEnvelope(artifact_id=identifier, kind=kind, payload=payload, semantic_key=identifier)


def _base(envelope: ArtifactEnvelope) -> dict[str, str]:
    return {"artifact_id": envelope.artifact_id, "sha256": base_hash_for_artifact(envelope)}


def _source(revision: str, artifact: ArtifactEnvelope, segment: str = "segment") -> dict:
    return {"kind": "source_text", "applies_to_revision_id": revision, "scope": {"segment_id": segment}, "base": _base(artifact), "replacement": "fixed"}


def _successor_event_set(revision: str, active_ids: tuple[str, ...], correction_id: str, event_kind: str, *prior_event_ids: str) -> tuple[CorrectionEvent, CorrectionSet]:
    """Construct non-circular pair: hash prior IDs, then add derived own ID."""
    event_id = correction_event_id_for(correction_id, event_kind, revision, active_ids, prior_event_ids)
    correction_set = CorrectionSet.create(revision, active_ids, (*prior_event_ids, event_id))
    return correction_event_for(correction_id, event_kind, correction_set), correction_set


def test_exact_utf8_canonical_grammar_and_all_payload_shapes():
    source = {"applies_to_revision_id": "revision", "base": {"artifact_id": "source", "sha256": "a" * 64}, "kind": "source_text", "replacement": "x", "scope": {"segment_id": "segment"}}
    assert parse_correction_json(canonical_json_bytes(source)) == source
    with pytest.raises(CorrectionError, match="canonical"):
        parse_correction_json(b'{"kind":"source_text", "applies_to_revision_id":"revision","scope":{"segment_id":"segment"},"base":{"artifact_id":"source","sha256":"' + b"a" * 64 + b'"},"replacement":"x"}')
    with pytest.raises(CorrectionError, match="fields mismatch"):
        correction_record_for({**source, "unknown": True})

    occurrence = {"kind": "target_occurrence", "applies_to_revision_id": "revision", "scope": {"occurrence_id": "occ", "segment_id": "segment", "mapping_id": "map", "start": 0, "end": 1, "expected_target_text": "old"}, "base": {"artifact_id": "translation", "sha256": "b" * 64}, "replacement": "new"}
    segment = {"kind": "target_segment", "applies_to_revision_id": "revision", "scope": {"segment_id": "segment", "expected_target_text": "old"}, "base": {"artifact_id": "translation", "sha256": "b" * 64}, "replacement": "new"}
    all_terms = {"kind": "terminology", "applies_to_revision_id": "revision", "scope": {"concept_id": "concept", "selector": {"kind": "all_concept_occurrences"}}, "base": {"projection": {"artifact_id": "projection", "sha256": "c" * 64}, "membership": {"artifact_id": "membership", "sha256": "d" * 64}}, "replacement": "term"}
    subset = {**all_terms, "scope": {"concept_id": "concept", "selector": {"kind": "occurrence_ids", "ids": ["one", "two"]}}}
    for payload in (occurrence, segment, all_terms, subset):
        assert correction_record_for(payload).correction_id
    with pytest.raises(CorrectionError, match="sorted"):
        correction_record_for({**subset, "scope": {"concept_id": "concept", "selector": {"kind": "occurrence_ids", "ids": ["two", "one"]}}})


def test_records_events_and_sets_are_immutable_canonical_fsynced_storage(tmp_path):
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    record = correction_record_for(_source("revision", source))
    event, correction_set = _successor_event_set("revision", (record.correction_id,), record.correction_id, "apply")
    # Canonical derivation is non-circular: event binds successor state and
    # exact other event IDs, then set ID binds complete event closure.
    assert event.event_id == correction_event_id_for(record.correction_id, "apply", "revision", (record.correction_id,), ())
    assert event.event_id != correction_event_id_for(record.correction_id, "apply", "other-revision", (record.correction_id,))
    store = CorrectionStore(tmp_path)
    assert store.put_record(record) == record
    assert store.get_record(record.correction_id) == record
    assert store.put_set(correction_set) == correction_set
    assert store.put_event(event) == event
    assert store.get_event(event.event_id) == event
    event_path = tmp_path / "corrections" / "events" / f"{event.event_id}.json"
    other_set = CorrectionSet.create("revision")
    store.put_set(other_set)
    event_path.write_text(replace(event, correction_set_id=other_set.set_id).to_json(), encoding="utf-8")
    with pytest.raises(CorrectionError, match="linkage"):
        store.get_event(event.event_id)
    assert store.get_set(correction_set.set_id) == correction_set
    path = tmp_path / "corrections" / "records" / f"{record.correction_id}.json"
    assert path.read_bytes() == record.to_json().encode("utf-8")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CorrectionError):
        store.get_record(record.correction_id)


def test_execution_impact_is_stored_separately_from_nonexecuting_projection(tmp_path):
    entry = {"stage": "translation", "subject_id": "segment", "base_artifact_id": "artifact"}
    correction = CorrectionImpact(
        base_revision_id="revision", projection_plan_id="plan", projected_universe=(entry,), affected=(entry,),
    )
    execution = CorrectionImpact(
        phase="execution", base_revision_id="revision", projection_plan_id="plan",
        projected_universe=(entry,), affected=(entry,), regenerated=(entry,),
    )

    store = CorrectionStore(tmp_path)
    store.put_impact(correction)
    store.put_impact(execution)
    assert store.get_impact("plan") == correction
    assert store.get_impact("plan", phase="execution") == execution


def test_matching_base_selects_deterministic_inputs_and_no_materialization():
    source = _artifact("source", {"segment_id": "segment"})
    translation = _artifact("translation", {"segment_id": "segment", "translated_text": "old", "mappings": [{"mapping_id": "map", "occurrence_id": "occ", "segment_id": "segment", "start": 0, "end": 3, "target_text": "old"}]}, kind="TranslationArtifact")
    projection = _artifact("projection", {"concept_id": "concept"})
    membership = _artifact("membership", {"concept_id": "concept", "occurrence_ids": ["occ"]})
    records = [
        correction_record_for(_source("revision", source)),
        correction_record_for({"kind": "target_occurrence", "applies_to_revision_id": "revision", "scope": {"occurrence_id": "occ", "segment_id": "segment", "mapping_id": "map", "start": 0, "end": 3, "expected_target_text": "old"}, "base": _base(translation), "replacement": "new"}),
        correction_record_for({"kind": "terminology", "applies_to_revision_id": "revision", "scope": {"concept_id": "concept", "selector": {"kind": "occurrence_ids", "ids": ["occ"]}}, "base": {"projection": _base(projection), "membership": _base(membership)}, "replacement": "term"}),
    ]
    correction_set = CorrectionSet.create("revision", [item.correction_id for item in records])
    resolution = resolve_correction_set(correction_set, list(reversed(records)), base_revision_id="revision", selected_artifacts={item.artifact_id: item for item in (source, translation, projection, membership)})
    assert [item.subject_id for item in resolution.source_inputs] == ["segment"]
    assert [item.subject_id for item in resolution.target_inputs] == ["occ"]
    assert [item.subject_id for item in resolution.terminology_inputs] == ["concept"]
    assert "correction_protected" in {finding.kind for finding in resolution.findings}
    assert resolution.applicable_correction_ids == tuple(sorted(item.correction_id for item in records))


def test_orphan_supersession_closure_is_inapplicable_and_never_selected():
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    orphan_parent = correction_record_for(_source("revision", source), supersedes_id="missing-predecessor")
    orphan_successor = correction_record_for(_source("revision", source), supersedes_id=orphan_parent.correction_id)
    correction_set = CorrectionSet.create("revision", [orphan_parent.correction_id, orphan_successor.correction_id])

    result = resolve_correction_set(
        correction_set,
        [orphan_parent, orphan_successor],
        base_revision_id="revision",
        selected_artifacts={source.artifact_id: source},
    )

    by_correction = {finding.evidence["correction_id"]: finding.evidence["reason"] for finding in result.findings if finding.kind == "correction_inapplicable"}
    assert by_correction == {
        orphan_parent.correction_id: "supersedes_predecessor_missing",
        orphan_successor.correction_id: "supersedes_predecessor_closure_incomplete",
    }
    assert not result.applicable_correction_ids


def test_historical_supersession_round_trips_through_record_event_set_ancestry(tmp_path):
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    predecessor = correction_record_for(_source("revision", source))
    successor_payload = _source("revision", source)
    successor_payload["replacement"] = "newer"
    successor = correction_record_for(successor_payload, supersedes_id=predecessor.correction_id)
    predecessor_event, predecessor_set = _successor_event_set(
        "revision", (predecessor.correction_id,), predecessor.correction_id, "apply",
    )
    successor_event, successor_set = _successor_event_set(
        "revision", (successor.correction_id,), successor.correction_id, "supersede", predecessor_event.event_id,
    )

    store = CorrectionStore(tmp_path)
    for record in (predecessor, successor):
        store.put_record(record)
    for correction_set, event in ((predecessor_set, predecessor_event), (successor_set, successor_event)):
        store.put_set(correction_set)
        store.put_event(event)

    bundle = tmp_path / "sealed" / "revision"
    (bundle / "artifacts").mkdir(parents=True)
    (bundle / "bundle-manifest.json").write_bytes(canonical_json_bytes({"artifact_ids": [source.artifact_id]}))
    (bundle / "artifacts" / f"{source.artifact_id}.json").write_text(source.to_json(), encoding="utf-8")

    class Revisions:
        def snapshot(self, revision_id):
            assert revision_id == "revision"

        def _revision_path(self, revision_id):
            assert revision_id == "revision"
            return bundle

    result = resolve_selected_overlays(store, Revisions(), base_revision_id="revision", correction_set_id=successor_set.set_id)
    assert result.applicable_correction_ids == (successor.correction_id,)
    assert result.source_inputs[0].replacement == "newer"


def test_historical_supersession_rejects_missing_event_or_scope_cycle():
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    predecessor = correction_record_for(_source("revision", source))
    successor_payload = _source("revision", source)
    successor_payload["replacement"] = "newer"
    successor = correction_record_for(successor_payload, supersedes_id=predecessor.correction_id)
    correction_set = CorrectionSet.create("revision", (successor.correction_id,))
    # Store-backed resolution supplies event ancestry; direct resolver receives
    # explicit evidence too, so an orphaned immutable record cannot activate it.
    missing_event = resolve_correction_set(
        correction_set, (predecessor, successor), base_revision_id="revision",
        selected_artifacts={source.artifact_id: source}, event_activated_correction_ids=frozenset({successor.correction_id}),
    )
    reasons = {finding.evidence["correction_id"]: finding.evidence["reason"] for finding in missing_event.findings}
    assert reasons[predecessor.correction_id] == "record_event_ancestry_missing"
    assert reasons[successor.correction_id] == "supersedes_predecessor_inapplicable"

    # Cycles are rejected before content applicability.  Mutating an in-memory
    # fixture models a corrupt retained record; a persisted reader rejects its
    # content hash independently.
    predecessor.supersedes_id = successor.correction_id
    cycle = resolve_correction_set(correction_set, (predecessor, successor), base_revision_id="revision", selected_artifacts={source.artifact_id: source})
    assert {finding.evidence["reason"] for finding in cycle.findings} == {"supersedes_predecessor_cycle"}


def test_historical_supersession_rejects_scope_mismatch_and_branch_conflict():
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    translation = _artifact("translation", {"segment_id": "segment", "translated_text": "old"})
    predecessor = correction_record_for(_source("revision", source))
    wrong_kind = correction_record_for(
        {"kind": "target_segment", "applies_to_revision_id": "revision", "scope": {"segment_id": "segment", "expected_target_text": "old"}, "base": _base(translation), "replacement": "new"},
        supersedes_id=predecessor.correction_id,
    )
    mismatch = resolve_correction_set(
        CorrectionSet.create("revision", (wrong_kind.correction_id,)), (predecessor, wrong_kind),
        base_revision_id="revision", selected_artifacts={"source": source, "translation": translation},
    )
    assert {finding.evidence["reason"] for finding in mismatch.findings} == {"supersedes_predecessor_scope_mismatch"}

    first_payload = _source("revision", source)
    first_payload["replacement"] = "one"
    second_payload = _source("revision", source)
    second_payload["replacement"] = "two"
    first = correction_record_for(first_payload, supersedes_id=predecessor.correction_id)
    second = correction_record_for(second_payload, supersedes_id=predecessor.correction_id)
    conflict = resolve_correction_set(
        CorrectionSet.create("revision", (first.correction_id, second.correction_id)), (predecessor, first, second),
        base_revision_id="revision", selected_artifacts={"source": source},
    )
    assert not conflict.applicable_correction_ids
    assert {finding.kind for finding in conflict.findings} == {"correction_conflict"}
    assert all(finding.audit_category == "conflict" for finding in conflict.findings)
    assert all(finding.evidence.get("trigger") for finding in conflict.findings)


def test_stale_supersession_ancestor_makes_all_successors_inapplicable():
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    stale_payload = _source("revision", source)
    stale_payload["base"]["sha256"] = "0" * 64
    stale_ancestor = correction_record_for(stale_payload)
    successor = correction_record_for(_source("revision", source), supersedes_id=stale_ancestor.correction_id)
    descendant = correction_record_for(_source("revision", source), supersedes_id=successor.correction_id)
    correction_set = CorrectionSet.create("revision", [stale_ancestor.correction_id, successor.correction_id, descendant.correction_id])

    result = resolve_correction_set(
        correction_set,
        [stale_ancestor, successor, descendant],
        base_revision_id="revision",
        selected_artifacts={source.artifact_id: source},
    )

    reasons = {finding.evidence["correction_id"]: finding.evidence["reason"] for finding in result.findings}
    assert reasons[stale_ancestor.correction_id] == "base_hash_mismatch"
    assert reasons[successor.correction_id] == "supersedes_predecessor_inapplicable"
    assert reasons[descendant.correction_id] == "supersedes_predecessor_inapplicable"
    assert not result.applicable_correction_ids


def test_revision_mismatched_supersession_ancestor_makes_all_successors_inapplicable():
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    mismatched_ancestor = correction_record_for(_source("old-revision", source))
    successor = correction_record_for(_source("revision", source), supersedes_id=mismatched_ancestor.correction_id)
    descendant = correction_record_for(_source("revision", source), supersedes_id=successor.correction_id)
    correction_set = CorrectionSet.create("revision", [mismatched_ancestor.correction_id, successor.correction_id, descendant.correction_id])

    result = resolve_correction_set(
        correction_set,
        [mismatched_ancestor, successor, descendant],
        base_revision_id="revision",
        selected_artifacts={source.artifact_id: source},
    )

    reasons = {finding.evidence["correction_id"]: finding.evidence["reason"] for finding in result.findings}
    assert reasons[mismatched_ancestor.correction_id] == "record_revision_mismatch"
    assert reasons[successor.correction_id] == "supersedes_predecessor_inapplicable"
    assert reasons[descendant.correction_id] == "supersedes_predecessor_inapplicable"
    assert not result.applicable_correction_ids


def test_stale_inapplicable_conflict_ambiguous_superseded_and_protected_stay_visible():
    source = _artifact("source", {"segment_id": "segment"})
    translation = _artifact("translation", {"segment_id": "segment", "translated_text": "old", "mappings": [{"mapping_id": "map", "occurrence_id": "occ", "segment_id": "segment", "start": 0, "end": 2, "target_text": "old"}, {"mapping_id": "map", "occurrence_id": "occ", "segment_id": "segment", "start": 0, "end": 2, "target_text": "old"}]}, kind="TranslationArtifact")
    old = correction_record_for(_source("revision", source))
    newer = correction_record_for(_source("revision", source), supersedes_id=old.correction_id)
    conflict_one = correction_record_for({"kind": "target_segment", "applies_to_revision_id": "revision", "scope": {"segment_id": "other", "expected_target_text": ""}, "base": _base(_artifact("other", {"segment_id": "other", "translated_text": ""}, kind="TranslationArtifact")), "replacement": "one"})
    conflict_two = correction_record_for({"kind": "target_segment", "applies_to_revision_id": "revision", "scope": {"segment_id": "other", "expected_target_text": ""}, "base": _base(_artifact("other2", {"segment_id": "other", "translated_text": ""}, kind="TranslationArtifact")), "replacement": "two"})
    ambiguous = correction_record_for({"kind": "target_occurrence", "applies_to_revision_id": "revision", "scope": {"occurrence_id": "occ", "segment_id": "segment", "mapping_id": "map", "start": 0, "end": 2, "expected_target_text": "old"}, "base": _base(translation), "replacement": "new"})
    stale_payload = _source("revision", source)
    stale_payload["base"]["sha256"] = "0" * 64
    stale = correction_record_for(stale_payload)
    correction_set = CorrectionSet.create("revision", [old.correction_id, newer.correction_id, conflict_one.correction_id, conflict_two.correction_id, ambiguous.correction_id, stale.correction_id])
    artifacts = {"source": source, "translation": translation, "other": _artifact("other", {"segment_id": "other", "translated_text": ""}, kind="TranslationArtifact"), "other2": _artifact("other2", {"segment_id": "other", "translated_text": ""}, kind="TranslationArtifact")}
    result = resolve_correction_set(correction_set, [old, newer, conflict_one, conflict_two, ambiguous, stale], base_revision_id="revision", selected_artifacts=artifacts)
    kinds = {finding.kind for finding in result.findings}
    assert {"correction_stale", "correction_mapping_ambiguous", "correction_superseded", "correction_conflict"} <= kinds
    assert newer.correction_id in result.applicable_correction_ids

    mismatch = resolve_correction_set(correction_set, [old], base_revision_id="other-revision", selected_artifacts=artifacts)
    assert {finding.kind for finding in mismatch.findings} == {"correction_set_inapplicable"}


def test_event_rejects_alternate_superset_set_pointer_and_resolver_rejects_its_closure(tmp_path):
    source = _artifact("source", {"segment_id": "segment"})
    record = correction_record_for(_source("revision", source))
    active_ids = (record.correction_id,)
    store = CorrectionStore(tmp_path)
    store.put_record(record)

    first_event, first_set = _successor_event_set("revision", active_ids, record.correction_id, "apply")
    # This is a separately valid later successor with same base/active IDs,
    # but its exact event closure is a strict superset of first_set's.
    second_event, superset = _successor_event_set(
        "revision", active_ids, record.correction_id, "apply", first_event.event_id,
    )
    for correction_set, event in ((first_set, first_event), (superset, second_event)):
        store.put_set(correction_set)
        store.put_event(event)
    assert store.get_event(first_event.event_id) == first_event
    assert store.get_event(second_event.event_id) == second_event
    assert CorrectionEvent.from_json(second_event.to_json()) == second_event
    assert CorrectionSet.from_json(superset.to_json()) == superset

    # Merely replacing first event's pointer with valid same-state superset must
    # fail: first event hashes no other IDs, while superset contains second ID.
    first_path = tmp_path / "corrections" / "events" / f"{first_event.event_id}.json"
    second_path = tmp_path / "corrections" / "events" / f"{second_event.event_id}.json"
    first_path.write_text(replace(first_event, correction_set_id=superset.set_id).to_json(), encoding="utf-8")
    with pytest.raises(CorrectionError, match="linkage"):
        store.get_event(first_event.event_id)

    # Reverse pointer drops first event from second's required closure; reject
    # missing IDs too, then restore it for resolver-level superset rejection.
    first_path.write_text(first_event.to_json(), encoding="utf-8")
    second_path.write_text(replace(second_event, correction_set_id=first_set.set_id).to_json(), encoding="utf-8")
    with pytest.raises(CorrectionError, match="linkage"):
        store.get_event(second_event.event_id)
    second_path.write_text(second_event.to_json(), encoding="utf-8")
    first_path.write_text(replace(first_event, correction_set_id=superset.set_id).to_json(), encoding="utf-8")

    bundle = tmp_path / "sealed" / "revision"
    (bundle / "artifacts").mkdir(parents=True)
    (bundle / "bundle-manifest.json").write_bytes(canonical_json_bytes({"artifact_ids": [source.artifact_id]}))
    (bundle / "artifacts" / f"{source.artifact_id}.json").write_text(source.to_json(), encoding="utf-8")

    class Revisions:
        def snapshot(self, revision_id):
            assert revision_id == "revision"

        def _revision_path(self, revision_id):
            assert revision_id == "revision"
            return bundle

    result = resolve_selected_overlays(store, Revisions(), base_revision_id="revision", correction_set_id=superset.set_id)
    assert not result.applicable_correction_ids
    assert {finding.evidence["reason"] for finding in result.findings} == {"event_closure_incomplete"}


def test_missing_relation_evidence_rejects_each_kind_and_supersession_descendant():
    def occurrence(artifact):
        return {
            "kind": "target_occurrence", "applies_to_revision_id": "revision",
            "scope": {"occurrence_id": "occ", "segment_id": "segment", "mapping_id": "map", "start": 0, "end": 3, "expected_target_text": "old"},
            "base": _base(artifact), "replacement": "new",
        }

    def target_segment(artifact):
        return {
            "kind": "target_segment", "applies_to_revision_id": "revision",
            "scope": {"segment_id": "segment", "expected_target_text": "old"},
            "base": _base(artifact), "replacement": "new",
        }

    def terminology(projection, membership):
        return {
            "kind": "terminology", "applies_to_revision_id": "revision",
            "scope": {"concept_id": "concept", "selector": {"kind": "all_concept_occurrences"}},
            "base": {"projection": _base(projection), "membership": _base(membership)}, "replacement": "term",
        }

    valid_mapping = {
        "segment_id": "segment", "translated_text": "old",
        "mappings": [{"mapping_id": "map", "occurrence_id": "occ", "segment_id": "segment", "start": 0, "end": 3, "target_text": "old"}],
    }
    artifact_payloads = {
        "source-missing": {}, "source-valid": {"segment_id": "segment"},
        "occurrence-missing": {"segment_id": "segment"}, "occurrence-valid": valid_mapping,
        "target-missing": {"segment_id": "segment"}, "target-valid": {"segment_id": "segment", "translated_text": "old"},
        "projection-missing": {}, "membership-for-concept": {"concept_id": "concept", "occurrence_ids": ["occ"]},
        "projection-valid": {"concept_id": "concept"}, "membership-valid-concept": {"concept_id": "concept", "occurrence_ids": ["occ"]},
        "projection-for-membership-concept": {"concept_id": "concept"}, "membership-concept-missing": {"occurrence_ids": ["occ"]},
        "projection-valid-membership-concept": {"concept_id": "concept"}, "membership-valid-membership-concept": {"concept_id": "concept", "occurrence_ids": ["occ"]},
        "projection-for-membership": {"concept_id": "concept"}, "membership-missing": {"concept_id": "concept"},
        "projection-valid-membership": {"concept_id": "concept"}, "membership-valid": {"concept_id": "concept", "occurrence_ids": ["occ"]},
    }
    artifacts = {
        artifact_id: _artifact(
            artifact_id, payload,
            kind="TranslationArtifact" if artifact_id.startswith(("occurrence-", "target-")) else "test",
        )
        for artifact_id, payload in artifact_payloads.items()
    }
    cases = [
        ("source_relation_missing", _source("revision", artifacts["source-missing"]), _source("revision", artifacts["source-valid"])),
        ("target_relation_missing", occurrence(artifacts["occurrence-missing"]), occurrence(artifacts["occurrence-valid"])),
        ("target_relation_missing", target_segment(artifacts["target-missing"]), target_segment(artifacts["target-valid"])),
        ("concept_relation_missing", terminology(artifacts["projection-missing"], artifacts["membership-for-concept"]), terminology(artifacts["projection-valid"], artifacts["membership-valid-concept"])),
        ("concept_relation_missing", terminology(artifacts["projection-for-membership-concept"], artifacts["membership-concept-missing"]), terminology(artifacts["projection-valid-membership-concept"], artifacts["membership-valid-membership-concept"])),
        ("membership_relation_missing", terminology(artifacts["projection-for-membership"], artifacts["membership-missing"]), terminology(artifacts["projection-valid-membership"], artifacts["membership-valid"])),
    ]

    for expected_reason, invalid_payload, valid_payload in cases:
        ancestor = correction_record_for(invalid_payload)
        descendant = correction_record_for(valid_payload, supersedes_id=ancestor.correction_id)
        correction_set = CorrectionSet.create("revision", (ancestor.correction_id, descendant.correction_id))
        result = resolve_correction_set(correction_set, (ancestor, descendant), base_revision_id="revision", selected_artifacts=artifacts)
        reasons = {finding.evidence["correction_id"]: finding.evidence["reason"] for finding in result.findings}
        assert reasons[ancestor.correction_id] == expected_reason
        assert reasons[descendant.correction_id] == "supersedes_predecessor_inapplicable"
        assert not result.applicable_correction_ids


def _sealed_source_revision(tmp_path):
    """Small independently verifiable active base for correction-command tests."""
    artifacts = ArtifactStore(tmp_path)
    source = artifacts.put(
        "RawSourceSegment", {"segment_id": "segment", "source_text": "old"}, semantic_key="source-key",
    )
    revision_id = "revision"
    snapshot = RevisionSnapshot(revision_id=revision_id, selected_artifact_ids=(source.artifact_id,))
    provenance = {"test": "corrections"}
    epub = io.BytesIO()
    with zipfile.ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/btran-provenance.json", canonical_json_bytes(provenance))
    revisions = RevisionStore(tmp_path, artifacts)
    revisions.seal_bundle(snapshot, provenance, epub.getvalue())
    revisions.activate(revision_id)
    return artifacts, revisions, source, revision_id


def test_selected_closure_is_compact_correction_boundary(tmp_path):
    source = _artifact("source", {"segment_id": "segment", "source_text": "old"})
    record = correction_record_for(_source("revision", source))
    event, correction_set = _successor_event_set("revision", (record.correction_id,), record.correction_id, "apply")
    store = CorrectionStore(tmp_path)
    store.put_record(record)
    store.put_set(correction_set)
    store.put_event(event)

    class NoRevisionTraversal:
        def snapshot(self, revision_id):
            raise AssertionError("selected closure must avoid revision traversal")

        def _revision_path(self, revision_id):
            raise AssertionError("selected closure must avoid archive reads")

    closure = {source.artifact_id: source}
    result = resolve_selected_overlays(
        store, NoRevisionTraversal(), base_revision_id="revision",
        correction_set_id=correction_set.set_id, selected_closure=closure,
    )
    assert result.applicable_correction_ids == (record.correction_id,)
    assert result.source_inputs[0].replacement == "fixed"


def test_target_corrections_require_current_translation_leaf_and_span():
    effective = _artifact(
        "effective", {"segment_id": "segment", "translated_text": "old", "mappings": []},
        kind="EffectiveTargetSegment",
    )
    wrong_span = _artifact(
        "translation", {"segment_id": "segment", "translated_text": "other", "mappings": [
            {"mapping_id": "map", "occurrence_id": "occ", "segment_id": "segment", "start": 0, "end": 3, "target_text": "old"},
        ]}, kind="TranslationArtifact",
    )
    segment = correction_record_for({
        "kind": "target_segment", "applies_to_revision_id": "revision",
        "scope": {"segment_id": "segment", "expected_target_text": "old"},
        "base": _base(effective), "replacement": "new",
    })
    occurrence = correction_record_for({
        "kind": "target_occurrence", "applies_to_revision_id": "revision",
        "scope": {"occurrence_id": "occ", "segment_id": "segment", "mapping_id": "map", "start": 0, "end": 3, "expected_target_text": "old"},
        "base": _base(wrong_span), "replacement": "new",
    })
    result = resolve_correction_set(
        CorrectionSet.create("revision", (segment.correction_id, occurrence.correction_id)),
        (segment, occurrence), base_revision_id="revision",
        selected_artifacts={effective.artifact_id: effective, wrong_span.artifact_id: wrong_span},
    )
    assert not result.target_inputs
    reasons = {finding.evidence["correction_id"]: finding.evidence["reason"] for finding in result.findings}
    assert reasons[segment.correction_id] == "target_relation_missing"
    assert reasons[occurrence.correction_id] == "mapping_text_mismatch"


def test_apply_revert_and_supersede_publish_atomic_set_and_nonexecuting_impact(tmp_path):
    _, revisions, source, revision_id = _sealed_source_revision(tmp_path)
    store = CorrectionStore(tmp_path)
    first_payload = _source(revision_id, source)
    first, impact = correction_transition(store, revisions, event_kind="apply", payload=first_payload)
    pointer = (tmp_path / "active-correction-set.json").read_bytes()
    assert first.active_correction_ids
    assert impact.projection_plan_id and impact.regenerated == ()
    assert impact.correction_id == first.active_correction_ids[0]
    assert impact.correction_set_id == first.set_id
    assert store.correction_time_impact(impact.correction_id) == impact
    partition = (*impact.affected, *impact.unaffected, *impact.ambiguous, *impact.protected)
    assert len(partition) == len({(item["stage"], item["subject_id"], item["base_artifact_id"]) for item in partition})
    assert {(item["stage"], item["subject_id"], item["base_artifact_id"]) for item in partition} == {
        (item["stage"], item["subject_id"], item["base_artifact_id"]) for item in impact.projected_universe
    }
    assert {item["base_artifact_id"] for item in impact.reused} <= {
        item["base_artifact_id"] for item in (*impact.unaffected, *impact.protected)
    }

    # Invalid stale base is rejected before mutable pointer publication.
    stale = _source(revision_id, source)
    stale["base"]["sha256"] = "0" * 64
    with pytest.raises(CorrectionError):
        correction_transition(store, revisions, event_kind="apply", payload=stale)
    assert (tmp_path / "active-correction-set.json").read_bytes() == pointer

    reverted, reverted_impact = correction_transition(
        store, revisions, event_kind="revert", correction_id=first.active_correction_ids[0], revision_id=revision_id,
    )
    assert not reverted.active_correction_ids and reverted_impact.regenerated == ()
    newer = _source(revision_id, source)
    newer["replacement"] = "newer"
    # Re-apply then supersede proves successor state/pointer progression.
    applied, _ = correction_transition(store, revisions, event_kind="apply", payload=first_payload)
    successor, successor_impact = correction_transition(
        store, revisions, event_kind="supersede", supersedes_id=applied.active_correction_ids[0], payload=newer,
    )
    assert successor.active_correction_ids != applied.active_correction_ids
    assert successor_impact.projection_plan_id != impact.projection_plan_id


def test_correction_cli_apply_revert_supersede_and_revision_activation_keep_pointers(tmp_path, monkeypatch, capsys):
    _, _, source, revision_id = _sealed_source_revision(tmp_path)
    payload_path = tmp_path / "apply.json"
    payload = _source(revision_id, source)
    payload_path.write_bytes(canonical_json_bytes(payload))
    monkeypatch.setattr(sys, "argv", ["btran", "correction", "apply", str(tmp_path), "--payload", str(payload_path)])
    cli_main()
    store = CorrectionStore(tmp_path)
    active = store.get_set(json.loads((tmp_path / "active-correction-set.json").read_text())["set_id"])
    correction_id = active.active_correction_ids[0]
    before_revision_activate = (tmp_path / "active-correction-set.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["btran", "revision", "activate", str(tmp_path), revision_id])
    cli_main()
    assert (tmp_path / "active-correction-set.json").read_bytes() == before_revision_activate

    monkeypatch.setattr(sys, "argv", ["btran", "correction", "revert", str(tmp_path), "--correction-id", correction_id, "--revision", revision_id])
    cli_main()
    newer = _source(revision_id, source); newer["replacement"] = "newer"
    payload_path.write_bytes(canonical_json_bytes(newer))
    monkeypatch.setattr(sys, "argv", ["btran", "correction", "apply", str(tmp_path), "--payload", str(payload_path)])
    cli_main()
    active = store.get_set(json.loads((tmp_path / "active-correction-set.json").read_text())["set_id"])
    old_id = active.active_correction_ids[0]
    replacement = _source(revision_id, source); replacement["replacement"] = "latest"
    payload_path.write_bytes(canonical_json_bytes(replacement))
    monkeypatch.setattr(sys, "argv", ["btran", "correction", "supersede", str(tmp_path), "--supersedes", old_id, "--payload", str(payload_path)])
    cli_main()
    assert "correction_supersede" in capsys.readouterr().out
