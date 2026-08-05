"""Task-11 reconciliation artifacts are informational and immutable."""
from __future__ import annotations

from btran.artifacts import ArtifactStore
from btran.reconciliation import reconcile_effective
from btran.schema import EffectivePage, EffectiveSegment


def _target_inputs(tmp_path, *, text="A creature sleeps."):
    store = ArtifactStore(tmp_path / "store")
    segment = EffectiveSegment(
        effective_segment_id="effective-segment-1", segment_id="segment-1", source_lang="ja",
        source_text="猫", effective_text=text, render_lang="en", mode="translated",
        translation_artifact_id="translation-1",
    )
    target = store.put("EffectiveTargetSegment", segment.to_dict(), semantic_key="target-segment-1")
    page = EffectivePage(
        effective_page_id="effective-page-1", page_id="page-1",
        effective_segment_ids=(segment.effective_segment_id,), source_langs=("ja",),
    )
    target_page = store.put("EffectiveTargetPage", page.to_dict(), dependency_ids=(target.artifact_id,), semantic_key="target-page-1")
    membership = store.put("ConceptMembership", {"membership": "cat"}, semantic_key="membership-1")
    projection = store.put("ConceptProjection", {
        "projection_id": "projection-1", "concept_id": "cat", "membership_id": membership.artifact_id,
        "selector_occurrence_ids": ["occurrence-1"], "target_form": "cat", "correction_id": None,
    }, dependency_ids=(membership.artifact_id,), semantic_key="projection-1")
    return store, target_page.artifact_id, projection.artifact_id, membership.artifact_id


def test_reconcile_effective_persists_missing_term_assessment_and_optional_review(tmp_path):
    store, page_id, projection_id, membership_id = _target_inputs(tmp_path)

    result = reconcile_effective(
        effective_pages=(page_id,), projections=(projection_id,), store=store,
        base_revision_id="revision-1",
    )

    assert result.status == "completed"
    assert [issue.kind for issue in result.issues] == ["missing_term"]
    findings = [store.get_finding(item) for item in result.finding_ids]
    requests = [item for item in findings if item.kind == "review_request"]
    assert all(item.requires_action is False for item in requests)
    request = next(item for item in requests if item.evidence["trigger"] == "missing_term")
    assert request.evidence == {
        "trigger": "missing_term", "suggested_correction_kind": "terminology",
        "applicable_subject_ids": ["cat", "segment-1"], "base_revision_id": "revision-1",
        "base_artifact_ids": sorted([membership_id, projection_id]), "scope": "all_concept_occurrences",
    }
    assert any(item.kind == "stage_summary" for item in findings)
    missing = next(item for item in findings if item.kind == "missing_term")
    assert missing.audit_category == "validation"
    assert missing.evidence["trigger"] == "validation"
    assert result.assessment_artifact_ids


def test_context_conflict_preserves_full_selected_projection_closure(tmp_path):
    store, page_id, first_projection_id, _ = _target_inputs(tmp_path, text="cat feline creature")

    def projection(*, key, concept_id, occurrence_id, target_form):
        membership = store.put("ConceptMembership", {"membership": key}, semantic_key=f"membership-{key}")
        return store.put("ConceptProjection", {
            "projection_id": key, "concept_id": concept_id, "membership_id": membership.artifact_id,
            "selector_occurrence_ids": [occurrence_id], "target_form": target_form, "correction_id": None,
        }, dependency_ids=(membership.artifact_id,), semantic_key=f"projection-{key}").artifact_id

    conflicting_projection_id = projection(key="projection-2", concept_id="feline", occurrence_id="occurrence-1", target_form="feline")
    unrelated_projection_id = projection(key="projection-3", concept_id="creature", occurrence_id="occurrence-2", target_form="creature")
    selected_ids = tuple(sorted((first_projection_id, conflicting_projection_id, unrelated_projection_id)))

    result = reconcile_effective(
        effective_pages=(page_id,), projections=selected_ids, store=store,
        base_revision_id="revision-1",
    )

    conflict = next(issue for issue in result.issues if issue.kind == "context_conflict")
    assert set(conflict.evidence["projection_ids"]) == {first_projection_id, conflicting_projection_id}
    assert result.projection_artifact_ids == selected_ids
    persisted = store.get(result.artifact_id)
    assert tuple(persisted.payload["projection_artifact_ids"]) == selected_ids
    assert persisted.dependency_ids == tuple(sorted((page_id, *selected_ids)))


def test_declared_page_and_segment_order_wins_over_artifact_order(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    first = EffectiveSegment(
        effective_segment_id="effective-segment-z", segment_id="segment-z", source_lang="ja",
        source_text="猫", effective_text="cat", render_lang="en", mode="translated",
        translation_artifact_id="translation-z",
    )
    second = EffectiveSegment(
        effective_segment_id="effective-segment-a", segment_id="segment-a", source_lang="ja",
        source_text="犬", effective_text="dog", render_lang="en", mode="translated",
        translation_artifact_id="translation-a",
    )
    first_artifact = store.put("EffectiveTargetSegment", first.to_dict(), semantic_key="target-z")
    second_artifact = store.put("EffectiveTargetSegment", second.to_dict(), semantic_key="target-a")
    page = EffectivePage(
        effective_page_id="effective-page-order", page_id="page-order",
        effective_segment_ids=(first.effective_segment_id, second.effective_segment_id), source_langs=("ja",),
    )
    page_artifact = store.put(
        "EffectiveTargetPage", page.to_dict(),
        # ArtifactStore canonicalizes these dependencies; the page declaration
        # remains the only valid child order.
        dependency_ids=(second_artifact.artifact_id, first_artifact.artifact_id),
        semantic_key="target-page-order",
    )
    membership = store.put("ConceptMembership", {"membership": "order"}, semantic_key="membership-order")
    projection = store.put(
        "ConceptProjection", {
            "projection_id": "projection-order", "concept_id": "order", "membership_id": membership.artifact_id,
            "selector_occurrence_ids": [], "target_form": "missing", "correction_id": None,
        }, dependency_ids=(membership.artifact_id,), semantic_key="projection-order",
    )
    result = reconcile_effective(
        effective_pages=(page_artifact.artifact_id,), projections=(projection.artifact_id,), store=store,
        base_revision_id="revision-1",
    )
    assert result.status == "completed"
    assert result.issues[0].evidence["segment_ids"] == ["segment-z", "segment-a"]
    persisted = store.get(result.artifact_id)
    assert persisted.payload["effective_page_artifact_ids"] == [page_artifact.artifact_id]


def test_reconciliation_reads_only_explicitly_selected_pages_and_projections(tmp_path):
    store, page_id, _, _ = _target_inputs(tmp_path, text="cat")
    membership = store.put("ConceptMembership", {"membership": "historic"}, semantic_key="historic-membership")
    historic = store.put("ConceptProjection", {
        "projection_id": "historic", "concept_id": "historic", "membership_id": membership.artifact_id,
        "selector_occurrence_ids": ["historic-occurrence"], "target_form": "missing", "correction_id": None,
    }, dependency_ids=(membership.artifact_id,), semantic_key="historic-projection")

    result = reconcile_effective(effective_pages=(page_id,), projections=(), store=store, base_revision_id="revision-1")

    assert result.status == "completed"
    assert result.projection_artifact_ids == ()
    assert result.issues == ()
    # Presence in immutable cache history cannot become selected input.
    assert store.get(historic.artifact_id).artifact_id == historic.artifact_id


def test_reconcile_effective_exception_keeps_selected_projections_and_returns_fallback(tmp_path):
    store, _, projection_id, _ = _target_inputs(tmp_path)

    result = reconcile_effective(
        effective_pages=("missing-page",), projections=(projection_id,), store=store,
        base_revision_id="revision-1",
    )

    assert result.status == "degraded"
    assert result.projection_artifact_ids == (projection_id,)
    assert result.issues == ()
    assert result.error_evidence["exception_type"]
    findings = [store.get_finding(item) for item in result.finding_ids]
    assert {item.kind for item in findings} >= {"reconciliation_exception", "reconciliation_fallback", "uncertainty", "review_request", "stage_summary"}
    assert {item.audit_category for item in findings if item.kind in {"reconciliation_exception", "reconciliation_fallback"}} == {"failure", "fallback"}
    request = next(item for item in findings if item.kind == "review_request")
    assert request.requires_action is False
    assert request.evidence["trigger"] == "degraded_unknown_confidence"
