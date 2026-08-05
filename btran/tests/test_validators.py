"""Task-11 validation remains diagnostic content, never a gate."""
from __future__ import annotations

from btran.artifacts import ArtifactStore
from btran.orchestrator_contract import OrderedEffectivePage, SelectedEffectiveContent
from btran.reconciliation import reconcile_effective
from btran.schema import EffectivePage, EffectiveSegment
from btran.validators import (
    check_block_schema,
    detect_language,
    validate_effective,
)


def _inputs(tmp_path, *, text="The cat sleeps."):
    store = ArtifactStore(tmp_path / "store")
    segment = EffectiveSegment(
        effective_segment_id="effective-segment-1", segment_id="segment-1", source_lang="en",
        source_text="The cat sleeps.", effective_text=text, render_lang="en", mode="translated",
        translation_artifact_id="translation-1",
    )
    target = store.put("EffectiveTargetSegment", segment.to_dict(), semantic_key="target-segment-1")
    page = EffectivePage(effective_page_id="effective-page-1", page_id="page-1", effective_segment_ids=(segment.effective_segment_id,), source_langs=("en",))
    target_page = store.put("EffectiveTargetPage", page.to_dict(), dependency_ids=(target.artifact_id,), semantic_key="target-page-1")
    projection = store.put("ConceptProjection", {
        "projection_id": "projection-1", "concept_id": "cat", "membership_id": "membership-1",
        "selector_occurrence_ids": ["occurrence-1"], "target_form": "cat", "correction_id": None,
    }, semantic_key="projection-1")
    reconciliation = reconcile_effective(effective_pages=(target_page.artifact_id,), projections=(projection.artifact_id,), store=store, base_revision_id="revision-1")
    return store, target_page.artifact_id, reconciliation


def test_legacy_helpers_remain_deterministic_migration_utilities():
    from btran.schema import SourceBlock
    assert check_block_schema([SourceBlock("b1", "paragraph", "text", 0)]) == []
    assert detect_language("Bonjour le monde et merci") == "fr"


def test_validator_rule_exception_is_persisted_and_other_rules_continue(tmp_path):
    store, page_id, reconciliation = _inputs(tmp_path)
    called = []

    def broken(*_):
        raise RuntimeError("boom")

    def still_runs(*_):
        called.append(True)
        return ("intentional validation error",)

    result = validate_effective(
        effective_pages=(page_id,), reconciliation=reconciliation, store=store,
        base_revision_id="revision-1", mode="translated",
        rules={"broken": broken, "still_runs": still_runs},
    )

    assert result.status == "degraded"
    assert called == [True]
    assert [item.rule for item in result.rule_results] == ["broken", "still_runs"]
    findings = [store.get_finding(item) for item in result.finding_ids]
    assert {item.kind for item in findings} >= {"validator_exception", "validation_error", "review_request", "stage_summary"}
    requests = [item for item in findings if item.kind == "review_request"]
    assert all(item.requires_action is False for item in requests)
    assert {item.evidence["trigger"] for item in requests} >= {"degraded_unknown_confidence", "validation_error"}


def test_validation_setup_failure_returns_degraded_artifact(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    result = validate_effective(
        effective_pages=("missing-page",), reconciliation=object(), store=store,
        base_revision_id="revision-1", mode="translated",
    )
    assert result.status == "degraded"
    findings = [store.get_finding(item) for item in result.finding_ids]
    assert {item.kind for item in findings} >= {"validation_exception", "stage_summary"}


def test_validation_setup_failure_with_selected_page_keeps_sorted_fallback_signals(tmp_path):
    store, page_id, reconciliation = _inputs(tmp_path)
    result = validate_effective(
        effective_pages=(page_id,), reconciliation=reconciliation, store=store,
        base_revision_id="revision-1", mode="translated", rules={},
    )
    assert result.status == "degraded"
    findings = [store.get_finding(item) for item in result.finding_ids]
    assert {item.kind for item in findings} >= {
        "validation_exception", "uncertainty", "review_request", "stage_summary",
    }


def test_validation_semantic_key_and_dependency_change_with_selected_reconciliation(tmp_path):
    store, page_id, first = _inputs(tmp_path, text="cat")
    membership = store.put("ConceptMembership", {"membership": "membership-2"}, semantic_key="membership-2")
    alternate_projection = store.put("ConceptProjection", {
        "projection_id": "projection-2", "concept_id": "cat", "membership_id": membership.artifact_id,
        "selector_occurrence_ids": ["occurrence-2"], "target_form": "feline", "correction_id": None,
    }, dependency_ids=(membership.artifact_id,), semantic_key="projection-2")
    second = reconcile_effective(effective_pages=(page_id,), projections=(alternate_projection.artifact_id,),
                                 store=store, base_revision_id="revision-1")

    first_validation = validate_effective(effective_pages=(page_id,), reconciliation=first, store=store,
                                          base_revision_id="revision-1", mode="translated")
    second_validation = validate_effective(effective_pages=(page_id,), reconciliation=second, store=store,
                                           base_revision_id="revision-1", mode="translated")

    assert first.artifact_id != second.artifact_id
    assert first_validation.reconciliation_artifact_id == first.artifact_id
    assert second_validation.reconciliation_artifact_id == second.artifact_id
    assert first.artifact_id in store.get(first_validation.artifact_id).dependency_ids
    assert second.artifact_id in store.get(second_validation.artifact_id).dependency_ids
    assert store.get(first_validation.artifact_id).semantic_key != store.get(second_validation.artifact_id).semantic_key


def test_validation_preserves_selected_page_and_declared_segment_order(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    page_ids = []
    for page_name in ("page-a", "page-b"):
        segment = EffectiveSegment(
            effective_segment_id=f"{page_name}-segment", segment_id=f"{page_name}-source",
            source_lang="en", source_text=page_name, effective_text=page_name,
            render_lang="en", mode="native",
        )
        target = store.put("EffectiveTargetSegment", segment.to_dict(), semantic_key=f"target-{page_name}")
        page = EffectivePage(
            effective_page_id=f"effective-{page_name}", page_id=page_name,
            effective_segment_ids=(segment.effective_segment_id,), source_langs=("en",),
        )
        target_page = store.put("EffectiveTargetPage", page.to_dict(), dependency_ids=(target.artifact_id,), semantic_key=f"page-{page_name}")
        page_ids.append(target_page.artifact_id)
    reconciliation = reconcile_effective(effective_pages=tuple(reversed(page_ids)), projections=(), store=store, base_revision_id="revision-1")
    observed = []

    def ordered(pages, *_):
        observed.extend(page.page_id for _, page, segments in pages for _, segment in segments)
        return ()

    result = validate_effective(
        effective_pages=tuple(reversed(page_ids)), reconciliation=reconciliation,
        store=store, base_revision_id="revision-1", mode="native", rules={"ordered": ordered},
    )
    assert observed == ["page-b", "page-a"]
    assert result.effective_page_artifact_ids == tuple(reversed(page_ids))
    payload = store.get(result.artifact_id).payload
    assert payload["effective_page_artifact_ids"] == list(reversed(page_ids))


def test_validation_findings_use_fc7_categories_and_keep_continuation(tmp_path):
    store, page_id, reconciliation = _inputs(tmp_path)

    def errors(*_):
        return ("missing_term", "context_conflict for selected mapping", "ambiguous mapping")

    def broken(*_):
        raise RuntimeError("validator broke")

    result = validate_effective(
        effective_pages=(page_id,), reconciliation=reconciliation, store=store,
        base_revision_id="revision-1", mode="translated",
        rules={"errors": errors, "broken": broken},
    )
    findings = [store.get_finding(item) for item in result.finding_ids]
    categories = {item.audit_category for item in findings if item.audit_category is not None}
    assert {"validation", "actionable_ambiguity", "failure", "fallback"} <= categories
    assert not any(
        item.audit_category == "conflict"
        for item in findings
    )
    assert all(item.requires_action is False for item in findings)
    assert any(item.kind == "stage_summary" and item.audit_category is None for item in findings)


def test_validation_consumes_selected_effective_content_without_store_reads(tmp_path, monkeypatch):
    store, page_id, reconciliation = _inputs(tmp_path)
    page_envelope = store.get(page_id)
    segment_envelope = store.get(page_envelope.dependency_ids[0])
    selected = SelectedEffectiveContent((OrderedEffectivePage(
        EffectivePage.from_dict(page_envelope.payload),
        (EffectiveSegment.from_dict(segment_envelope.payload),),
    ),))
    observed = []

    def ordered(pages, *_):
        observed.extend((page_id, segment.segment_id) for page_id, _, segments in pages for _, segment in segments)
        return ()

    def fail_reload(_):
        raise AssertionError("selected closure must not be reloaded from ArtifactStore")

    monkeypatch.setattr(store, "get", fail_reload)
    result = validate_effective(
        effective_pages=selected, reconciliation=reconciliation, store=store,
        base_revision_id="revision-1", mode="translated", rules={"ordered": ordered},
    )

    assert result.status == "completed"
    assert observed == [("effective-page-1", "segment-1")]
    assert result.effective_page_artifact_ids == ("effective-page-1",)


def test_native_validation_uses_source_equivalent_rules_and_omits_target_rules(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    segment = EffectiveSegment(effective_segment_id="effective-segment-1", segment_id="segment-1", source_lang="en", source_text="The cat sleeps.", effective_text="The cat sleeps.", render_lang="en", mode="native")
    target = store.put("EffectiveTargetSegment", segment.to_dict(), semantic_key="target-segment-1")
    page = EffectivePage(effective_page_id="effective-page-1", page_id="page-1", effective_segment_ids=(segment.effective_segment_id,), source_langs=("en",))
    target_page = store.put("EffectiveTargetPage", page.to_dict(), dependency_ids=(target.artifact_id,), semantic_key="target-page-1")
    reconciliation = reconcile_effective(effective_pages=(target_page.artifact_id,), projections=(), store=store, base_revision_id="revision-1")
    result = validate_effective(effective_pages=(target_page.artifact_id,), reconciliation=reconciliation, store=store, base_revision_id="revision-1", mode="native")
    assert {item.rule for item in result.rule_results} == {"effective_structure", "non_empty_text", "source_language"}
