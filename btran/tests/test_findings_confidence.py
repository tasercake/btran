"""Task-1 finding, confidence, and non-gating review-request contracts."""

import importlib.util

import pytest

from btran.schema import (
    ConfidenceAssessment,
    SchemaError,
    review_requests_for,
    uncertainty_finding,
)


BASE = {
    "stage": "terminology",
    "subject_ids": ("occurrence-1",),
    "suggested_correction_kind": "terminology",
    "base_revision_id": "revision-1",
    "base_artifact_ids": ("artifact-1",),
    "scope": "occurrence",
}


def _assessment(score, *, stage="terminology", signals=()):
    return ConfidenceAssessment(
        subject_id="occurrence-1", producing_stage=stage,
        producing_artifact_id="artifact-1", score=score, signals=signals,
    )


@pytest.mark.parametrize(
    ("kwargs", "trigger"),
    [
        ({"assessment": _assessment(0.79)}, "low_confidence"),
        ({"assessment": _assessment(None), "degraded_or_fallback": True}, "degraded_unknown_confidence"),
        ({"assessment": _assessment(None, signals=("model_fallback",))}, "degraded_unknown_confidence"),
        ({"ambiguity": "source_sense"}, "source_sense_ambiguity"),
        ({"ambiguity": "concept"}, "concept_ambiguity"),
        ({"ambiguity": "mapping"}, "mapping_ambiguity"),
        ({"ambiguity": "correction"}, "correction_ambiguity"),
        ({"reconciliation_issue": "reconciliation_conflict"}, "reconciliation_conflict"),
        ({"reconciliation_issue": "missing_term"}, "missing_term"),
        ({"validation_error": True}, "validation_error"),
    ],
)
def test_every_review_trigger_is_non_gating_and_has_exact_evidence(kwargs, trigger):
    requests = review_requests_for(**BASE, **kwargs)
    assert len(requests) == 1
    finding = requests[0]
    assert finding.kind == "review_request"
    assert finding.requires_action is False
    assert finding.evidence == {
        "trigger": trigger,
        "suggested_correction_kind": "terminology",
        "applicable_subject_ids": ["occurrence-1"],
        "base_revision_id": "revision-1",
        "base_artifact_ids": ["artifact-1"],
        "scope": "occurrence",
    }


def test_confidence_range_signals_and_uncertainty_finding_are_strict():
    assessment = _assessment(0.5, signals=("model_low_confidence",))
    finding = uncertainty_finding(assessment)
    assert finding.kind == "uncertainty"
    assert finding.requires_action is False
    assert assessment.uncertainty_finding_id == finding.finding_id
    with pytest.raises(SchemaError):
        _assessment(1.01)
    with pytest.raises(SchemaError):
        _assessment(0.5, signals=("z", "a"))


def test_none_confidence_without_degraded_artifact_does_not_request_review():
    assert review_requests_for(**BASE, assessment=_assessment(None)) == ()


def test_review_module_is_absent_and_requests_are_findings_only():
    assert importlib.util.find_spec("btran.review") is None


def test_subset_scope_carries_exact_occurrence_ids():
    finding = review_requests_for(
        **{**BASE, "scope": "subset_occurrence_ids", "occurrence_ids": ("occurrence-2", "occurrence-3")},
        ambiguity="mapping",
    )[0]
    assert finding.evidence["occurrence_ids"] == ["occurrence-2", "occurrence-3"]
