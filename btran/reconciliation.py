"""Informational reconciliation over selected effective content.

This module never changes a glossary, schedules translation, or asks an operator to
answer before returning.  Its output is an immutable, inspectable stage artifact.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from btran.artifacts import ArtifactStore, artifact_id_for
from btran.schema import (
    ConfidenceAssessment,
    EffectivePage,
    EffectiveSegment,
    Finding,
    actionable_uncertainty_finding,
    PageExtraction,
    TerminologyEntry,
    TerminologyMap,
    TranslatedBlock,
    canonical_json,
    review_requests_for,
    stage_summary_finding,
    uncertainty_finding,
)

RECONCILIATION_ARTIFACT_KIND = "ReconciliationArtifact"
RECONCILIATION_ASSESSMENT_KIND = "ConfidenceAssessment"


@dataclass(frozen=True)
class GlossaryChange:
    """Legacy migration value; reconciliation no longer creates these."""
    concept_id: str
    old_target_term: str
    new_target_term: str


@dataclass(frozen=True)
class TermIssue:
    """Legacy-compatible, non-mutating terminology observation."""
    concept_id: str
    kind: str
    pages: tuple[int, ...] = ()
    expected_target_term: str = ""
    observed_target_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReconciliationIssue:
    issue_id: str
    kind: str
    subject_ids: tuple[str, ...]
    base_artifact_ids: tuple[str, ...]
    expected_target_term: str = ""
    observed_target_terms: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "kind": self.kind,
            "subject_ids": list(self.subject_ids),
            "base_artifact_ids": list(self.base_artifact_ids),
            "expected_target_term": self.expected_target_term,
            "observed_target_terms": list(self.observed_target_terms),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class ReconciliationArtifact:
    """Typed selected-input reconciliation result.

    ``projection_artifact_ids`` are deliberately unchanged on every fallback.
    They are selected input, never a mutable reconciliation output.
    """
    artifact_id: str
    effective_page_artifact_ids: tuple[str, ...]
    projection_artifact_ids: tuple[str, ...]
    issues: tuple[ReconciliationIssue, ...]
    finding_ids: tuple[str, ...]
    assessment_artifact_ids: tuple[str, ...]
    stage_summary_finding_id: str
    status: str
    error_evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    """Read-only compatibility view for migration callers.

    It intentionally contains no reviewer callback result and no changed glossary.
    """
    glossary_v2: TerminologyMap
    glossary_diff: list[GlossaryChange]
    issues: list[TermIssue]
    affected_pages: list[int]


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _term_in_text(term: str, text: str, *, allow_regular_variant: bool = False) -> bool:
    normalized_term = _normalize_text(term)
    normalized_text = _normalize_text(text)
    if not normalized_term:
        return False
    if normalized_term.isascii() and normalized_term.replace(" ", "").isalnum():
        forms = [normalized_term]
        if allow_regular_variant:
            forms.extend((normalized_term + "s", normalized_term + "es", normalized_term + "'s", normalized_term + "’s"))
            if len(normalized_term) > 1 and normalized_term.endswith("y") and normalized_term[-2] not in "aeiou":
                forms.append(normalized_term[:-1] + "ies")
        return re.search(rf"(?<!\w)(?:{'|'.join(re.escape(form) for form in sorted(forms, key=len, reverse=True))})(?!\w)", normalized_text) is not None
    return normalized_term in normalized_text


def _entry_mentions_page(entry: TerminologyEntry, extraction: PageExtraction) -> bool:
    mentioned = {_normalize_text(mention.term) for mention in extraction.term_mentions}
    source_text = "\n".join(block.text for block in extraction.blocks)
    return any(_normalize_text(term) in mentioned or _term_in_text(term, source_text) for term in entry.source_terms)


def index_terms_to_pages(extractions: list[PageExtraction], glossary: TerminologyMap) -> dict[str, set[int]]:
    index: dict[str, set[int]] = {}
    for extraction in extractions:
        for entry in glossary.entries:
            if _entry_mentions_page(entry, extraction):
                index.setdefault(entry.concept_id, set()).add(extraction.page_number)
    return index


def glossary_diff(v1: TerminologyMap, v2: TerminologyMap) -> list[GlossaryChange]:
    old, new = ({entry.concept_id: entry for entry in value.entries} for value in (v1, v2))
    return [GlossaryChange(key, old[key].target_term, new[key].target_term) for key in sorted(old.keys() & new.keys()) if old[key].target_term != new[key].target_term]


def _translated_page_text(blocks: list[TranslatedBlock] | None) -> str:
    return "\n".join(block.translated_text for block in blocks or [])


def _legacy_issues(*, glossary: TerminologyMap, extractions: list[PageExtraction], translations: dict[int, list[TranslatedBlock]]) -> list[TermIssue]:
    page_index = index_terms_to_pages(extractions, glossary)
    known_targets = [(entry.target_term, entry.concept_id) for entry in glossary.entries]
    issues: list[TermIssue] = []
    for entry in glossary.entries:
        missing, conflicts, observed = [], [], set()
        for page_number in sorted(page_index.get(entry.concept_id, ())):
            text = _translated_page_text(translations.get(page_number))
            if _term_in_text(entry.target_term, text, allow_regular_variant=True):
                continue
            competing = [target for target, concept_id in known_targets if concept_id != entry.concept_id and _term_in_text(target, text, allow_regular_variant=True)]
            if competing:
                conflicts.append(page_number); observed.update(competing)
            else:
                missing.append(page_number)
        if missing:
            issues.append(TermIssue(entry.concept_id, "missing_term", tuple(missing), entry.target_term))
        if conflicts:
            issues.append(TermIssue(entry.concept_id, "context_conflict", tuple(conflicts), entry.target_term, tuple(sorted(observed))))
    return issues


def reconcile(*, glossary: TerminologyMap, extractions: list[PageExtraction], translations: dict[int, list[TranslatedBlock]]) -> ReconciliationResult:
    """Migration inspection only. No callback, mutation, or retranslation contract."""
    issues = _legacy_issues(glossary=glossary, extractions=extractions, translations=translations)
    affected = sorted({page for issue in issues for page in issue.pages})
    return ReconciliationResult(glossary, [], issues, affected)


def _issue_id(kind: str, subjects: Sequence[str], bases: Sequence[str], evidence: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json({"kind": kind, "subject_ids": sorted(set(subjects)), "base_artifact_ids": sorted(set(bases)), "evidence": dict(evidence)}).encode()).hexdigest()


def _selected_pages(value: Any, store: ArtifactStore) -> tuple[tuple[str, EffectivePage, tuple[tuple[str, EffectiveSegment], ...]], ...]:
    """Load selected pages once, rebuilding children in declared order.

    Dependency rows are canonicalized as sets by the artifact store.  They are
    therefore never a valid source of reading order; ``effective_segment_ids``
    on the selected page is the authority.
    """
    leaves = getattr(value, "leaves", value)
    if not isinstance(leaves, (tuple, list)):
        raise ValueError("effective pages must be Task-10 leaves or selected page artifact IDs")
    page_ids = [leaf.page_artifact_id if hasattr(leaf, "page_artifact_id") else leaf for leaf in leaves]
    output = []
    for page_id in page_ids:
        if not isinstance(page_id, str):
            raise ValueError("effective page ID is invalid")
        envelope = store.get(page_id)
        if envelope.kind != "EffectiveTargetPage":
            raise ValueError("reconciliation requires selected effective target pages")
        page = EffectivePage.from_dict(envelope.payload)
        children: dict[str, tuple[str, EffectiveSegment]] = {}
        for dependency_id in envelope.dependency_ids:
            child = store.get(dependency_id)
            if child.kind not in {"EffectiveTargetSegment", "DiagnosticEffectiveTargetSegment"}:
                raise ValueError("effective page has an unexpected child artifact")
            segment = EffectiveSegment.from_dict(child.payload)
            if segment.effective_segment_id in children:
                raise ValueError("effective page has duplicate segment identity")
            children[segment.effective_segment_id] = (child.artifact_id, segment)
        if set(children) != set(page.effective_segment_ids):
            raise ValueError("effective page segment closure/order is invalid")
        try:
            # Lookup by the page's declared IDs, not by dependency/storage order.
            segments = tuple(children[segment_id] for segment_id in page.effective_segment_ids)
        except KeyError as exc:
            raise ValueError("effective page segment closure/order is invalid") from exc
        output.append((envelope.artifact_id, page, segments))
    if len({page.page_id for _, page, _ in output}) != len(output):
        raise ValueError("effective pages duplicate page identity")
    return tuple(output)


def _selected_projections(value: Any, store: ArtifactStore) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    ids = getattr(value, "projection_artifact_ids", value)
    if ids is None:
        return ()
    if not isinstance(ids, (tuple, list)):
        raise ValueError("projections must be Task-8 run or selected artifact IDs")
    output = []
    for artifact_id in ids:
        envelope = store.get(artifact_id)
        if envelope.kind != "ConceptProjection":
            raise ValueError("reconciliation requires selected ConceptProjection artifacts")
        body = envelope.payload
        if not isinstance(body.get("concept_id"), str) or not isinstance(body.get("target_form"), str):
            raise ValueError("projection payload is invalid")
        output.append((envelope.artifact_id, body))
    return tuple(sorted(output))


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    """De-duplicate selected IDs without destroying declared execution order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _reconciliation_payload(page_ids: Sequence[str], projection_ids: Sequence[str], issues: Sequence[ReconciliationIssue], status: str, error: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"effective_page_artifact_ids": list(_ordered_unique(page_ids)), "projection_artifact_ids": list(_ordered_unique(projection_ids)), "issues": [item.to_dict() for item in issues], "status": status, "error_evidence": dict(error) if error else None, "algorithm_version": "reconciliation-v1"}


def _put_assessment(store: ArtifactStore, *, reconciliation_id: str, issue: ReconciliationIssue, base_revision_id: str) -> tuple[str, tuple[str, ...]]:
    # Keep the compatibility uncertainty record, but also expose actionable
    # evidence through the shared audit helper.  The helper deliberately does
    # not infer actionability from the confidence score.
    signal = "conflict" if issue.kind == "context_conflict" else "validation"
    assessment = ConfidenceAssessment(subject_id=issue.issue_id, producing_stage="reconciliation", producing_artifact_id=reconciliation_id, score=0, signals=tuple(sorted(("reconciliation_issue", signal))))
    uncertainty = actionable_uncertainty_finding(assessment)
    if uncertainty is None:  # defensive compatibility path for future signals
        uncertainty = uncertainty_finding(assessment)
    store.put_finding(uncertainty)
    request = review_requests_for(assessment=assessment, reconciliation_issue="reconciliation_conflict" if issue.kind == "context_conflict" else "missing_term", stage="reconciliation", subject_ids=issue.subject_ids, suggested_correction_kind="terminology", base_revision_id=base_revision_id, base_artifact_ids=issue.base_artifact_ids, scope="all_concept_occurrences")
    finding_ids = [uncertainty.finding_id]
    for finding in request:
        store.put_finding(finding); finding_ids.append(finding.finding_id)
    artifact = store.put(RECONCILIATION_ASSESSMENT_KIND, assessment.to_dict(), dependency_ids=(reconciliation_id,), finding_ids=tuple(sorted(finding_ids)), semantic_key=f"confidence:{reconciliation_id}:{issue.issue_id}")
    return artifact.artifact_id, tuple(sorted(finding_ids))


def reconcile_effective(*, effective_pages: Any, projections: Any, store: ArtifactStore, base_revision_id: str) -> ReconciliationArtifact:
    """Inspect selected effective pages/projections and always return typed artifact.

    A full-stage exception becomes unchanged-projection degraded output.  It is
    intentionally not propagated to validators/rendering.
    """
    page_ids: tuple[str, ...] = ()
    projection_ids: tuple[str, ...] = ()
    issues: tuple[ReconciliationIssue, ...] = ()
    findings: list[str] = []
    error: Mapping[str, Any] | None = None
    status = "completed"
    try:
        pages = _selected_pages(effective_pages, store)
        # The caller's page sequence is the declared content order.  Never
        # replace it with lexical artifact-ID order.
        page_ids = tuple(page_id for page_id, _, _ in pages)
        selected = _selected_projections(projections, store)
        projection_ids = tuple(item[0] for item in selected)
        texts = [(segment.segment_id, segment.effective_text) for _, _, segments in pages for _, segment in segments]
        observations: list[ReconciliationIssue] = []
        for projection_id, projection in selected:
            concept = projection["concept_id"]
            target = projection["target_form"]
            # A projection with no target form is a source-form/native fallback,
            # not a target-term defect.
            if not target:
                continue
            matching = [segment_id for segment_id, text in texts if _term_in_text(target, text, allow_regular_variant=True)]
            if not matching:
                segment_ids = tuple(segment_id for segment_id, _ in texts)
                subjects = tuple(sorted({concept, *segment_ids}))
                bases = tuple(sorted((projection_id, projection["membership_id"])))
                evidence = {"concept_id": concept, "projection_id": projection_id, "target_form": target, "segment_ids": list(segment_ids)}
                observations.append(ReconciliationIssue(_issue_id("missing_term", subjects, bases, evidence), "missing_term", subjects, bases, target, (), evidence))
        # Two selected projections assigning different forms to one exact occurrence are a conflict.
        occurrence_forms: dict[str, list[tuple[str, str, str]]] = {}
        for projection_id, projection in selected:
            for occurrence_id in projection.get("selector_occurrence_ids", []):
                occurrence_forms.setdefault(occurrence_id, []).append((projection_id, projection["concept_id"], projection["target_form"]))
        for occurrence_id, values in sorted(occurrence_forms.items()):
            forms = sorted({value[2] for value in values})
            if len(forms) > 1:
                conflict_projection_ids = tuple(sorted(value[0] for value in values))
                bases = tuple(sorted({*conflict_projection_ids, *(store.get(value[0]).payload["membership_id"] for value in values)}))
                subjects = tuple(sorted({occurrence_id, *(value[1] for value in values)}))
                evidence = {"occurrence_id": occurrence_id, "projection_ids": list(conflict_projection_ids), "target_forms": forms}
                observations.append(ReconciliationIssue(_issue_id("context_conflict", subjects, bases, evidence), "context_conflict", subjects, bases, "", tuple(forms), evidence))
        issues = tuple(sorted(observations, key=lambda item: item.issue_id))
    except Exception as exc:
        status = "degraded"
        error = {"exception_type": type(exc).__name__, "message": str(exc)}
        # Preserve known selected projection IDs even when later input parsing fails.
        raw = getattr(projections, "projection_artifact_ids", projections)
        if isinstance(raw, (tuple, list)) and all(isinstance(item, str) for item in raw):
            projection_ids = tuple(sorted(set(raw)))
        subjects = projection_ids or page_ids or ("reconciliation",)
        failure = Finding(
            kind="reconciliation_exception", severity="error", stage="reconciliation",
            audit_category="failure", subject_refs=tuple(sorted(set(subjects))),
            evidence={"trigger": "failure", **dict(error)},
            message="Reconciliation failed while loading or processing selected content; unchanged projections remain selected.",
            dependency_ids=tuple(sorted(set(projection_ids))),
        )
        store.put_finding(failure); findings.append(failure.finding_id)
        # The typed artifact is still returned, so continuation is separately
        # visible as fallback rather than hiding the primary failure.
        fallback = Finding(
            kind="reconciliation_fallback", severity="warning", stage="reconciliation",
            audit_category="fallback", subject_refs=tuple(sorted(set(subjects))),
            evidence={"trigger": "fallback", "continued_from": failure.finding_id, "selected_projection_ids": list(projection_ids)},
            message="Reconciliation continued with unchanged selected projections.",
            dependency_ids=tuple(sorted(set(projection_ids))),
        )
        store.put_finding(fallback); findings.append(fallback.finding_id)

    payload = _reconciliation_payload(page_ids, projection_ids, issues, status, error)
    reconciliation_id = artifact_id_for(RECONCILIATION_ARTIFACT_KIND, payload, tuple(sorted(set((*page_ids, *projection_ids)))))
    assessment_ids: list[str] = []
    for issue in issues:
        category = "conflict" if issue.kind == "context_conflict" else "validation"
        quality = Finding(
            kind=issue.kind, severity="warning", stage="reconciliation",
            audit_category=category, subject_refs=issue.subject_ids,
            evidence={"trigger": category, **issue.to_dict()},
            message=("Terminology context conflict requires inspection."
                     if category == "conflict" else
                     "Selected terminology target was not found in effective content."),
            dependency_ids=issue.base_artifact_ids,
        )
        store.put_finding(quality); findings.append(quality.finding_id)
        assessment_id, ids = _put_assessment(store, reconciliation_id=reconciliation_id, issue=issue, base_revision_id=base_revision_id)
        assessment_ids.append(assessment_id); findings.extend(ids)
    if status == "degraded":
        assessment = ConfidenceAssessment(subject_id="reconciliation", producing_stage="reconciliation", producing_artifact_id=reconciliation_id, score=None, signals=("degraded", "fallback"))
        uncertainty = actionable_uncertainty_finding(assessment) or uncertainty_finding(assessment)
        store.put_finding(uncertainty); findings.append(uncertainty.finding_id)
        bases = projection_ids or page_ids
        if bases:
            for request in review_requests_for(assessment=assessment, degraded_or_fallback=True, stage="reconciliation", subject_ids=("reconciliation",), suggested_correction_kind="target_segment", base_revision_id=base_revision_id, base_artifact_ids=bases, scope="segment"):
                store.put_finding(request); findings.append(request.finding_id)
        assessment_artifact = store.put(RECONCILIATION_ASSESSMENT_KIND, assessment.to_dict(), dependency_ids=(reconciliation_id,), finding_ids=tuple(sorted(set(findings))), semantic_key=f"confidence:{reconciliation_id}:fallback")
        assessment_ids.append(assessment_artifact.artifact_id)
    summary = stage_summary_finding("reconciliation", status, {"pages": len(page_ids), "projections": len(projection_ids), "issues": len(issues), "exceptions": int(status == "degraded")}, subject_refs=tuple(sorted({*page_ids, *projection_ids})))
    store.put_finding(summary); findings.append(summary.finding_id)
    envelope = store.put(RECONCILIATION_ARTIFACT_KIND, payload, dependency_ids=tuple(sorted(set((*page_ids, *projection_ids)))), finding_ids=tuple(sorted(set(findings))), semantic_key=hashlib.sha256(canonical_json(payload).encode()).hexdigest())
    if envelope.artifact_id != reconciliation_id:
        raise AssertionError("reconciliation artifact identity changed while publishing")
    return ReconciliationArtifact(envelope.artifact_id, page_ids, projection_ids, issues, tuple(sorted(set(findings))), tuple(sorted(assessment_ids)), summary.finding_id, status, error)
