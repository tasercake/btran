"""Informational validation for legacy pages and selected effective artifacts."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from btran.artifacts import ArtifactStore, artifact_id_for, validation_semantic_key
from btran.reconciliation import ReconciliationArtifact
from btran.schema import (
    ConfidenceAssessment,
    EffectivePage,
    EffectiveSegment,
    Finding,
    PageExtraction,
    PageResult,
    SourceBlock,
    TerminologyMap,
    TranslatedBlock,
    review_requests_for,
    stage_summary_finding,
    uncertainty_finding,
)

VALID_BLOCK_TYPES = frozenset({"heading", "paragraph", "caption", "footnote", "table", "list_item", "pull_quote", "illustration", "quote", "page_number"})
VALIDATION_STAGES = ("block_schema", "non_empty_text", "translation_language", "illustration_count", "block_id_correspondence", "glossary_consistency")
VALIDATION_ARTIFACT_KIND = "ValidationArtifact"
VALIDATION_ASSESSMENT_KIND = "ConfidenceAssessment"
_COMMON_WORDS = {
    "en": frozenset({"the", "and", "is", "are", "this", "that", "with", "for", "of", "to"}),
    "fr": frozenset({"le", "la", "les", "de", "des", "et", "bonjour", "merci", "avec", "pour"}),
    "es": frozenset({"el", "la", "los", "las", "de", "y", "hola", "gracias", "con", "para"}),
    "de": frozenset({"der", "die", "das", "und", "ist", "mit", "für", "ein", "eine", "hallo"}),
}


# Legacy pure validators remain migration readers/helpers. They do not gate a
# typed run; validate_effective below persists their observations as content.
def check_block_schema(blocks: Sequence[SourceBlock]) -> list[str]:
    errors: list[str] = []; seen_ids: set[str] = set(); seen_orders: set[int] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, SourceBlock): errors.append(f"block {index} is not a SourceBlock"); continue
        if not isinstance(block.id, str) or not block.id.strip(): errors.append(f"block {index} has an empty id")
        elif block.id in seen_ids: errors.append(f"duplicate id: {block.id}")
        else: seen_ids.add(block.id)
        if block.type not in VALID_BLOCK_TYPES: errors.append(f"block {block.id} has unsupported type: {block.type}")
        if not isinstance(block.reading_order, int) or isinstance(block.reading_order, bool) or block.reading_order < 0: errors.append(f"block {block.id} has invalid reading_order")
        elif block.reading_order in seen_orders: errors.append(f"duplicate reading_order: {block.reading_order}")
        else: seen_orders.add(block.reading_order)
    return errors


def check_translated_block_schema(blocks: Sequence[TranslatedBlock]) -> list[str]:
    return [f"translated block {index} is not a TranslatedBlock" if not isinstance(block, TranslatedBlock) else f"translated block {index} has an empty block_id" for index, block in enumerate(blocks) if not isinstance(block, TranslatedBlock) or not isinstance(block.block_id, str) or not block.block_id.strip()]


def _has_text(value: object) -> bool: return isinstance(value, str) and bool(value.strip())


def check_non_empty_text_fields(result: PageResult, source: PageExtraction | None = None) -> list[str]:
    errors = []
    if not _has_text(result.page_text): errors.append("page_text is empty")
    if not _has_text(result.translated_text): errors.append("translated_text is empty")
    for block in (source.blocks if source is not None else result.blocks):
        if isinstance(block, SourceBlock) and not _has_text(block.text): errors.append(f"source block {block.id} text is empty")
    for block in result.translated_blocks:
        if isinstance(block, TranslatedBlock) and not _has_text(block.translated_text): errors.append(f"translated block {block.block_id} text is empty")
    return errors


def detect_language(text: str) -> str | None:
    if not _has_text(text): return None
    if re.search(r"[\u3040-\u30ff]", text): return "ja"
    if re.search(r"[\uac00-\ud7af]", text): return "ko"
    if re.search(r"[\u4e00-\u9fff]", text): return "zh"
    words = re.findall(r"[a-zA-ZÀ-ÿ]+", text.casefold())
    language, score = max(((language, sum(word in vocabulary for word in words)) for language, vocabulary in _COMMON_WORDS.items()), key=lambda item: item[1])
    return language if score else None


def check_translation_language(result: PageResult) -> list[str]:
    detected = detect_language(result.translated_text)
    return [f"translated_text appears to be {detected}, expected {result.target_lang}"] if detected is not None and detected != result.target_lang.casefold() else []


def check_illustration_count(source: PageExtraction, result: PageResult) -> list[str]:
    if len(source.illustrations) != len(result.image_descriptions): return [f"expected {len(source.illustrations)} illustration descriptions, got {len(result.image_descriptions)}"]
    return [f"illustration description {index} is empty" for index, description in enumerate(result.image_descriptions) if not _has_text(description)]


def check_block_id_correspondence(source: PageExtraction, result: PageResult) -> list[str]:
    bad_source = [f"source block {index} is not a SourceBlock" for index, block in enumerate(source.blocks) if not isinstance(block, SourceBlock)]
    bad_target = [f"translated block {index} is not a TranslatedBlock" for index, block in enumerate(result.translated_blocks) if not isinstance(block, TranslatedBlock)]
    if bad_source or bad_target: return bad_source + bad_target
    source_ids = {block.id for block in source.blocks}; target_ids = [block.block_id for block in result.translated_blocks]; target_set = set(target_ids); errors = []
    missing, extra, duplicate = sorted(source_ids - target_set), sorted(target_set - source_ids), sorted({item for item in target_ids if target_ids.count(item) > 1})
    if missing: errors.append(f"missing translated block IDs: {', '.join(missing)}")
    if extra: errors.append(f"extra translated block IDs: {', '.join(extra)}")
    if duplicate: errors.append(f"duplicate translated block IDs: {', '.join(duplicate)}")
    return errors


def _contains_target_term(translation: str, target_term: str) -> bool:
    translation, target_term = translation.casefold(), target_term.casefold()
    return re.search(r"(?<!\w)" + re.escape(target_term) + r"(?!\w)", translation) is not None if re.search(r"[a-zà-ÿ]", target_term) else target_term in translation


def check_glossary_consistency(source: PageExtraction, result: PageResult, glossary: TerminologyMap) -> list[str]:
    targets: dict[str, set[str]] = {}
    for entry in glossary.entries:
        for source_term in entry.source_terms: targets.setdefault(source_term.casefold(), set()).add(entry.target_term)
    target_by_id = {block.block_id: block.translated_text for block in result.translated_blocks if isinstance(block, TranslatedBlock)}
    errors = []
    for mention in source.term_mentions:
        forms = targets.get(mention.term.casefold())
        if forms is not None and not any(_contains_target_term(target_by_id.get(mention.block_id, ""), form) for form in forms):
            errors.append(f"block {mention.block_id} translates glossary term '{mention.term}' without required target '{' or '.join(repr(item) for item in sorted(forms, key=str.casefold))}'")
    return errors


def validate_page(source: PageExtraction, result: PageResult, glossary: TerminologyMap) -> dict[str, list[str]]:
    return {"block_schema": check_block_schema(source.blocks) + check_translated_block_schema(result.translated_blocks), "non_empty_text": check_non_empty_text_fields(result, source), "translation_language": check_translation_language(result), "illustration_count": check_illustration_count(source, result), "block_id_correspondence": check_block_id_correspondence(source, result), "glossary_consistency": check_glossary_consistency(source, result, glossary)}


@dataclass(frozen=True)
class ValidationRuleResult:
    rule: str
    errors: tuple[str, ...]
    exception: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "errors": list(self.errors), "exception": dict(self.exception) if self.exception else None}


@dataclass(frozen=True)
class ValidationArtifact:
    artifact_id: str
    reconciliation_artifact_id: str
    effective_page_artifact_ids: tuple[str, ...]
    rule_results: tuple[ValidationRuleResult, ...]
    finding_ids: tuple[str, ...]
    assessment_artifact_ids: tuple[str, ...]
    stage_summary_finding_id: str
    status: str
    error_evidence: Mapping[str, Any] | None = None


def _selected_effective_pages(value: Any, store: ArtifactStore) -> tuple[tuple[str, EffectivePage, tuple[tuple[str, EffectiveSegment], ...]], ...]:
    leaves = getattr(value, "leaves", value)
    if not isinstance(leaves, (tuple, list)): raise ValueError("effective pages must be Task-10 leaves or selected page artifact IDs")
    output = []
    for item in leaves:
        artifact_id = item.page_artifact_id if hasattr(item, "page_artifact_id") else item
        envelope = store.get(artifact_id)
        if envelope.kind != "EffectiveTargetPage": raise ValueError("validation requires selected effective target pages")
        page = EffectivePage.from_dict(envelope.payload); segments = []
        for child_id in envelope.dependency_ids:
            child = store.get(child_id)
            if child.kind in {"EffectiveTargetSegment", "DiagnosticEffectiveTargetSegment"}:
                segments.append((child.artifact_id, EffectiveSegment.from_dict(child.payload)))
        if tuple(item[1].effective_segment_id for item in segments) != page.effective_segment_ids: raise ValueError("effective page segment closure/order is invalid")
        output.append((envelope.artifact_id, page, tuple(segments)))
    return tuple(output)


def _rule_structure(pages: tuple[tuple[str, EffectivePage, tuple[tuple[str, EffectiveSegment], ...]], ...], _: ReconciliationArtifact, __: str) -> tuple[str, ...]:
    errors = []
    seen_pages, seen_segments = set(), set()
    for _, page, segments in pages:
        if page.page_id in seen_pages: errors.append(f"duplicate page ID: {page.page_id}")
        seen_pages.add(page.page_id)
        if not segments: errors.append(f"page {page.page_id} has no effective segments")
        for _, segment in segments:
            if segment.segment_id in seen_segments: errors.append(f"duplicate segment ID: {segment.segment_id}")
            seen_segments.add(segment.segment_id)
    return tuple(errors)


def _rule_non_empty(pages: tuple[tuple[str, EffectivePage, tuple[tuple[str, EffectiveSegment], ...]], ...], _: ReconciliationArtifact, __: str) -> tuple[str, ...]:
    return tuple(f"effective segment {segment.segment_id} text is empty" for _, _, segments in pages for _, segment in segments if not _has_text(segment.effective_text))


def _rule_language(pages: tuple[tuple[str, EffectivePage, tuple[tuple[str, EffectiveSegment], ...]], ...], _: ReconciliationArtifact, mode: str) -> tuple[str, ...]:
    errors = []
    for _, _, segments in pages:
        for _, segment in segments:
            expected = segment.source_lang if mode == "native" else segment.render_lang
            detected = detect_language(segment.effective_text)
            if expected and expected != "und" and detected is not None and detected != expected.casefold(): errors.append(f"effective segment {segment.segment_id} appears to be {detected}, expected {expected}")
    return tuple(errors)


def _rule_reconciliation(_: tuple[tuple[str, EffectivePage, tuple[tuple[str, EffectiveSegment], ...]], ...], reconciliation: ReconciliationArtifact, mode: str) -> tuple[str, ...]:
    if mode == "native": return ()
    return tuple(f"reconciliation {issue.kind}: {', '.join(issue.subject_ids)}" for issue in reconciliation.issues)


def _rules(mode: str) -> dict[str, Callable[[Any, ReconciliationArtifact, str], Sequence[str]]]:
    values = {"effective_structure": _rule_structure, "non_empty_text": _rule_non_empty, "source_language" if mode == "native" else "translation_language": _rule_language}
    if mode != "native": values["reconciliation_terms"] = _rule_reconciliation
    return values


def _validation_payload(reconciliation_id: str, page_ids: Sequence[str], results: Sequence[ValidationRuleResult], status: str, error: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"reconciliation_artifact_id": reconciliation_id, "effective_page_artifact_ids": list(sorted(set(page_ids))), "rule_results": [result.to_dict() for result in results], "status": status, "error_evidence": dict(error) if error else None, "rules_version": "validation-v1"}


def _put_validation_assessment(store: ArtifactStore, *, validation_id: str, subject: str, review_subject_ids: Sequence[str], base_ids: Sequence[str], base_revision_id: str, signal: str, validation_error: bool) -> tuple[str, tuple[str, ...]]:
    degraded = not validation_error
    signals = tuple(sorted({"fallback", signal})) if degraded else (signal,)
    assessment = ConfidenceAssessment(subject_id=subject, producing_stage="validation", producing_artifact_id=validation_id, score=None if degraded else 0, signals=signals)
    uncertainty = uncertainty_finding(assessment); store.put_finding(uncertainty)
    requests = review_requests_for(assessment=assessment, degraded_or_fallback=degraded, validation_error=validation_error, stage="validation", subject_ids=tuple(sorted(set(review_subject_ids))), suggested_correction_kind="target_segment", base_revision_id=base_revision_id, base_artifact_ids=tuple(sorted(set(base_ids))), scope="segment")
    finding_ids = [uncertainty.finding_id]
    for finding in requests: store.put_finding(finding); finding_ids.append(finding.finding_id)
    artifact = store.put(VALIDATION_ASSESSMENT_KIND, assessment.to_dict(), dependency_ids=(validation_id,), finding_ids=tuple(sorted(finding_ids)), semantic_key=f"confidence:{validation_id}:{subject}")
    return artifact.artifact_id, tuple(sorted(finding_ids))


def validate_effective(*, effective_pages: Any, reconciliation: ReconciliationArtifact, store: ArtifactStore, base_revision_id: str, mode: str, rules: Mapping[str, Callable[[Any, ReconciliationArtifact, str], Sequence[str]]] | None = None) -> ValidationArtifact:
    """Run independently-contained rules over selected effective content.

    ``reconciliation`` is a required typed semantic input. All exceptions turn
    into persisted diagnostics and a degraded artifact; no exception gates render.
    """
    page_ids: tuple[str, ...] = (); results: list[ValidationRuleResult] = []; findings: list[str] = []; assessment_ids: list[str] = []; error: Mapping[str, Any] | None = None; status = "completed"
    try:
        if not isinstance(reconciliation, ReconciliationArtifact): raise ValueError("validation requires explicit ReconciliationArtifact")
        if mode not in {"native", "translated"}: raise ValueError("validation mode is invalid")
        pages = _selected_effective_pages(effective_pages, store)
        page_ids = tuple(sorted(item[0] for item in pages))
        active_rules = dict(_rules(mode) if rules is None else rules)
        if not active_rules or any(not isinstance(name, str) or not name or not callable(rule) for name, rule in active_rules.items()): raise ValueError("validation rules are invalid")
        for name in sorted(active_rules):
            try:
                errors = tuple(str(item) for item in active_rules[name](pages, reconciliation, mode))
                results.append(ValidationRuleResult(name, errors))
            except Exception as exc:
                status = "degraded"; detail = {"exception_type": type(exc).__name__, "message": str(exc)}
                results.append(ValidationRuleResult(name, (), detail))
                finding = Finding(kind="validator_exception", severity="error", stage="validation", subject_refs=(name,), evidence={"rule": name, **detail}, message="Validator rule failed; remaining rules continued.", dependency_ids=(reconciliation.artifact_id,))
                store.put_finding(finding); findings.append(finding.finding_id)
    except Exception as exc:
        status = "degraded"; error = {"exception_type": type(exc).__name__, "message": str(exc)}
        reconciliation_id = reconciliation.artifact_id if isinstance(reconciliation, ReconciliationArtifact) else "unavailable-reconciliation-artifact"
        finding = Finding(kind="validation_exception", severity="error", stage="validation", subject_refs=(reconciliation_id,), evidence=dict(error), message="Validation setup/aggregation failed; diagnostic content remains renderable.", dependency_ids=())
        store.put_finding(finding); findings.append(finding.finding_id)
    reconciliation_id = reconciliation.artifact_id if isinstance(reconciliation, ReconciliationArtifact) else "unavailable-reconciliation-artifact"
    payload = _validation_payload(reconciliation_id, page_ids, results, status, error)
    dependencies = tuple(sorted(set((*page_ids, reconciliation_id)))) if reconciliation_id != "unavailable-reconciliation-artifact" else page_ids
    validation_id = artifact_id_for(VALIDATION_ARTIFACT_KIND, payload, dependencies)
    review_subject_ids = tuple(sorted({segment.segment_id for _, _, segments in locals().get("pages", ()) for _, segment in segments}))
    for result in results:
        if result.errors or result.exception:
            subjects = review_subject_ids or (result.rule,)
            finding = Finding(kind="validation_error" if result.errors else "validator_exception", severity="warning" if result.errors else "error", stage="validation", subject_refs=subjects, evidence=result.to_dict(), message="Validation observation retained as informational content.", dependency_ids=(reconciliation_id,) if reconciliation_id != "unavailable-reconciliation-artifact" else ())
            store.put_finding(finding); findings.append(finding.finding_id)
            assessment_id, ids = _put_validation_assessment(store, validation_id=validation_id, subject=result.rule, review_subject_ids=subjects, base_ids=page_ids or dependencies, base_revision_id=base_revision_id, signal="validation_error" if result.errors else "validator_exception", validation_error=bool(result.errors))
            assessment_ids.append(assessment_id); findings.extend(ids)
    if error is not None:
        bases = page_ids or dependencies
        if bases:
            assessment_id, ids = _put_validation_assessment(store, validation_id=validation_id, subject="validation", review_subject_ids=("validation",), base_ids=bases, base_revision_id=base_revision_id, signal="degraded", validation_error=False)
            assessment_ids.append(assessment_id); findings.extend(ids)
    summary = stage_summary_finding("validation", status, {"pages": len(page_ids), "rules": len(results), "errors": sum(bool(item.errors) for item in results), "rule_exceptions": sum(item.exception is not None for item in results), "setup_exceptions": int(error is not None)}, subject_refs=tuple(sorted({*page_ids, reconciliation_id})))
    store.put_finding(summary); findings.append(summary.finding_id)
    # Reuse must key on selected reconciliation artifact, never only on page
    # content.  Projection IDs are retained in that selected reconciliation
    # closure and deliberately participate again as declared validation input.
    target_langs = tuple(sorted({segment.render_lang for _, _, segments in locals().get("pages", ())
                                for _, segment in segments if segment.render_lang}))
    target_lang = None if mode == "native" else (target_langs[0] if len(target_langs) == 1 else None)
    semantic_key = validation_semantic_key(
        rules_version="validation-v1", mode=mode, target_lang=target_lang,
        effective_page_ids=page_ids, reconciliation_artifact_id=reconciliation_id,
        projection_ids=reconciliation.projection_artifact_ids if isinstance(reconciliation, ReconciliationArtifact) else (),
    )
    envelope = store.put(VALIDATION_ARTIFACT_KIND, payload, dependency_ids=dependencies,
                         finding_ids=tuple(sorted(set(findings))), semantic_key=semantic_key)
    if envelope.artifact_id != validation_id: raise AssertionError("validation artifact identity changed while publishing")
    return ValidationArtifact(envelope.artifact_id, reconciliation_id, page_ids, tuple(results), tuple(sorted(set(findings))), tuple(sorted(assessment_ids)), summary.finding_id, status, error)
