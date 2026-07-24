"""One-round reconciliation of glossary usage across translated pages."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Callable

from btran.schema import PageExtraction, TerminologyEntry, TerminologyMap, TranslatedBlock


@dataclass(frozen=True)
class GlossaryChange:
    concept_id: str
    old_target_term: str
    new_target_term: str


@dataclass(frozen=True)
class TermIssue:
    concept_id: str
    kind: str
    pages: tuple[int, ...]
    expected_target_term: str
    observed_target_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReconciliationResult:
    glossary_v2: TerminologyMap
    glossary_diff: list[GlossaryChange]
    issues: list[TermIssue]
    affected_pages: list[int]


def index_terms_to_pages(
    extractions: list[PageExtraction], glossary: TerminologyMap
) -> dict[str, set[int]]:
    """Map each glossary concept to pages where a source synonym occurs."""
    index: dict[str, set[int]] = {}
    for extraction in extractions:
        mentioned = {mention.term.casefold() for mention in extraction.term_mentions}
        source_text = "\n".join(block.text for block in extraction.blocks).casefold()
        for entry in glossary.entries:
            if any(term.casefold() in mentioned or term.casefold() in source_text for term in entry.source_terms):
                index.setdefault(entry.concept_id, set()).add(extraction.page_number)
    return index


def glossary_diff(v1: TerminologyMap, v2: TerminologyMap) -> list[GlossaryChange]:
    """Return target-form changes between two frozen glossary versions."""
    old = {entry.concept_id: entry for entry in v1.entries}
    new = {entry.concept_id: entry for entry in v2.entries}
    return [
        GlossaryChange(concept_id, old[concept_id].target_term, new[concept_id].target_term)
        for concept_id in sorted(old.keys() & new.keys())
        if old[concept_id].target_term != new[concept_id].target_term
    ]


def _contains_target(text: str, target: str) -> bool:
    if not target:
        return False
    if target.isascii() and target.replace(" ", "").isalnum():
        return re.search(rf"(?<!\w){re.escape(target)}(?!\w)", text, flags=re.IGNORECASE) is not None
    return target.casefold() in text.casefold()


def _translated_page_text(blocks: list[TranslatedBlock] | None) -> str:
    return "\n".join(block.translated_text for block in blocks or [])


def _next_version(version: str) -> str:
    if version.isdigit():
        return str(int(version) + 1)
    parts = version.split(".")
    if parts and parts[0].isdigit():
        return ".".join([str(int(parts[0]) + 1), *(["0"] * (len(parts) - 1))])
    return f"{version}-v2"


def _glossary_hash(version: str, entries: list[TerminologyEntry], glossary: TerminologyMap) -> str:
    payload = {
        "version": version,
        "source_lang": glossary.source_lang,
        "target_lang": glossary.target_lang,
        "entries": [entry.to_dict() for entry in entries],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _with_reviewed_forms(
    glossary: TerminologyMap, reviewed_forms: dict[str, str]
) -> TerminologyMap:
    entries = [
        replace(entry, target_term=reviewed_forms.get(entry.concept_id, entry.target_term))
        for entry in glossary.entries
    ]
    version = _next_version(glossary.version)
    return TerminologyMap(
        version=version,
        hash=_glossary_hash(version, entries, glossary),
        source_lang=glossary.source_lang,
        target_lang=glossary.target_lang,
        entries=entries,
    )


def reconcile(
    *,
    glossary: TerminologyMap,
    extractions: list[PageExtraction],
    translations: dict[int, list[TranslatedBlock]],
    reviewer: Callable[[list[TermIssue]], dict[str, str]] | None = None,
) -> ReconciliationResult:
    """Inspect one translation pass and optionally review only glossary conflicts.

    A missing frozen form is an unambiguous page defect.  A target form that
    belongs to another glossary concept is a context conflict and is the only
    condition passed to the optional LLM-backed reviewer.  The result performs
    exactly one glossary update round; callers schedule any retranslation.
    """
    page_index = index_terms_to_pages(extractions, glossary)
    known_targets = {entry.target_term: entry.concept_id for entry in glossary.entries}
    issues: list[TermIssue] = []
    ambiguous: list[TermIssue] = []
    affected: set[int] = set()

    for entry in glossary.entries:
        pages = sorted(page_index.get(entry.concept_id, set()))
        missing_pages: list[int] = []
        conflict_pages: list[int] = []
        observed: set[str] = set()
        for page_number in pages:
            translated = _translated_page_text(translations.get(page_number))
            if _contains_target(translated, entry.target_term):
                continue
            competing = [
                target for target, concept_id in known_targets.items()
                if concept_id != entry.concept_id and _contains_target(translated, target)
            ]
            if competing:
                conflict_pages.append(page_number)
                observed.update(competing)
            else:
                missing_pages.append(page_number)

        if missing_pages:
            issue = TermIssue(
                entry.concept_id, "missing_term", tuple(missing_pages), entry.target_term
            )
            issues.append(issue)
            affected.update(missing_pages)
        if conflict_pages:
            issue = TermIssue(
                entry.concept_id,
                "context_conflict",
                tuple(conflict_pages),
                entry.target_term,
                tuple(sorted(observed)),
            )
            issues.append(issue)
            ambiguous.append(issue)
            affected.update(conflict_pages)

    reviewed_forms = reviewer(ambiguous) if reviewer is not None and ambiguous else {}
    glossary_v2 = _with_reviewed_forms(glossary, reviewed_forms)
    changes = glossary_diff(glossary, glossary_v2)
    for change in changes:
        affected.update(page_index.get(change.concept_id, set()))

    return ReconciliationResult(
        glossary_v2=glossary_v2,
        glossary_diff=changes,
        issues=issues,
        affected_pages=sorted(affected),
    )
