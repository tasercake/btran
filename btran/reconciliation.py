"""One-round reconciliation of glossary usage across translated pages."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
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
        alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
        return re.search(rf"(?<!\w)(?:{alternatives})(?!\w)", normalized_text) is not None
    return normalized_term in normalized_text


def _entry_mentions_page(entry: TerminologyEntry, extraction: PageExtraction) -> bool:
    mentioned = {_normalize_text(mention.term) for mention in extraction.term_mentions}
    source_text = "\n".join(block.text for block in extraction.blocks)
    return any(
        _normalize_text(term) in mentioned or _term_in_text(term, source_text)
        for term in entry.source_terms
    )


def index_terms_to_pages(
    extractions: list[PageExtraction], glossary: TerminologyMap
) -> dict[str, set[int]]:
    """Map every source alias/sense to pages where it is actually mentioned."""
    index: dict[str, set[int]] = {}
    for extraction in extractions:
        for entry in glossary.entries:
            if _entry_mentions_page(entry, extraction):
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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


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


def _ambiguous_source_pages(
    extractions: list[PageExtraction], glossary: TerminologyMap
) -> list[TermIssue]:
    """Identify aliases shared by concepts; their sense cannot be inferred by string matching."""
    aliases: dict[str, set[str]] = {}
    for entry in glossary.entries:
        for alias in entry.source_terms:
            normalized = _normalize_text(alias)
            if normalized:
                aliases.setdefault(normalized, set()).add(entry.concept_id)

    issues: list[TermIssue] = []
    for extraction in extractions:
        mentioned = {_normalize_text(mention.term) for mention in extraction.term_mentions}
        source_text = "\n".join(block.text for block in extraction.blocks)
        for alias, concept_ids in aliases.items():
            if len(concept_ids) < 2:
                continue
            if alias in mentioned or _term_in_text(alias, source_text):
                issues.append(
                    TermIssue(
                        concept_id="|".join(sorted(concept_ids)),
                        kind="ambiguous_source_sense",
                        pages=(extraction.page_number,),
                        expected_target_term="",
                    )
                )
    return issues


def _reviewed_form_changes(
    glossary: TerminologyMap,
    ambiguous: list[TermIssue],
    reviewer: Callable[[list[TermIssue]], dict[str, str]] | None,
) -> dict[str, str]:
    if reviewer is None or not ambiguous:
        return {}
    reviewed = reviewer(ambiguous)
    if not isinstance(reviewed, dict):
        raise ValueError("reviewer must return a mapping of concept IDs to target forms")
    allowed = {issue.concept_id for issue in ambiguous} & {entry.concept_id for entry in glossary.entries}
    unknown = set(reviewed) - allowed
    if unknown or any(not isinstance(form, str) or not form.strip() for form in reviewed.values()):
        raise ValueError("reviewer returned invalid glossary changes")
    old_forms = {entry.concept_id: entry.target_term for entry in glossary.entries}
    return {concept_id: form for concept_id, form in reviewed.items() if form != old_forms[concept_id]}


def reconcile(
    *,
    glossary: TerminologyMap,
    extractions: list[PageExtraction],
    translations: dict[int, list[TranslatedBlock]],
    reviewer: Callable[[list[TermIssue]], dict[str, str]] | None = None,
) -> ReconciliationResult:
    """Inspect one translation pass; return a page set but never schedule retranslation.

    Only a context conflict or an ambiguous source sense reaches ``reviewer``. A
    reviewer can update target forms once; no subsequent review/retranslation
    round is invoked here.
    """
    page_index = index_terms_to_pages(extractions, glossary)
    source_ambiguities = _ambiguous_source_pages(extractions, glossary)
    ambiguous_pages = {
        page_number
        for issue in source_ambiguities
        for page_number in issue.pages
    }
    known_targets = [(entry.target_term, entry.concept_id) for entry in glossary.entries]
    issues: list[TermIssue] = list(source_ambiguities)
    ambiguous: list[TermIssue] = list(source_ambiguities)
    affected: set[int] = set(ambiguous_pages)

    for entry in glossary.entries:
        pages = sorted(page_index.get(entry.concept_id, set()))
        missing_pages: list[int] = []
        conflict_pages: list[int] = []
        observed: set[str] = set()
        for page_number in pages:
            if page_number in ambiguous_pages:
                continue
            translated = _translated_page_text(translations.get(page_number))
            if _term_in_text(entry.target_term, translated, allow_regular_variant=True):
                continue
            competing = [
                target for target, concept_id in known_targets
                if concept_id != entry.concept_id
                and _term_in_text(target, translated, allow_regular_variant=True)
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

    reviewed_forms = _reviewed_form_changes(glossary, ambiguous, reviewer)
    glossary_v2 = _with_reviewed_forms(glossary, reviewed_forms) if reviewed_forms else glossary
    changes = glossary_diff(glossary, glossary_v2)
    for change in changes:
        affected.update(page_index.get(change.concept_id, set()))

    return ReconciliationResult(
        glossary_v2=glossary_v2,
        glossary_diff=changes,
        issues=issues,
        affected_pages=sorted(affected),
    )
