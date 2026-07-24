"""Deterministic terminology consolidation, sharding, and page glossary slices."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from btran.schema import SourceBlock, TermMention, TerminologyEntry, TerminologyMap

DEFAULT_TOKEN_BUDGET = 100_000
MAX_TOKEN_BUDGET = 120_000
HARD_TOKEN_CAP = 200_000


@dataclass(frozen=True)
class TermGroup:
    """All spellings and block IDs for one normalized source-term mention."""

    normalized_term: str
    forms: tuple[str, ...]
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class TermBatch:
    """A token-bounded, deterministic collection of term groups."""

    groups: tuple[TermGroup, ...]
    token_count: int


@dataclass(frozen=True)
class TerminologyShards:
    """Stable glossary shards plus normalized source alias lookup."""

    shards: tuple[TerminologyMap, ...]
    alias_index: dict[str, tuple[str, ...]]


def normalize_term(term: str) -> str:
    """Normalize a mention for deterministic matching while retaining its form."""
    return " ".join(unicodedata.normalize("NFKC", term).split()).casefold()


def estimate_tokens(text: str) -> int:
    """A deliberately simple, deterministic token estimate (about four chars/token)."""
    return (len(text) + 3) // 4


def _validate_budget(token_budget: int) -> None:
    if not 1 <= token_budget <= MAX_TOKEN_BUDGET:
        raise ValueError(f"token_budget must be between 1 and {MAX_TOKEN_BUDGET}")


def group_term_mentions(mentions: Iterable[TermMention]) -> list[TermGroup]:
    """Group equivalent source spellings and retain every originating block ID."""
    grouped: dict[str, tuple[set[str], set[str]]] = {}
    for mention in mentions:
        normalized = normalize_term(mention.term)
        if not normalized:
            continue
        forms, provenance = grouped.setdefault(normalized, (set(), set()))
        form = " ".join(unicodedata.normalize("NFKC", mention.term).split())
        if form:
            forms.add(form)
        provenance.add(mention.block_id)
    return [
        TermGroup(
            normalized_term=normalized,
            forms=tuple(sorted(forms, key=lambda value: (normalize_term(value), value))),
            provenance=tuple(sorted(provenance)),
        )
        for normalized, (forms, provenance) in sorted(grouped.items())
    ]


def _group_token_count(group: TermGroup) -> int:
    return estimate_tokens("\n".join((*group.forms, *group.provenance)))


def batch_term_groups(
    groups: Sequence[TermGroup], token_budget: int = DEFAULT_TOKEN_BUDGET
) -> list[TermBatch]:
    """Pack ordered groups without exceeding the requested budget.

    A single indivisible group may exceed the soft budget, but never the hard
    request cap.  This makes oversized aliases explicit instead of silently
    truncating terminology.
    """
    _validate_budget(token_budget)
    batches: list[TermBatch] = []
    current: list[TermGroup] = []
    current_tokens = 0
    for group in groups:
        group_tokens = _group_token_count(group)
        if group_tokens > HARD_TOKEN_CAP:
            raise ValueError("one term group exceeds the 200000 token hard cap")
        if current and current_tokens + group_tokens > token_budget:
            batches.append(TermBatch(tuple(current), current_tokens))
            current, current_tokens = [], 0
        current.append(group)
        current_tokens += group_tokens
        if group_tokens > token_budget:
            batches.append(TermBatch(tuple(current), current_tokens))
            current, current_tokens = [], 0
    if current:
        batches.append(TermBatch(tuple(current), current_tokens))
    return batches


def _request_items(groups: Sequence[TermGroup]) -> list[dict[str, object]]:
    return [
        {
            "source_terms": list(group.forms),
            "provenance": list(group.provenance),
        }
        for group in groups
    ]


def _consolidation_prompt(groups: Sequence[TermGroup]) -> str:
    payload = {"items": _request_items(groups)}
    return (
        "Consolidate these source terminology candidates. Return JSON only as "
        '{"entries": [{"concept_id": str, "source_terms": [str], '
        '"target_term": str, "provenance": [str], "confidence": number, '
        '"notes": str}]}. Preserve aliases and provenance.\n'
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _parse_entries(response: str) -> list[TerminologyEntry]:
    try:
        payload = json.loads(response)
        raw_entries = payload["entries"] if isinstance(payload, dict) else payload
        if not isinstance(raw_entries, list):
            raise TypeError("entries is not a list")
        entries = [TerminologyEntry.from_dict(dict(item)) for item in raw_entries]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Pi consolidation response must be JSON terminology entries") from exc
    for entry in entries:
        if not entry.concept_id or not entry.target_term or not entry.source_terms:
            raise ValueError("Pi consolidation response contains an incomplete entry")
    return entries


def _merge_same_target_entries(entries: Iterable[TerminologyEntry]) -> list[TerminologyEntry]:
    """Coalesce batch-local aliases when Pi assigns them the same target term."""
    merged: dict[str, list[TerminologyEntry]] = {}
    for entry in entries:
        merged.setdefault(normalize_term(entry.target_term), []).append(entry)
    result: list[TerminologyEntry] = []
    for target, candidates in sorted(merged.items()):
        if len(candidates) == 1:
            result.extend(candidates)
            continue
        ordered = _canonical_entries(candidates)
        result.append(
            TerminologyEntry(
                concept_id=ordered[0].concept_id,
                source_terms=sorted(
                    {term for entry in ordered for term in entry.source_terms},
                    key=lambda value: (normalize_term(value), value),
                ),
                target_term=ordered[0].target_term,
                provenance=sorted({block for entry in ordered for block in entry.provenance}),
                confidence=min(entry.confidence for entry in ordered),
                notes="; ".join(sorted({entry.notes for entry in ordered if entry.notes})),
            )
        )
    return _canonical_entries(result)


def _groups_from_entries(entries: Iterable[TerminologyEntry]) -> list[TermGroup]:
    return [
        TermGroup(
            normalized_term=entry.concept_id,
            forms=tuple(entry.source_terms),
            provenance=tuple(entry.provenance),
        )
        for entry in _canonical_entries(entries)
    ]


def _canonical_entries(entries: Iterable[TerminologyEntry]) -> list[TerminologyEntry]:
    canonical: list[TerminologyEntry] = []
    for entry in entries:
        canonical.append(
            TerminologyEntry(
                concept_id=entry.concept_id,
                source_terms=sorted(set(entry.source_terms), key=lambda value: (normalize_term(value), value)),
                target_term=entry.target_term,
                provenance=sorted(set(entry.provenance)),
                confidence=entry.confidence,
                notes=entry.notes,
            )
        )
    return sorted(canonical, key=lambda entry: (entry.concept_id, entry.target_term))


def consolidate_terminology(
    mentions: Iterable[TermMention],
    *,
    source_lang: str,
    target_lang: str,
    pi_call: Callable[[str], str],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_rounds: int = 8,
    version: str = "1",
) -> TerminologyMap:
    """Consolidate mentions with bounded, text-only Pi calls.

    Each round converts its token-bounded batches to entries, then feeds those
    entries into the next round until they fit in one batch. ``max_rounds`` is
    a hard stop for a model that fails to reduce its output.
    """
    _validate_budget(token_budget)
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    groups = group_term_mentions(mentions)
    if not groups:
        return freeze_terminology([], source_lang=source_lang, target_lang=target_lang, version=version)

    for round_number in range(max_rounds):
        batches = batch_term_groups(groups, token_budget)
        results: list[TerminologyEntry] = []
        for batch in batches:
            prompt = _consolidation_prompt(batch.groups)
            if estimate_tokens(prompt) > HARD_TOKEN_CAP:
                raise ValueError("consolidation request exceeds the 200000 token hard cap")
            results.extend(_parse_entries(pi_call(prompt)))
        results = _merge_same_target_entries(results)
        if len(batches) == 1:
            return freeze_terminology(
                results, source_lang=source_lang, target_lang=target_lang, version=version
            )
        groups = _groups_from_entries(results)
    raise ValueError(f"consolidation did not converge within {max_rounds} rounds")


def _hash_payload(
    entries: Iterable[TerminologyEntry], source_lang: str, target_lang: str, version: str
) -> str:
    payload = {
        "version": version,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "entries": [entry.to_dict() for entry in _canonical_entries(entries)],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def freeze_terminology(
    entries: Iterable[TerminologyEntry],
    *,
    source_lang: str,
    target_lang: str,
    version: str = "1",
) -> TerminologyMap:
    """Return an ordered glossary with a hash independent of input ordering/time."""
    canonical = _canonical_entries(entries)
    return TerminologyMap(
        version=version,
        hash=_hash_payload(canonical, source_lang, target_lang, version),
        source_lang=source_lang,
        target_lang=target_lang,
        entries=canonical,
    )


def _entry_token_count(entry: TerminologyEntry) -> int:
    return estimate_tokens(
        "\n".join(
            (entry.concept_id, *entry.source_terms, entry.target_term, *entry.provenance, entry.notes)
        )
    )


def shard_terminology_map(
    glossary: TerminologyMap, token_budget: int = DEFAULT_TOKEN_BUDGET
) -> TerminologyShards:
    """Split a large map stably and index every normalized source alias."""
    _validate_budget(token_budget)
    shards: list[list[TerminologyEntry]] = []
    current: list[TerminologyEntry] = []
    current_tokens = 0
    for entry in _canonical_entries(glossary.entries):
        entry_tokens = _entry_token_count(entry)
        if entry_tokens > HARD_TOKEN_CAP:
            raise ValueError("one terminology entry exceeds the 200000 token hard cap")
        if current and current_tokens + entry_tokens > token_budget:
            shards.append(current)
            current, current_tokens = [], 0
        current.append(entry)
        current_tokens += entry_tokens
        if entry_tokens > token_budget:
            shards.append(current)
            current, current_tokens = [], 0
    if current or not shards:
        shards.append(current)

    alias_index: dict[str, list[str]] = {}
    for entry in _canonical_entries(glossary.entries):
        for alias in entry.source_terms:
            key = normalize_term(alias)
            if key:
                alias_index.setdefault(key, []).append(entry.concept_id)
    return TerminologyShards(
        shards=tuple(
            freeze_terminology(
                shard,
                source_lang=glossary.source_lang,
                target_lang=glossary.target_lang,
                version=glossary.version,
            )
            for shard in shards
        ),
        alias_index={key: tuple(sorted(set(value))) for key, value in sorted(alias_index.items())},
    )


def _text_from_blocks(blocks: Iterable[SourceBlock | str]) -> str:
    return "\n".join(block.text if isinstance(block, SourceBlock) else str(block) for block in blocks)


def _alias_in_text(alias: str, text: str) -> bool:
    normalized_alias = normalize_term(alias)
    normalized_text = normalize_term(text)
    if not normalized_alias:
        return False
    # Word boundaries prevent aliases such as "art" matching "party".  Terms
    # without word characters (or CJK text) still use straightforward matching.
    if re.search(r"\w", normalized_alias):
        return re.search(r"(?<!\w)" + re.escape(normalized_alias) + r"(?!\w)", normalized_text) is not None
    return normalized_alias in normalized_text


def slice_for_page(
    glossary: TerminologyMap | TerminologyShards,
    blocks: Iterable[SourceBlock | str],
    *,
    previous_boundary: str = "",
    next_boundary: str = "",
) -> list[TerminologyEntry]:
    """Choose glossary entries whose source terms occur on or beside a page."""
    entries = (
        glossary.entries
        if isinstance(glossary, TerminologyMap)
        else [entry for shard in glossary.shards for entry in shard.entries]
    )
    text = "\n".join((_text_from_blocks(blocks), previous_boundary, next_boundary))
    return [
        entry
        for entry in _canonical_entries(entries)
        if any(_alias_in_text(alias, text) for alias in entry.source_terms)
    ]


def cache_identity_with_glossary(base_identity: str, glossary: TerminologyMap | str) -> str:
    """Bind a translation cache identity to a frozen glossary version/hash."""
    glossary_hash = glossary.hash if isinstance(glossary, TerminologyMap) else glossary
    return hashlib.sha256(f"{base_identity}\0{glossary_hash}".encode("utf-8")).hexdigest()


# Clear aliases for callers that use the nouns from the workflow description.
select_glossary_slice = slice_for_page
glossary_cache_identity = cache_identity_with_glossary
