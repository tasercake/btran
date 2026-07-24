"""Deterministic terminology consolidation, sharding, and page glossary slices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
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


class PiConsolidationError(ValueError):
    """A bounded, text-only Pi consolidation call failed."""


def normalize_term(term: str) -> str:
    """Normalize a mention for deterministic matching while retaining its form."""
    return " ".join(unicodedata.normalize("NFKC", term).split()).casefold()


def estimate_tokens(text: str) -> int:
    """Return a deterministic conservative bound for UTF-8 tokenizer input.

    A model tokenizer can split text differently by model and language, but it
    cannot need more byte pieces than the UTF-8 input has bytes.  Counting bytes
    avoids under-measuring short, CJK, and punctuation-heavy terminology.
    """
    return len(text.encode("utf-8"))


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
        # The normalized key is only for matching.  Keep the submitted spelling
        # verbatim in artifacts and prompts so downstream review remains auditable.
        forms.add(mention.term)
        provenance.add(mention.block_id)
    return [
        TermGroup(
            normalized_term=normalized,
            forms=tuple(sorted(forms, key=lambda value: (normalize_term(value), value))),
            provenance=tuple(sorted(provenance)),
        )
        for normalized, (forms, provenance) in sorted(grouped.items())
    ]


def batch_term_groups(
    groups: Sequence[TermGroup], token_budget: int = DEFAULT_TOKEN_BUDGET
) -> list[TermBatch]:
    """Pack complete consolidation requests within the configured input budget.

    A group is indivisible because its forms and provenance are audit data.  We
    reject, rather than truncate, a group that cannot fit the configured input
    limit.  The count includes the JSON envelope and instruction text actually
    sent to Pi, not merely the source terms.
    """
    _validate_budget(token_budget)
    batches: list[TermBatch] = []
    current: list[TermGroup] = []
    for group in groups:
        single_tokens = _request_token_count((group,))
        if single_tokens > HARD_TOKEN_CAP:
            raise ValueError("one term group exceeds the 200000 token hard cap")
        if single_tokens > token_budget:
            raise ValueError("one term group exceeds the configured token budget")

        candidate = (*current, group)
        candidate_tokens = _request_token_count(candidate)
        if current and candidate_tokens > token_budget:
            batches.append(TermBatch(tuple(current), _request_token_count(current)))
            current = [group]
        else:
            current.append(group)
    if current:
        batches.append(TermBatch(tuple(current), _request_token_count(current)))
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
        '"notes": str}]}. Preserve every supplied source spelling and block ID. '
        "Do not invent aliases. Keep distinct senses, context variants, conflicts, and "
        "low-confidence candidates as separate entries.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _request_token_count(groups: Sequence[TermGroup]) -> int:
    return estimate_tokens(_consolidation_prompt(groups))


def _parse_entries(response: str) -> list[TerminologyEntry]:
    """Parse only the exact, typed JSON response expected from the Pi worker."""
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Pi consolidation response must be JSON terminology entries") from exc
    if not isinstance(payload, dict) or set(payload) != {"entries"} or not isinstance(payload["entries"], list):
        raise ValueError("Pi consolidation response must contain only an entries list")

    entries: list[TerminologyEntry] = []
    allowed = {"concept_id", "source_terms", "target_term", "provenance", "confidence", "notes"}
    required = allowed - {"notes"}
    for raw_entry in payload["entries"]:
        if not isinstance(raw_entry, dict) or not required <= set(raw_entry) or set(raw_entry) - allowed:
            raise ValueError("Pi consolidation response contains an invalid entry shape")
        source_terms = raw_entry["source_terms"]
        provenance = raw_entry["provenance"]
        confidence = raw_entry["confidence"]
        notes = raw_entry.get("notes", "")
        if (
            not isinstance(raw_entry["concept_id"], str)
            or not raw_entry["concept_id"].strip()
            or not isinstance(raw_entry["target_term"], str)
            or not raw_entry["target_term"].strip()
            or not isinstance(source_terms, list)
            or not source_terms
            or not all(isinstance(term, str) and term for term in source_terms)
            or not isinstance(provenance, list)
            or not provenance
            or not all(isinstance(block_id, str) and block_id for block_id in provenance)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(notes, str)
        ):
            raise ValueError("Pi consolidation response contains invalid entry values")
        entries.append(TerminologyEntry.from_dict({**raw_entry, "notes": notes}))
    if not entries:
        raise ValueError("Pi consolidation response must not discard a non-empty batch")
    return entries


def _validate_entries_against_groups(
    entries: Sequence[TerminologyEntry], groups: Sequence[TermGroup]
) -> None:
    """Ensure a model cannot add or silently drop auditable source evidence."""
    expected_forms = {form for group in groups for form in group.forms}
    expected_provenance = {block_id for group in groups for block_id in group.provenance}
    provenance_by_form = {
        form: set(group.provenance) for group in groups for form in group.forms
    }
    returned_forms = {form for entry in entries for form in entry.source_terms}
    returned_provenance = {block_id for entry in entries for block_id in entry.provenance}
    unknown_forms = returned_forms - expected_forms
    unknown_provenance = returned_provenance - expected_provenance
    if unknown_forms:
        raise ValueError("Pi consolidation response contains unknown source spelling")
    if unknown_provenance:
        raise ValueError("Pi consolidation response contains unknown provenance")
    if returned_forms != expected_forms:
        raise ValueError("Pi consolidation response did not preserve every source spelling")
    if returned_provenance != expected_provenance:
        raise ValueError("Pi consolidation response did not preserve every provenance block ID")
    for entry in entries:
        allowed_provenance = set().union(*(provenance_by_form[form] for form in entry.source_terms))
        if not set(entry.provenance) <= allowed_provenance:
            raise ValueError("Pi consolidation response mismatched source spelling and provenance")


def make_pi_consolidation_call(
    *, pi_bin: str = "pi", model: str, timeout: float = 120
) -> Callable[[str], str]:
    """Create an ephemeral, tool-less text Pi caller with bounded cleanup."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    def pi_call(prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("Pi consolidation prompt must be text")
        command = [
            pi_bin,
            "-p",
            "--model",
            model,
            "--no-session",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-approve",
            prompt,
        ]
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise PiConsolidationError(f"could not start Pi: {exc}") from exc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.communicate()
            raise PiConsolidationError(f"Pi consolidation timed out after {timeout}s") from None
        if proc.returncode:
            raise PiConsolidationError(f"Pi consolidation failed with code {proc.returncode}: {stderr[:500]}")
        return stdout.strip()

    return pi_call


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
    return sorted(
        canonical,
        key=lambda entry: (
            entry.concept_id,
            entry.target_term,
            tuple(entry.source_terms),
            tuple(entry.provenance),
            entry.confidence,
            entry.notes,
        ),
    )


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

    for _ in range(max_rounds):
        batches = batch_term_groups(groups, token_budget)
        results: list[TerminologyEntry] = []
        for batch in batches:
            prompt = _consolidation_prompt(batch.groups)
            prompt_tokens = estimate_tokens(prompt)
            if prompt_tokens > token_budget:
                raise AssertionError("batching produced an over-budget consolidation request")
            if prompt_tokens > HARD_TOKEN_CAP:
                raise ValueError("consolidation request exceeds the 200000 token hard cap")
            batch_entries = _parse_entries(pi_call(prompt))
            _validate_entries_against_groups(batch_entries, batch.groups)
            results.extend(batch_entries)
        results = _canonical_entries(results)
        if len(batches) == 1:
            return freeze_terminology(
                results, source_lang=source_lang, target_lang=target_lang, version=version
            )

        next_groups = _groups_from_entries(results)
        next_batches = batch_term_groups(next_groups, token_budget)
        progress_before = (len(batches), len(groups), sum(batch.token_count for batch in batches))
        progress_after = (
            len(next_batches),
            len(next_groups),
            sum(batch.token_count for batch in next_batches),
        )
        if progress_after >= progress_before:
            raise ValueError("consolidation did not reduce across recursive batches")
        groups = next_groups
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
        if entry_tokens > token_budget:
            raise ValueError("one terminology entry exceeds the configured token budget")
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
