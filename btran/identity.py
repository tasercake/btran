"""Exact immutable identities for ``identity-v1``.

This module deliberately owns only identity construction and raw-hash
reconciliation.  It has no storage, cache, revision, or dependency-graph
policy.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from btran.schema import (
    BookRecord,
    Finding,
    PageRecord,
    SchemaError,
    Segment,
    TermOccurrence,
    TerminologyConcept,
    canonical_json_bytes,
    tagged_sha256,
)

IDENTITY_VERSION = "identity-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class IdentityError(ValueError):
    """An input cannot participate in the fixed ``identity-v1`` algorithm."""


def canonical_source_text(value: str) -> str:
    """Canonical source text: NFC plus CR/LF normalization, and nothing else."""
    if not isinstance(value, str):
        raise IdentityError("source text must be a string")
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def _text_part(value: str, name: str) -> bytes:
    if not isinstance(value, str):
        raise IdentityError(f"{name} must be a string")
    return unicodedata.normalize("NFC", value).encode("utf-8")


def _raw_digest(raw_file_sha256: str) -> bytes:
    if not isinstance(raw_file_sha256, str) or not _HEX64.fullmatch(raw_file_sha256):
        raise IdentityError("raw_file_sha256 must be lower-case SHA-256 hex")
    return bytes.fromhex(raw_file_sha256)


def raw_file_sha256(raw_bytes: bytes) -> str:
    """Return lower-case SHA-256 of source file bytes without decoding them."""
    if not isinstance(raw_bytes, bytes):
        raise IdentityError("raw file content must be bytes")
    return hashlib.sha256(raw_bytes).hexdigest()


def page_id_for_raw_sha256(raw_file_sha256: str) -> str:
    """Return logical page identity from 32 raw SHA-256 digest bytes."""
    return tagged_sha256("page-v1", _raw_digest(raw_file_sha256))


def page_id_for_bytes(raw_bytes: bytes) -> str:
    return page_id_for_raw_sha256(raw_file_sha256(raw_bytes))


def book_id_for_page_ids(page_ids: Sequence[str]) -> str:
    """Hash sorted unique logical page IDs; order and duplicate placement do not matter."""
    if not isinstance(page_ids, (list, tuple, set, frozenset)):
        raise IdentityError("page_ids must be a sequence")
    ids = tuple(sorted(set(_identity_id(page_id, "page_id") for page_id in page_ids)))
    return tagged_sha256("book-v1", canonical_json_bytes(list(ids)))


def _identity_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise IdentityError(f"{name} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value):
        raise IdentityError(f"{name} must be NFC-normalized")
    return value


def canonical_relative_path(path: str | PurePosixPath) -> str:
    """Validate canonical POSIX relative path used only for page placement.

    The algorithm never resolves a path, folds case, or removes whitespace.
    Callers must supply already-canonical POSIX spelling so two spellings are
    never silently made equivalent.
    """
    text = path.as_posix() if isinstance(path, PurePosixPath) else path
    if not isinstance(text, str) or not text:
        raise IdentityError("relative path must be a non-empty string")
    text = unicodedata.normalize("NFC", text)
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or text in {".", ".."}
        or "\\" in text
        or "//" in text
        or text.endswith("/")
        # PurePosixPath normalizes away dot components, so validate original
        # spelling as well: otherwise "./a" and "a/./b" get distinct IDs.
        or any(part in {"", ".", ".."} for part in text.split("/"))
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise IdentityError("relative path must be canonical and remain below its root")
    return text


def placement_id_for(page_id: str, raw_file_sha256: str, relative_path: str | PurePosixPath) -> str:
    """Return placement-only identity.  Never use this as logical content identity."""
    return tagged_sha256(
        "placement-v1",
        _text_part(_identity_id(page_id, "page_id"), "page_id"),
        _raw_digest(raw_file_sha256),
        _text_part(canonical_relative_path(relative_path), "relative_path"),
    )


@dataclass(frozen=True)
class PagePlacement:
    """A non-identity render placement for one physical source-file path."""

    page_id: str
    raw_file_sha256: str
    relative_path: str
    placement_id: str

    def __post_init__(self) -> None:
        page = _identity_id(self.page_id, "page_id")
        _raw_digest(self.raw_file_sha256)
        relative = canonical_relative_path(self.relative_path)
        if self.relative_path != relative:
            raise IdentityError("relative_path must be NFC-normalized canonical spelling")
        if page != page_id_for_raw_sha256(self.raw_file_sha256):
            raise IdentityError("page_id does not match raw_file_sha256")
        expected = placement_id_for(page, self.raw_file_sha256, relative)
        if self.placement_id != expected:
            raise IdentityError("placement_id does not match placement identity inputs")

    @classmethod
    def create(cls, page_id: str, raw_file_sha256: str, relative_path: str | PurePosixPath) -> "PagePlacement":
        relative = canonical_relative_path(relative_path)
        page = _identity_id(page_id, "page_id")
        _raw_digest(raw_file_sha256)
        expected_page_id = page_id_for_raw_sha256(raw_file_sha256)
        if page != expected_page_id:
            raise IdentityError("page_id does not match raw_file_sha256")
        return cls(page, raw_file_sha256, relative, placement_id_for(page, raw_file_sha256, relative))

    @property
    def evidence(self) -> dict[str, str]:
        return {
            "duplicate_discriminator": "same-raw-bytes",
            "raw_file_sha256": self.raw_file_sha256,
        }


def page_record_for_raw_sha256(raw_file_sha256: str) -> PageRecord:
    """Build a ``PageRecord`` with its required exact identity."""
    _raw_digest(raw_file_sha256)
    return PageRecord(page_id=page_id_for_raw_sha256(raw_file_sha256), raw_file_sha256=raw_file_sha256)


def page_record_for_bytes(raw_bytes: bytes) -> PageRecord:
    return page_record_for_raw_sha256(raw_file_sha256(raw_bytes))


def book_record_for_pages(pages: Sequence[PageRecord]) -> BookRecord:
    if not isinstance(pages, (list, tuple)):
        raise IdentityError("pages must be a sequence")
    page_ids: list[str] = []
    for page in pages:
        if not isinstance(page, PageRecord):
            raise IdentityError("pages must contain PageRecord values")
        expected = page_id_for_raw_sha256(page.raw_file_sha256)
        if page.page_id != expected:
            raise IdentityError("PageRecord page_id does not match raw_file_sha256")
        page_ids.append(page.page_id)
    unique = tuple(sorted(set(page_ids)))
    return BookRecord(book_id=book_id_for_page_ids(unique), page_ids=unique)


def normalized_non_text_structural_fields(value: Any) -> Any:
    """Validate/canonicalize caller-supplied non-text structural fields.

    ``canonical_json_bytes`` applies required NFC normalization and rejects
    non-JSON/non-finite values.  Decoding it gives an ordinary deterministic
    JSON value, retaining all declared structure (rather than dropping fields).
    """
    try:
        import json

        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (SchemaError, TypeError, ValueError) as exc:
        raise IdentityError("non-text structural fields are not canonical JSON data") from exc


def structural_anchor_for(
    kind: str,
    reading_order: int,
    source_text: str,
    non_text_structural_fields: Any = None,
) -> str:
    """Return segment structural anchor from exact declared structural inputs."""
    if not isinstance(kind, str) or not kind:
        raise IdentityError("kind must be a non-empty string")
    if not isinstance(reading_order, int) or isinstance(reading_order, bool) or reading_order <= 0:
        raise IdentityError("reading_order must be a positive integer")
    fields = normalized_non_text_structural_fields(
        {} if non_text_structural_fields is None else non_text_structural_fields
    )
    return tagged_sha256(
        "anchor-v1",
        canonical_json_bytes([
            unicodedata.normalize("NFC", kind),
            reading_order,
            canonical_source_text(source_text),
            fields,
        ]),
    )


def segment_id_for(page_id: str, structural_anchor: str) -> str:
    return tagged_sha256(
        "segment-v1",
        _text_part(_identity_id(page_id, "page_id"), "page_id"),
        _text_part(_identity_id(structural_anchor, "structural_anchor"), "structural_anchor"),
    )


def segment_for(
    page_id: str,
    kind: str,
    reading_order: int,
    source_text: str,
    source_lang: str | None,
    non_text_structural_fields: Any = None,
) -> Segment:
    """Build a canonical ``Segment`` and verify its identity inputs."""
    text = canonical_source_text(source_text)
    anchor = structural_anchor_for(kind, reading_order, text, non_text_structural_fields)
    return Segment(
        segment_id=segment_id_for(page_id, anchor),
        page_id=_identity_id(page_id, "page_id"),
        structural_anchor=anchor,
        kind=unicodedata.normalize("NFC", kind),
        source_text=text,
        source_lang=source_lang,
        reading_order=reading_order,
    )


def occurrence_id_for(segment_id: str, start: int, end: int, surface: str) -> str:
    """Return occurrence identity.  Use ``occurrence_for`` to validate a span."""
    _identity_id(segment_id, "segment_id")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (start, end)) or start < 0 or end <= start:
        raise IdentityError("occurrence span must be a non-empty half-open range")
    return tagged_sha256(
        "occurrence-v1",
        _text_part(segment_id, "segment_id"),
        str(start).encode("ascii"),
        str(end).encode("ascii"),
        _text_part(canonical_source_text(surface), "surface"),
    )


def occurrence_for(segment: Segment, start: int, end: int, surface: str | None = None) -> TermOccurrence:
    """Create a Unicode-code-point occurrence after exact source slice validation."""
    if not isinstance(segment, Segment):
        raise IdentityError("segment must be a Segment")
    if segment.source_text != canonical_source_text(segment.source_text):
        raise IdentityError("segment source_text is not canonical source text")
    if segment.source_lang is None:
        raise IdentityError("diagnostic placeholder segments cannot have term occurrences")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (start, end)) or start < 0 or end <= start:
        raise IdentityError("occurrence span must be a non-empty half-open range")
    if end > len(segment.source_text):
        raise IdentityError("occurrence span exceeds source text")
    actual_surface = segment.source_text[start:end]
    expected_surface = actual_surface if surface is None else canonical_source_text(surface)
    if actual_surface != expected_surface:
        raise IdentityError("occurrence span must exactly slice to surface")
    return TermOccurrence(
        occurrence_id=occurrence_id_for(segment.segment_id, start, end, actual_surface),
        segment_id=segment.segment_id,
        start=start,
        end=end,
        surface=actual_surface,
        source_lang=segment.source_lang,
    )


def concept_id_for(source_lang: str, canonical_source_form: str, occurrence_ids: Sequence[str]) -> str:
    if not isinstance(source_lang, str) or not source_lang:
        raise IdentityError("source_lang must be a non-empty string")
    if not isinstance(occurrence_ids, (list, tuple, set, frozenset)):
        raise IdentityError("occurrence_ids must be a sequence")
    ids = tuple(sorted(set(_identity_id(item, "occurrence_id") for item in occurrence_ids)))
    if not ids:
        raise IdentityError("concept must contain occurrence IDs")
    return tagged_sha256(
        "concept-v1",
        canonical_json_bytes([
            unicodedata.normalize("NFC", source_lang),
            canonical_source_text(canonical_source_form),
            list(ids),
        ]),
    )


def concept_for(source_lang: str, canonical_source_form: str, occurrence_ids: Sequence[str]) -> TerminologyConcept:
    ids = tuple(sorted(set(_identity_id(item, "occurrence_id") for item in occurrence_ids)))
    return TerminologyConcept(
        concept_id=concept_id_for(source_lang, canonical_source_form, ids),
        source_lang=source_lang,
        canonical_source_form=canonical_source_text(canonical_source_form),
        occurrence_ids=ids,
    )


@dataclass(frozen=True)
class SegmentPlacement:
    """A detector occurrence of a logical segment; duplicates share ``segment_id``."""

    segment_id: str
    structural_anchor: str
    reading_order: int


@dataclass(frozen=True)
class CanonicalSegments:
    """In-memory canonicalized detector result; findings are informational."""

    segments: tuple[Segment, ...]
    placements: tuple[SegmentPlacement, ...]
    findings: tuple[Finding, ...]


def _identity_finding(kind: str, subject_refs: Sequence[str], evidence: dict[str, Any]) -> Finding:
    return Finding(
        kind=kind,
        severity="warning",
        stage="identity",
        subject_refs=tuple(sorted(set(subject_refs))),
        evidence=evidence,
        message=kind.replace("_", " "),
    )


def diagnostic_placeholder_segment(page_id: str, reason: str) -> Segment:
    """Deterministic identity-stage placeholder for invalid detector ordering."""
    text = f"[identity-v1 diagnostic: {reason}]"
    return segment_for(page_id, "diagnostic_placeholder", 1, text, None, {"reason": reason})


def canonical_root_segments(page_id: str, blocks: Sequence[Mapping[str, Any]]) -> CanonicalSegments:
    """Canonicalize detector blocks by root ``reading_order``.

    Each block must provide ``kind``, ``reading_order``, ``source_text`` (or
    legacy ``text``), and optional ``non_text_structural_fields``.  Non-positive
    orders, and one order assigned to distinct anchors, reject the root sequence
    and produce one explicit diagnostic placeholder.  Exact duplicate anchors
    are retained once, with every detector placement referring to that segment.
    """
    if not isinstance(blocks, (list, tuple)):
        raise IdentityError("blocks must be a sequence")
    try:
        parsed: list[Segment] = []
        for block in blocks:
            if not isinstance(block, Mapping):
                raise IdentityError("block must be an object")
            declared_fields = block.get("non_text_structural_fields")
            if declared_fields is None:
                declared_fields = {
                    key: value for key, value in block.items()
                    if key not in {"id", "kind", "type", "source_text", "text", "source_lang", "reading_order"}
                }
            parsed.append(segment_for(
                page_id,
                block.get("kind", block.get("type")),
                block.get("reading_order"),
                block.get("source_text", block.get("text")),
                block.get("source_lang"),
                declared_fields,
            ))
    except (SchemaError, TypeError, IdentityError) as exc:
        placeholder = diagnostic_placeholder_segment(page_id, "invalid_block")
        return CanonicalSegments(
            (placeholder,), (SegmentPlacement(placeholder.segment_id, placeholder.structural_anchor, 1),),
            (_identity_finding("invalid_root_sequence", (page_id,), {"reason": str(exc)}),),
        )

    by_order: dict[int, set[str]] = {}
    by_order_anchor: dict[tuple[int, str], Segment] = {}
    invalid_duplicate = False
    for segment in parsed:
        by_order.setdefault(segment.reading_order, set()).add(segment.structural_anchor)
        key = (segment.reading_order, segment.structural_anchor)
        existing = by_order_anchor.setdefault(key, segment)
        # An anchor deliberately excludes source language.  Thus a second block
        # can share it only when it is the same complete logical segment.
        if existing != segment:
            invalid_duplicate = True
    if any(len(anchors) > 1 for anchors in by_order.values()) or invalid_duplicate:
        placeholder = diagnostic_placeholder_segment(page_id, "invalid_reading_order")
        return CanonicalSegments(
            (placeholder,), (SegmentPlacement(placeholder.segment_id, placeholder.structural_anchor, 1),),
            (_identity_finding(
                "invalid_root_sequence", (page_id,),
                {"reading_orders": [segment.reading_order for segment in parsed]},
            ),),
        )

    segments_by_anchor: dict[str, Segment] = {}
    placements: list[SegmentPlacement] = []
    findings: list[Finding] = []
    for segment in sorted(parsed, key=lambda item: item.reading_order):
        duplicate = segments_by_anchor.get(segment.structural_anchor)
        if duplicate is None:
            segments_by_anchor[segment.structural_anchor] = segment
            duplicate = segment
        else:
            findings.append(_identity_finding(
                "duplicate_segment_identity", (page_id, duplicate.segment_id),
                {"structural_anchor": duplicate.structural_anchor, "segment_id": duplicate.segment_id},
            ))
        placements.append(SegmentPlacement(duplicate.segment_id, duplicate.structural_anchor, segment.reading_order))
    return CanonicalSegments(
        tuple(sorted(segments_by_anchor.values(), key=lambda segment: segment.reading_order)),
        tuple(placements),
        tuple(findings),
    )


@dataclass(frozen=True)
class RawHashReconciliation:
    """Result of zero/one/many exact raw-hash matching against historical pages."""

    raw_file_sha256: str
    status: str
    page_id: str | None
    candidate_page_ids: tuple[str, ...]
    finding: Finding | None = None


def _existing_page_parts(page: PageRecord | Mapping[str, str]) -> tuple[str, str]:
    if isinstance(page, PageRecord):
        page_id, digest = page.page_id, page.raw_file_sha256
    elif isinstance(page, Mapping):
        try:
            page_id, digest = _identity_id(page["page_id"], "page_id"), page["raw_file_sha256"]
        except KeyError as exc:
            raise IdentityError("existing page needs page_id and raw_file_sha256") from exc
    else:
        raise IdentityError("existing pages must be PageRecord values or mappings")
    _raw_digest(digest)
    return page_id, digest


def reconcile_raw_hash(raw_file_sha256: str, existing_pages: Sequence[PageRecord | Mapping[str, str]]) -> RawHashReconciliation:
    """Reconcile only exact raw hashes; never use filenames, text, or fuzzy hashes."""
    _raw_digest(raw_file_sha256)
    if not isinstance(existing_pages, (list, tuple)):
        raise IdentityError("existing_pages must be a sequence")
    matches = sorted({page_id for page_id, digest in map(_existing_page_parts, existing_pages) if digest == raw_file_sha256})
    if not matches:
        return RawHashReconciliation(raw_file_sha256, "new", None, ())
    if len(matches) == 1:
        return RawHashReconciliation(raw_file_sha256, "reused", matches[0], tuple(matches))
    finding = _identity_finding(
        "duplicate_identity_ambiguous", matches,
        {"raw_file_sha256": raw_file_sha256, "candidate_page_ids": matches},
    )
    return RawHashReconciliation(raw_file_sha256, "ambiguous", None, tuple(matches), finding)


def reconcile_book_pages(
    discovered_raw_hashes: Sequence[str], existing_pages: Sequence[PageRecord | Mapping[str, str]],
) -> tuple[RawHashReconciliation, ...]:
    """Apply exact zero/one/many reconciliation independently to discovered files."""
    if not isinstance(discovered_raw_hashes, (list, tuple)):
        raise IdentityError("discovered_raw_hashes must be a sequence")
    return tuple(reconcile_raw_hash(raw_hash, existing_pages) for raw_hash in discovered_raw_hashes)


# Short aliases keep the identities easy to use at stage boundaries.
page_id = page_id_for_raw_sha256
book_id = book_id_for_page_ids
placement_id = placement_id_for
structural_anchor = structural_anchor_for
segment_id = segment_id_for
occurrence_id = occurrence_id_for
concept_id = concept_id_for
