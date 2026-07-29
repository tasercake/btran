"""Legacy migration records plus strict canonical schemas for btran state.

New pipeline state uses the versioned records below.  The pre-existing records at
bottom intentionally remain permissive migration readers only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence, get_origin, get_type_hints

SCHEMA_VERSION = "schema-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_BCP47 = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_SEVERITIES = frozenset({"info", "warning", "error"})
_STAGE_STATUSES = frozenset({"completed", "degraded"})
_MODES = frozenset({"native", "translated"})
_REVIEW_SCOPES = frozenset({"occurrence", "segment", "all_concept_occurrences", "subset_occurrence_ids"})


class SchemaError(ValueError):
    """A persisted canonical-schema value is malformed or non-canonical."""


def _nfc(value: str, name: str = "string") -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if value != normalized:
        raise SchemaError(f"{name} must be NFC-normalized")
    return value


def _normalized(value: Any) -> Any:
    """Return canonical, NFC-normalized JSON-compatible data for writers."""
    if isinstance(value, CanonicalRecord):
        return _normalized(value.to_dict())
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {
            unicodedata.normalize("NFC", str(key)): _normalized(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError("non-finite floats are not canonical JSON data")
        return value
    if isinstance(value, (type(None), bool, int)):
        return value
    raise SchemaError(f"not canonical JSON data: {type(value).__name__}")


def _normalize_schema_value(value: Any) -> Any:
    """Normalize constructor input while preserving nested canonical records."""
    if isinstance(value, CanonicalRecord):
        return value
    if isinstance(value, Mapping):
        return {
            unicodedata.normalize("NFC", str(key)): _normalize_schema_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_schema_value(item) for item in value]
    return _normalized(value)


def _require_nfc(value: Any, name: str = "value") -> None:
    if isinstance(value, CanonicalRecord):
        _require_nfc(value.to_dict(), name)
    elif isinstance(value, str):
        _nfc(value, name)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _nfc(key, f"{name} key")
            _require_nfc(item, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_nfc(item, name)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(f"{name} cannot contain non-finite floats")
    elif not isinstance(value, (type(None), bool, int)):
        raise SchemaError(f"{name} is not JSON-compatible")


def canonical_json(value: Any) -> str:
    """Canonical Unicode JSON text (NFC, sorted keys, no insignificant space)."""
    if isinstance(value, CanonicalRecord):
        value = value.to_dict()
    return json.dumps(
        _normalized(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def tagged_sha256(tag: str, *parts: bytes) -> str:
    """H(tag, parts...) from PLAN.md.  Callers choose each part's encoding."""
    if not isinstance(tag, str) or not tag.isascii():
        raise SchemaError("hash tag must be ASCII")
    if any(not isinstance(part, bytes) for part in parts):
        raise SchemaError("hash parts must be bytes")
    return hashlib.sha256(tag.encode("ascii") + b"\0" + b"\0".join(parts)).hexdigest()


def _string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    value = _nfc(value, name)
    if not allow_empty and not value:
        raise SchemaError(f"{name} must not be empty")
    return value


def _identifier(value: Any, name: str) -> str:
    return _string(value, name)


def _sorted_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SchemaError(f"{name} must be an array")
    checked = tuple(_identifier(item, name) for item in values)
    if tuple(sorted(set(checked))) != checked:
        raise SchemaError(f"{name} must be sorted and unique")
    return checked


def _ordered_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SchemaError(f"{name} must be an array")
    checked = tuple(_identifier(item, name) for item in values)
    if len(set(checked)) != len(checked):
        raise SchemaError(f"{name} must not contain duplicates")
    return checked


def _language(value: str | None, name: str, *, permit_none: bool = False) -> None:
    if value is None and permit_none:
        return
    value = _string(value, name)
    if not _BCP47.fullmatch(value):
        raise SchemaError(f"{name} must be a BCP-47 language tag")


@dataclass
class CanonicalRecord:
    """Base for strict records; unknown/missing persisted fields always reject."""

    schema_version: str = SCHEMA_VERSION
    _schema_name: ClassVar[str] = "record"

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"unsupported {self._schema_name} schema version")
        # Writers normalize all schema strings; rejecting readers check raw input before this.
        for item in fields(self):
            setattr(self, item.name, _normalize_schema_value(getattr(self, item.name)))
        # JSON arrays deserialize as lists; canonical records expose immutable tuples.
        for name, annotation in get_type_hints(type(self)).items():
            if get_origin(annotation) is tuple and isinstance(getattr(self, name, None), list):
                setattr(self, name, tuple(getattr(self, name)))
        self._validate()

    def _validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        result = _normalized(asdict(self))
        # Constructors reject non-NFC persisted strings; writers canonicalize values
        # supplied by in-process callers, as required by canonical writer contract.
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_json().encode("utf-8"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            raise SchemaError(f"{cls._schema_name} must be an object")
        expected = {item.name for item in fields(cls)}
        actual = set(data)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise SchemaError(f"{cls._schema_name} fields mismatch: missing={missing}, unknown={unknown}")
        _require_nfc(data, cls._schema_name)
        return cls(**dict(data))

    @classmethod
    def from_json(cls, text: str):
        _nfc(text, "JSON text")
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SchemaError(f"invalid {cls._schema_name} JSON") from exc
        record = cls.from_dict(data)
        if text.encode("utf-8") != record.to_json().encode("utf-8"):
            raise SchemaError(f"{cls._schema_name} JSON is not canonical")
        return record

    @classmethod
    def from_file(cls, path: Path):
        try:
            return cls.from_json(path.read_bytes().decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SchemaError(f"{cls._schema_name} file is not UTF-8") from exc


@dataclass
class Finding(CanonicalRecord):
    finding_id: str = ""
    kind: str = ""
    severity: str = "info"
    stage: str = ""
    subject_refs: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    dependency_ids: tuple[str, ...] = ()
    requires_action: bool = False
    _schema_name: ClassVar[str] = "finding"

    def _validate(self) -> None:
        _string(self.kind, "kind")
        _string(self.stage, "stage")
        _string(self.message, "message", allow_empty=True)
        if self.severity not in _SEVERITIES:
            raise SchemaError("finding severity must be info, warning, or error")
        if self.requires_action is not False:
            raise SchemaError("findings are informational and requires_action must be False")
        _sorted_unique(self.subject_refs, "subject_refs")
        _sorted_unique(self.dependency_ids, "dependency_ids")
        if not isinstance(self.evidence, dict):
            raise SchemaError("finding evidence must be an object")
        _require_nfc(self.evidence, "finding evidence")
        expected_id = tagged_sha256(
            "finding-v1",
            canonical_json_bytes({
                "schema_version": self.schema_version, "kind": self.kind, "severity": self.severity,
                "stage": self.stage, "subject_refs": list(self.subject_refs), "evidence": self.evidence,
                "message": self.message, "dependency_ids": list(self.dependency_ids),
                "requires_action": False,
            }),
        )
        if self.finding_id and self.finding_id != expected_id:
            raise SchemaError("finding_id does not match canonical finding body")
        self.finding_id = expected_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping) or not data.get("finding_id"):
            raise SchemaError("persisted finding requires canonical finding_id")
        return super().from_dict(data)


@dataclass
class ConfidenceAssessment(CanonicalRecord):
    subject_id: str = ""
    producing_stage: str = ""
    producing_artifact_id: str = ""
    score: float | None = None
    signals: tuple[str, ...] = ()
    assessment_version: str = "confidence-v1"
    uncertainty_finding_id: str | None = None
    _schema_name: ClassVar[str] = "confidence_assessment"

    def _validate(self) -> None:
        _identifier(self.subject_id, "subject_id")
        _string(self.producing_stage, "producing_stage")
        _identifier(self.producing_artifact_id, "producing_artifact_id")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not 0 <= self.score <= 1:
                raise SchemaError("confidence score must be in [0, 1] or null")
        _sorted_unique(self.signals, "signals")
        _string(self.assessment_version, "assessment_version")
        if self.uncertainty_finding_id is not None:
            _identifier(self.uncertainty_finding_id, "uncertainty_finding_id")


@dataclass
class StageRecord(CanonicalRecord):
    stage: str = ""
    status: str = "completed"
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    stage_summary_finding_id: str = ""
    _schema_name: ClassVar[str] = "stage_record"

    def _validate(self) -> None:
        _string(self.stage, "stage")
        if self.status not in _STAGE_STATUSES:
            raise SchemaError("stage status must be completed or degraded")
        _sorted_unique(self.input_artifact_ids, "input_artifact_ids")
        _sorted_unique(self.output_artifact_ids, "output_artifact_ids")
        finding_ids = _sorted_unique(self.finding_ids, "finding_ids")
        _identifier(self.stage_summary_finding_id, "stage_summary_finding_id")
        if self.stage_summary_finding_id not in finding_ids:
            raise SchemaError("completed/degraded stage must retain its stage_summary finding")


@dataclass
class ArtifactEnvelope(CanonicalRecord):
    artifact_id: str = ""
    kind: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    dependency_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    semantic_key: str = ""
    _schema_name: ClassVar[str] = "artifact_envelope"

    def _validate(self) -> None:
        _identifier(self.artifact_id, "artifact_id")
        _string(self.kind, "kind")
        if not isinstance(self.payload, dict):
            raise SchemaError("artifact payload must be an object")
        _require_nfc(self.payload, "artifact payload")
        _sorted_unique(self.dependency_ids, "dependency_ids")
        _sorted_unique(self.finding_ids, "finding_ids")
        _identifier(self.semantic_key, "semantic_key")


@dataclass
class BookRecord(CanonicalRecord):
    book_id: str = ""
    page_ids: tuple[str, ...] = ()
    _schema_name: ClassVar[str] = "book_record"

    def _validate(self) -> None:
        _identifier(self.book_id, "book_id")
        _sorted_unique(self.page_ids, "page_ids")


@dataclass
class PageRecord(CanonicalRecord):
    page_id: str = ""
    raw_file_sha256: str = ""
    duplicate_discriminator: str = "same-raw-bytes"
    _schema_name: ClassVar[str] = "page_record"

    def _validate(self) -> None:
        _identifier(self.page_id, "page_id")
        if not isinstance(self.raw_file_sha256, str) or not _HEX64.fullmatch(self.raw_file_sha256):
            raise SchemaError("raw_file_sha256 must be lower-case SHA-256 hex")
        if self.duplicate_discriminator != "same-raw-bytes":
            raise SchemaError("duplicate_discriminator must be same-raw-bytes")


@dataclass
class Segment(CanonicalRecord):
    segment_id: str = ""
    page_id: str = ""
    structural_anchor: str = ""
    kind: str = ""
    source_text: str = ""
    source_lang: str | None = None
    reading_order: int = 0
    _schema_name: ClassVar[str] = "segment"

    def _validate(self) -> None:
        _identifier(self.segment_id, "segment_id")
        _identifier(self.page_id, "page_id")
        _identifier(self.structural_anchor, "structural_anchor")
        _string(self.kind, "kind")
        _string(self.source_text, "source_text", allow_empty=True)
        _language(self.source_lang, "source_lang", permit_none=True)
        if self.source_lang is None and self.kind != "diagnostic_placeholder":
            raise SchemaError("only diagnostic placeholder segment may omit source_lang")
        if not isinstance(self.reading_order, int) or isinstance(self.reading_order, bool) or self.reading_order <= 0:
            raise SchemaError("reading_order must be a positive integer")


@dataclass
class TermOccurrence(CanonicalRecord):
    occurrence_id: str = ""
    segment_id: str = ""
    start: int = 0
    end: int = 0
    surface: str = ""
    source_lang: str = ""
    _schema_name: ClassVar[str] = "term_occurrence"

    def _validate(self) -> None:
        _identifier(self.occurrence_id, "occurrence_id")
        _identifier(self.segment_id, "segment_id")
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (self.start, self.end)) or self.start < 0 or self.end <= self.start:
            raise SchemaError("occurrence span must be non-empty half-open non-negative range")
        _string(self.surface, "surface")
        _language(self.source_lang, "source_lang")


@dataclass
class TerminologyConcept(CanonicalRecord):
    concept_id: str = ""
    source_lang: str = ""
    canonical_source_form: str = ""
    occurrence_ids: tuple[str, ...] = ()
    _schema_name: ClassVar[str] = "terminology_concept"

    def _validate(self) -> None:
        _identifier(self.concept_id, "concept_id")
        _language(self.source_lang, "source_lang")
        _string(self.canonical_source_form, "canonical_source_form")
        ids = _sorted_unique(self.occurrence_ids, "occurrence_ids")
        if not ids:
            raise SchemaError("concept must contain occurrence IDs")


@dataclass
class ConceptProjection(CanonicalRecord):
    projection_id: str = ""
    concept_id: str = ""
    membership_id: str = ""
    selector_occurrence_ids: tuple[str, ...] = ()
    target_form: str = ""
    correction_id: str | None = None
    _schema_name: ClassVar[str] = "concept_projection"

    def _validate(self) -> None:
        _identifier(self.projection_id, "projection_id")
        _identifier(self.concept_id, "concept_id")
        _identifier(self.membership_id, "membership_id")
        _sorted_unique(self.selector_occurrence_ids, "selector_occurrence_ids")
        _string(self.target_form, "target_form", allow_empty=True)
        if self.correction_id is not None:
            _identifier(self.correction_id, "correction_id")


@dataclass
class TranslationArtifact(CanonicalRecord):
    translation_artifact_id: str = ""
    segment_id: str = ""
    target_lang: str = ""
    translated_text: str = ""
    source_artifact_id: str = ""
    projection_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    _schema_name: ClassVar[str] = "translation_artifact"

    def _validate(self) -> None:
        _identifier(self.translation_artifact_id, "translation_artifact_id")
        _identifier(self.segment_id, "segment_id")
        _language(self.target_lang, "target_lang")
        _string(self.translated_text, "translated_text", allow_empty=True)
        _identifier(self.source_artifact_id, "source_artifact_id")
        _sorted_unique(self.projection_ids, "projection_ids")
        _sorted_unique(self.finding_ids, "finding_ids")


@dataclass
class OccurrenceTargetMapping(CanonicalRecord):
    mapping_id: str = ""
    occurrence_id: str = ""
    segment_id: str = ""
    translation_artifact_id: str = ""
    start: int = 0
    end: int = 0
    target_text: str = ""
    _schema_name: ClassVar[str] = "occurrence_target_mapping"

    def _validate(self) -> None:
        _identifier(self.mapping_id, "mapping_id")
        _identifier(self.occurrence_id, "occurrence_id")
        _identifier(self.segment_id, "segment_id")
        _identifier(self.translation_artifact_id, "translation_artifact_id")
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (self.start, self.end)) or self.start < 0 or self.end <= self.start:
            raise SchemaError("mapping span must be non-empty half-open non-negative range")
        _string(self.target_text, "target_text")


@dataclass
class EffectiveSegment(CanonicalRecord):
    effective_segment_id: str = ""
    segment_id: str = ""
    source_lang: str | None = None
    source_text: str = ""
    effective_text: str = ""
    render_lang: str = ""
    mode: str = "native"
    translation_artifact_id: str | None = None
    source_overlay_artifact_id: str | None = None
    target_overlay_artifact_id: str | None = None
    correction_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    _schema_name: ClassVar[str] = "effective_segment"

    def _validate(self) -> None:
        _identifier(self.effective_segment_id, "effective_segment_id")
        _identifier(self.segment_id, "segment_id")
        _language(self.source_lang, "source_lang", permit_none=True)
        _string(self.source_text, "source_text", allow_empty=True)
        _string(self.effective_text, "effective_text", allow_empty=True)
        _language(self.render_lang, "render_lang")
        if self.mode not in _MODES:
            raise SchemaError("effective segment mode must be native or translated")
        for name in ("translation_artifact_id", "source_overlay_artifact_id", "target_overlay_artifact_id"):
            value = getattr(self, name)
            if value is not None:
                _identifier(value, name)
        _sorted_unique(self.correction_ids, "correction_ids")
        _sorted_unique(self.finding_ids, "finding_ids")
        if self.mode == "native":
            if self.translation_artifact_id is not None or self.target_overlay_artifact_id is not None:
                raise SchemaError("native effective segment cannot own target translation/overlay")
            if self.effective_text != self.source_text:
                raise SchemaError("native effective text must equal source text")
            expected_lang = self.source_lang if self.source_lang is not None else "und"
            if self.render_lang != expected_lang:
                raise SchemaError("native render_lang must equal source_lang (or und diagnostic)")
            if self.source_lang is None and not self.finding_ids:
                raise SchemaError("diagnostic effective segment must retain a finding")
        else:
            # Target documents retain source-stage diagnostics as target-mode
            # ``und`` leaves. They carry provenance/findings but have no
            # translation artifact or target-language text.
            if self.source_lang is None:
                if self.render_lang != "und":
                    raise SchemaError("translated diagnostic effective segment render_lang must be und")
                if not self.finding_ids:
                    raise SchemaError("diagnostic effective segment must retain a finding")
                if self.translation_artifact_id is not None or self.target_overlay_artifact_id is not None:
                    raise SchemaError("translated diagnostic effective segment cannot own translation/overlay")
            elif self.render_lang == "und":
                raise SchemaError("translated effective segment render_lang cannot be und")


@dataclass
class EffectivePage(CanonicalRecord):
    effective_page_id: str = ""
    page_id: str = ""
    effective_segment_ids: tuple[str, ...] = ()
    source_langs: tuple[str, ...] = ()
    display_metadata: dict[str, Any] = field(default_factory=dict)
    finding_ids: tuple[str, ...] = ()
    _schema_name: ClassVar[str] = "effective_page"

    def _validate(self) -> None:
        _identifier(self.effective_page_id, "effective_page_id")
        _identifier(self.page_id, "page_id")
        _ordered_unique(self.effective_segment_ids, "effective_segment_ids")
        languages = _sorted_unique(self.source_langs, "source_langs")
        for lang in languages:
            _language(lang, "source_langs")
        if "und" in languages:
            raise SchemaError("effective page source_langs cannot include diagnostic und")
        if not isinstance(self.display_metadata, dict):
            raise SchemaError("display_metadata must be an object")
        _require_nfc(self.display_metadata, "display_metadata")
        _sorted_unique(self.finding_ids, "finding_ids")


@dataclass
class RevisionSnapshot(CanonicalRecord):
    revision_id: str = ""
    selected_artifact_ids: tuple[str, ...] = ()
    selected_finding_ids: tuple[str, ...] = ()
    # Immutable semantic-key bindings for selected canonical artifacts.  These
    # IDs are copied into sealed bundles so cache reuse can prove key->artifact
    # selection without consulting mutable cache history or index order.
    selected_cache_attestation_ids: tuple[str, ...] = ()
    correction_set_id: str | None = None
    _schema_name: ClassVar[str] = "revision_snapshot"

    def _validate(self) -> None:
        _identifier(self.revision_id, "revision_id")
        _sorted_unique(self.selected_artifact_ids, "selected_artifact_ids")
        _sorted_unique(self.selected_finding_ids, "selected_finding_ids")
        _sorted_unique(self.selected_cache_attestation_ids, "selected_cache_attestation_ids")
        if self.correction_set_id is not None:
            _identifier(self.correction_set_id, "correction_set_id")

    def to_dict(self) -> dict[str, Any]:
        body = super().to_dict()
        # Read old sealed snapshots byte-for-byte: an absent optional mapping
        # denotes no retained semantic attestations.
        if not self.selected_cache_attestation_ids:
            body.pop("selected_cache_attestation_ids")
        return body

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if isinstance(data, Mapping) and "selected_cache_attestation_ids" not in data:
            data = {**data, "selected_cache_attestation_ids": ()}
        return super().from_dict(data)


@dataclass
class CorrectionRecord(CanonicalRecord):
    correction_id: str = ""
    kind: str = ""
    applies_to_revision_id: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    replacement: str = ""
    supersedes_id: str | None = None
    _schema_name: ClassVar[str] = "correction_record"

    def _validate(self) -> None:
        _identifier(self.correction_id, "correction_id")
        _string(self.kind, "kind")
        _identifier(self.applies_to_revision_id, "applies_to_revision_id")
        if not isinstance(self.scope, dict) or not isinstance(self.base, dict):
            raise SchemaError("correction scope and base must be objects")
        _require_nfc(self.scope, "correction scope")
        _require_nfc(self.base, "correction base")
        _string(self.replacement, "replacement", allow_empty=True)
        if self.supersedes_id is not None:
            _identifier(self.supersedes_id, "supersedes_id")


@dataclass
class CorrectionEvent(CanonicalRecord):
    """Immutable transition evidence for one successor correction-set state.

    ``event_id`` binds these state fields, not ``correction_set_id``.  This
    deliberately breaks the event-ID/set-ID hash cycle: the set hashes event
    IDs, while an event hashes the successor's base revision and active IDs.
    The stored set pointer is then checked against both values on load.
    """
    event_id: str = ""
    correction_id: str = ""
    event_kind: str = ""
    correction_set_id: str = ""
    successor_base_revision_id: str = ""
    successor_active_correction_ids: tuple[str, ...] = ()
    _schema_name: ClassVar[str] = "correction_event"

    def _validate(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.correction_id, "correction_id")
        if self.event_kind not in {"apply", "revert", "supersede"}:
            raise SchemaError("correction event kind is invalid")
        _identifier(self.correction_set_id, "correction_set_id")
        _identifier(self.successor_base_revision_id, "successor_base_revision_id")
        _sorted_unique(self.successor_active_correction_ids, "successor_active_correction_ids")


@dataclass
class CorrectionImpact(CanonicalRecord):
    # Correction commands only project impact; execution records may later
    # report actual produced artifacts against the same projection plan.
    phase: str = "correction"
    base_revision_id: str = ""
    projection_plan_id: str = ""
    # Correction-time binding lets later execution use this immutable plan,
    # rather than readdressing a correction under a newer active event set.
    correction_id: str | None = None
    correction_set_id: str | None = None
    projected_universe: tuple[dict[str, Any], ...] = ()
    affected: tuple[dict[str, Any], ...] = ()
    unaffected: tuple[dict[str, Any], ...] = ()
    ambiguous: tuple[dict[str, Any], ...] = ()
    protected: tuple[dict[str, Any], ...] = ()
    reused: tuple[dict[str, Any], ...] = ()
    regenerated: tuple[dict[str, Any], ...] = ()
    _schema_name: ClassVar[str] = "correction_impact"

    def _validate_entry_list(self, entries: Sequence[dict[str, Any]], name: str) -> tuple[tuple[str, str, str], ...]:
        if not isinstance(entries, (tuple, list)):
            raise SchemaError(f"{name} must be an array")
        normal: list[tuple[str, str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"stage", "subject_id", "base_artifact_id"}:
                raise SchemaError(f"{name} entries need stage, subject_id, base_artifact_id")
            _require_nfc(entry, name)
            normal.append((_string(entry["stage"], "stage"), _identifier(entry["subject_id"], "subject_id"), _identifier(entry["base_artifact_id"], "base_artifact_id")))
        if tuple(sorted(set(normal))) != tuple(normal):
            raise SchemaError(f"{name} must be sorted and unique")
        return tuple(normal)

    def _validate(self) -> None:
        if self.phase not in {"correction", "execution"}:
            raise SchemaError("correction impact phase is invalid")
        _identifier(self.base_revision_id, "base_revision_id")
        _identifier(self.projection_plan_id, "projection_plan_id")
        if (self.correction_id is None) != (self.correction_set_id is None):
            raise SchemaError("correction impact correction ID and set ID must be paired")
        if self.correction_id is not None:
            _identifier(self.correction_id, "correction_id")
            _identifier(self.correction_set_id, "correction_set_id")
        universe = self._validate_entry_list(self.projected_universe, "projected_universe")
        groups = {name: self._validate_entry_list(getattr(self, name), name) for name in ("affected", "unaffected", "ambiguous", "protected", "reused", "regenerated")}
        partitions = (groups["affected"], groups["unaffected"], groups["ambiguous"], groups["protected"])
        if set().union(*map(set, partitions)) != set(universe) or sum(map(len, partitions)) != len(universe):
            raise SchemaError("affected/unaffected/ambiguous/protected must be disjoint exhaustive universe")
        if not set(groups["reused"]).issubset(set(groups["unaffected"]) | set(groups["protected"])):
            raise SchemaError("reused must only overlap unaffected/protected")
        if self.phase == "correction" and groups["regenerated"]:
            raise SchemaError("non-executing correction impact regenerated must be empty")


@dataclass
class RefreshAttempt(CanonicalRecord):
    refresh_attempt_id: str = ""
    base_revision_id: str = ""
    reachable_artifact_ids: tuple[str, ...] = ()
    candidate_revision_id: str = ""
    _schema_name: ClassVar[str] = "refresh_attempt"

    def _validate(self) -> None:
        _identifier(self.refresh_attempt_id, "refresh_attempt_id")
        _identifier(self.base_revision_id, "base_revision_id")
        _sorted_unique(self.reachable_artifact_ids, "reachable_artifact_ids")
        _identifier(self.candidate_revision_id, "candidate_revision_id")


@dataclass
class DependencyGraphEdge(CanonicalRecord):
    edge_id: str = ""
    stable_subject_id: str = ""
    parent_artifact_id: str = ""
    child_artifact_id: str = ""
    stage: str = ""
    edge_kind: str = ""
    _schema_name: ClassVar[str] = "dependency_graph_edge"

    def _validate(self) -> None:
        for name in ("edge_id", "stable_subject_id", "parent_artifact_id", "child_artifact_id"):
            _identifier(getattr(self, name), name)
        _string(self.stage, "stage")
        _string(self.edge_kind, "edge_kind")
        if self.parent_artifact_id == self.child_artifact_id:
            raise SchemaError("dependency edge cannot self-reference")


# Alias makes persisted-edge purpose obvious without adding graph storage in Task 1.
DependencyEdge = DependencyGraphEdge


@dataclass
class RunReport(CanonicalRecord):
    run_id: str = ""
    mode: str = "native"
    content_finding_ids: tuple[str, ...] = ()
    uncertainty_finding_ids: tuple[str, ...] = ()
    review_finding_ids: tuple[str, ...] = ()
    recoverable_failure_finding_ids: tuple[str, ...] = ()
    invocation_failures: tuple[dict[str, Any], ...] = ()
    cache_events: tuple[dict[str, Any], ...] = ()
    # Ordered physical output positions. Logical artifacts remain sets; this is
    # intentionally sequence-valued display/provenance data.
    placement_provenance: tuple[dict[str, Any], ...] = ()
    correction_execution_projection_plan_ids: tuple[str, ...] = ()
    refresh_attempt_ids: tuple[str, ...] = ()
    selected_base_revision_id: str | None = None
    candidate_revision_id: str | None = None
    active_revision_id: str | None = None
    final_epub_status: str = ""
    stage_records: tuple[StageRecord, ...] = ()
    _schema_name: ClassVar[str] = "run_report"

    def _validate(self) -> None:
        _identifier(self.run_id, "run_id")
        if self.mode not in _MODES:
            raise SchemaError("report mode must be native or translated")
        groups = ("content_finding_ids", "uncertainty_finding_ids", "review_finding_ids", "recoverable_failure_finding_ids")
        for name in groups:
            _sorted_unique(getattr(self, name), name)
        for name in ("correction_execution_projection_plan_ids", "refresh_attempt_ids"):
            _sorted_unique(getattr(self, name), name)
        for name in ("selected_base_revision_id", "candidate_revision_id", "active_revision_id"):
            value = getattr(self, name)
            if value is not None:
                _identifier(value, name)
        _string(self.final_epub_status, "final_epub_status")
        for name in ("invocation_failures", "cache_events"):
            entries = getattr(self, name)
            if not isinstance(entries, (tuple, list)):
                raise SchemaError(f"{name} must be an array")
            _require_nfc(entries, name)
        if not isinstance(self.placement_provenance, (tuple, list)):
            raise SchemaError("placement_provenance must be an array")
        placement_ids: set[str] = set()
        for item in self.placement_provenance:
            if not isinstance(item, Mapping):
                raise SchemaError("placement_provenance entries must be objects")
            required = {"placement_id", "page_id", "effective_page_id", "relative_path",
                        "effective_page_artifact_id", "effective_segment_artifact_ids"}
            if set(item) != required:
                raise SchemaError("placement_provenance entry has invalid fields")
            for name in ("placement_id", "page_id", "effective_page_id", "effective_page_artifact_id"):
                _identifier(item[name], f"placement_provenance.{name}")
            _string(item["relative_path"], "placement_provenance.relative_path", allow_empty=True)
            _ordered_unique(item["effective_segment_artifact_ids"], "placement_provenance.effective_segment_artifact_ids")
            if item["placement_id"] in placement_ids:
                raise SchemaError("placement_provenance placement IDs must be unique")
            placement_ids.add(item["placement_id"])
        if not isinstance(self.stage_records, (tuple, list)):
            raise SchemaError("stage_records must be an array")
        stages: list[str] = []
        records: list[StageRecord] = []
        for record in self.stage_records:
            if not isinstance(record, StageRecord):
                if not isinstance(record, Mapping):
                    raise SchemaError("stage_records entries must be StageRecord objects")
                record = StageRecord.from_dict(record)
            records.append(record)
            stages.append(record.stage)
        self.stage_records = tuple(records)
        if len(set(stages)) != len(stages):
            raise SchemaError("run report stage records must have unique stages")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        data = dict(data)
        if isinstance(data.get("stage_records"), list):
            data["stage_records"] = tuple(StageRecord.from_dict(item) for item in data["stage_records"])
        return super().from_dict(data)


def uncertainty_finding(assessment: ConfidenceAssessment) -> Finding:
    """Create deterministic informational uncertainty finding for an assessment."""
    if not isinstance(assessment, ConfidenceAssessment):
        raise SchemaError("assessment must be ConfidenceAssessment")
    evidence = {
        "assessment_version": assessment.assessment_version,
        "producing_artifact_id": assessment.producing_artifact_id,
        "score": assessment.score,
        "signals": list(assessment.signals),
    }
    finding = Finding(kind="uncertainty", severity="warning" if assessment.score is None or assessment.score < .8 else "info", stage=assessment.producing_stage, subject_refs=(assessment.subject_id,), evidence=evidence, message="Confidence assessment recorded.", dependency_ids=(assessment.producing_artifact_id,))
    assessment.uncertainty_finding_id = finding.finding_id
    return finding


def stage_summary_finding(stage: str, status: str, counts: Mapping[str, int], *, subject_refs: Sequence[str] = ()) -> Finding:
    """Required informational summary for a completed/degraded stage."""
    if status not in _STAGE_STATUSES:
        raise SchemaError("stage summary status must be completed or degraded")
    if not isinstance(counts, Mapping) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts.values()):
        raise SchemaError("stage summary counts must be non-negative integers")
    return Finding(kind="stage_summary", severity="info", stage=stage, subject_refs=tuple(sorted(set(subject_refs))), evidence={"status": status, "counts": dict(counts)}, message="Stage completed informationally.")


def review_request(
    *, trigger: str, stage: str, subject_ids: Sequence[str], suggested_correction_kind: str,
    base_revision_id: str, base_artifact_ids: Sequence[str], scope: str,
    occurrence_ids: Sequence[str] = (), message: str = "Review is optional; pipeline continues.",
) -> Finding:
    """One shared deterministic, strictly non-gating review-request policy primitive."""
    allowed_triggers = {
        "low_confidence", "degraded_unknown_confidence", "source_sense_ambiguity",
        "concept_ambiguity", "mapping_ambiguity", "correction_ambiguity",
        "reconciliation_conflict", "missing_term", "validation_error",
    }
    if trigger not in allowed_triggers:
        raise SchemaError("review request trigger is not policy-defined")
    subjects = _sorted_unique(subject_ids, "subject_ids")
    bases = _sorted_unique(base_artifact_ids, "base_artifact_ids")
    if not subjects or not bases:
        raise SchemaError("review request requires exact applicable subjects and base artifacts")
    _identifier(base_revision_id, "base_revision_id")
    if scope not in _REVIEW_SCOPES:
        raise SchemaError("review request scope is invalid")
    occurrence_ids = _sorted_unique(occurrence_ids, "occurrence_ids")
    if scope == "subset_occurrence_ids" and not occurrence_ids:
        raise SchemaError("subset occurrence scope requires occurrence IDs")
    if scope != "subset_occurrence_ids" and occurrence_ids:
        raise SchemaError("only subset occurrence scope may name occurrence IDs")
    evidence: dict[str, Any] = {
        "trigger": trigger,
        "suggested_correction_kind": _string(suggested_correction_kind, "suggested_correction_kind"),
        "applicable_subject_ids": list(subjects),
        "base_revision_id": base_revision_id,
        "base_artifact_ids": list(bases),
        "scope": scope,
    }
    if occurrence_ids:
        evidence["occurrence_ids"] = list(occurrence_ids)
    return Finding(kind="review_request", severity="warning", stage=stage, subject_refs=subjects, evidence=evidence, message=message, dependency_ids=bases, requires_action=False)


def review_requests_for(
    *, assessment: ConfidenceAssessment | None = None, degraded_or_fallback: bool = False,
    ambiguity: str | None = None, reconciliation_issue: str | None = None,
    validation_error: bool = False, stage: str | None = None, subject_ids: Sequence[str] = (),
    suggested_correction_kind: str = "target_segment", base_revision_id: str = "",
    base_artifact_ids: Sequence[str] = (), scope: str = "segment", occurrence_ids: Sequence[str] = (),
) -> tuple[Finding, ...]:
    """Apply every Task-1 trigger. Returns findings only; never changes control flow."""
    triggers: list[str] = []
    if assessment is not None:
        if assessment.score is not None and assessment.score < .8:
            triggers.append("low_confidence")
        fallback_signal = {
            "degraded", "fallback", "parser_fallback", "model_fallback", "diagnostic_placeholder",
            "malformed_output", "retry_exhaustion",
        }
        if assessment.score is None and (degraded_or_fallback or bool(set(assessment.signals) & fallback_signal)):
            triggers.append("degraded_unknown_confidence")
    if ambiguity is not None:
        trigger = f"{ambiguity}_ambiguity" if not ambiguity.endswith("_ambiguity") else ambiguity
        triggers.append(trigger)
    if reconciliation_issue is not None:
        triggers.append(reconciliation_issue)
    if validation_error:
        triggers.append("validation_error")
    actual_stage = stage or (assessment.producing_stage if assessment is not None else "")
    return tuple(review_request(trigger=trigger, stage=actual_stage, subject_ids=subject_ids, suggested_correction_kind=suggested_correction_kind, base_revision_id=base_revision_id, base_artifact_ids=base_artifact_ids, scope=scope, occurrence_ids=occurrence_ids) for trigger in sorted(set(triggers)))


# ---------------------------------------------------------------------------
# Legacy migration readers. New state must not use these permissive schemas.
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@dataclass
class SourceBlock:
    id: str
    type: str
    text: str
    reading_order: int
    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict): return cls(**d)
    def to_file(self, path: Path) -> None: _write_json(path, self.to_dict())
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class TermMention:
    term: str
    block_id: str
    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict): return cls(**d)
    def to_file(self, path: Path) -> None: _write_json(path, self.to_dict())
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class PageExtraction:
    page_number: int
    image_path: str
    sha256: str
    phash: str
    source_lang: str
    model: str
    timestamp: str = ""
    blocks: list[SourceBlock] = field(default_factory=list)
    term_mentions: list[TermMention] = field(default_factory=list)
    illustrations: list[str] = field(default_factory=list)
    def __post_init__(self) -> None:
        if not self.timestamp: self.timestamp = datetime.now(timezone.utc).isoformat()
    @classmethod
    def from_dict(cls, d: dict):
        values = d.copy(); values["blocks"] = [SourceBlock.from_dict(x) for x in values.get("blocks", [])]; values["term_mentions"] = [TermMention.from_dict(x) for x in values.get("term_mentions", [])]; values.setdefault("illustrations", []); values.setdefault("timestamp", ""); return cls(**values)
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class TranslatedBlock:
    block_id: str
    translated_text: str
    @classmethod
    def from_dict(cls, d: dict): return cls(**d)
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class TerminologyEntry:
    concept_id: str
    source_terms: list[str]
    target_term: str
    provenance: list[str]
    confidence: float
    notes: str = ""
    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict): return cls(**d)
    def to_file(self, path: Path) -> None: _write_json(path, self.to_dict())
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class TerminologyMap:
    version: str
    hash: str
    source_lang: str
    target_lang: str
    entries: list[TerminologyEntry]
    created_at: str = ""
    def __post_init__(self) -> None:
        if not self.created_at: self.created_at = datetime.now(timezone.utc).isoformat()
    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict):
        values = d.copy(); values["entries"] = [TerminologyEntry.from_dict(x) for x in values.get("entries", [])]; values.setdefault("created_at", ""); return cls(**values)
    def to_file(self, path: Path) -> None: _write_json(path, self.to_dict())
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class Manifest:
    input_dir: str
    pages: list[dict]
    total_pages: int
    @classmethod
    def from_dict(cls, d: dict): return cls(**d)
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class PageResult:
    page_number: int
    sha256: str
    phash: str
    image_path: str = ""
    source_lang: str = ""
    target_lang: str = ""
    page_text: str = ""
    translated_text: str = ""
    image_descriptions: list[str] = field(default_factory=list)
    model: str = ""
    timestamp: str = ""
    retry_count: int = 0
    blocks: list[SourceBlock] = field(default_factory=list)
    translated_blocks: list[TranslatedBlock] = field(default_factory=list)
    term_mentions: list[TermMention] = field(default_factory=list)
    illustrations: list[str] = field(default_factory=list)
    def __post_init__(self):
        if not self.timestamp: self.timestamp = datetime.now(timezone.utc).isoformat()
    @classmethod
    def from_dict(cls, d: dict):
        values = d.copy()
        for key, default in {"image_descriptions": [], "model": "", "timestamp": "", "retry_count": 0, "blocks": [], "translated_blocks": [], "term_mentions": [], "illustrations": []}.items(): values.setdefault(key, default)
        values["blocks"] = [SourceBlock.from_dict(x) for x in values["blocks"]]; values["translated_blocks"] = [TranslatedBlock.from_dict(x) for x in values["translated_blocks"]]; values["term_mentions"] = [TermMention.from_dict(x) for x in values["term_mentions"]]; return cls(**values)
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))


@dataclass
class ErrorResult:
    page_number: int
    image_path: str = ""
    error: str = ""
    retry_count: int = 0
    model: str = ""
    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict):
        values = d.copy()
        for key, default in {"image_path": "", "retry_count": 0, "model": ""}.items(): values.setdefault(key, default)
        return cls(**values)
    def to_file(self, path: Path) -> None: _write_json(path, self.to_dict())
    @classmethod
    def from_file(cls, path: Path): return cls.from_dict(_read_json(path))
