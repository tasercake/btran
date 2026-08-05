"""Legacy manifest migration helpers plus raw-hash book discovery."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from btran.artifacts import (
    ArtifactEnvelope,
    ArtifactError,
    ArtifactStore,
    DependencyGraphEdge,
    LegacyReadOnlyArtifactStore,
    LegacyRevisionStore,
    RevisionStore,
    artifact_id_for,
    dependency_edge_id_for,
    V2RevisionStore,
)
from btran.identity import (
    PagePlacement,
    book_record_for_pages,
    page_id_for_raw_sha256,
    page_record_for_raw_sha256,
    raw_file_sha256,
    reconcile_book_pages,
)
from btran.schema import (
    EffectivePage,
    EffectiveSegment,
    BookRecord,
    Finding,
    Manifest,
    PageRecord,
    RevisionSnapshot,
    SchemaError,
    canonical_json,
    canonical_json_bytes,
)


MANIFEST_FILENAME = "manifest.json"
DISCOVERY_FILENAME = "book-discovery.json"
DISCOVERY_VERSION = "book-discovery-v1"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic"})
_SOURCE_PAGE_CACHE_LEAF_KINDS = frozenset({
    "RawSourceExtraction", "DiagnosticSourceFallback", "EffectiveSourcePage",
    "DiagnosticEffectiveSourcePage",
})
_TRANSLATION_SEGMENT_CACHE_LEAF_KINDS = frozenset({
    "TranslationArtifact", "DiagnosticTranslationFallback", "EffectiveTargetSegment",
    "DiagnosticEffectiveTargetSegment",
})


class ManifestValidationError(ValueError):
    """Raised when a legacy manifest cannot safely identify its input pages."""


@dataclass(frozen=True)
class InvocationFailure:
    """Safe invocation-boundary diagnostic for all input filesystem failures."""

    code: str
    path: str
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if self.code != "input_access":
            raise ValueError("discovery invocation failures must use input_access")

    @classmethod
    def input_access(cls, path: Path | str, error: BaseException) -> "InvocationFailure":
        # Do not retain traceback or chained exception objects.  Callers can
        # persist/print this small machine-readable diagnostic safely.
        return cls("input_access", str(path), type(error).__name__, str(error))

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class DiscoveredPage:
    """Physical placement and logical raw-hash page identity discovered this run."""

    page: PageRecord
    placement: PagePlacement
    reconciliation: str


@dataclass(frozen=True)
class BookDiscovery:
    """Discovery result.  A supported file is accepted without image decoding."""

    book: BookRecord | None
    pages: tuple[DiscoveredPage, ...]
    findings: tuple[Finding, ...]
    invocation_failure: InvocationFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.invocation_failure is None


class SelectedClosureError(ManifestValidationError):
    """A selected revision is not a self-contained, readable closure."""


class _FrozenDict(dict):
    """A dict-compatible recursively immutable JSON object."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("selected closure data is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __ior__(self, other: Any):
        self._immutable(other)

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenDict":
        return self


class _FrozenList(list):
    """A list-compatible recursively immutable JSON array."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("selected closure data is immutable")

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _immutable

    def __iadd__(self, other: Any):
        self._immutable(other)

    def __imul__(self, other: Any):
        self._immutable(other)

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenList":
        return self


def _deep_immutable(value: Any) -> Any:
    """Freeze JSON-shaped data while preserving dict/list accessor behavior."""
    if isinstance(value, Mapping):
        return _FrozenDict({key: _deep_immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_deep_immutable(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_immutable(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_immutable(item) for item in value)
    return value


class _ImmutableArtifactEnvelope(ArtifactEnvelope):
    """Read-only defensive view of a validated artifact envelope."""

    @property
    def __class__(self) -> type[Any]:
        # Preserve compatibility with callers checking the concrete schema type.
        return ArtifactEnvelope

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_selected_closure_sealed", False):
            raise TypeError("selected closure data is immutable")
        object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArtifactEnvelope):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in (
            "schema_version", "artifact_id", "kind", "payload", "dependency_ids",
            "finding_ids", "semantic_key",
        ))


class _ImmutableSchemaRecord:
    """Mixin for a sealed, compatibility-preserving schema-record view."""

    @property
    def __class__(self) -> type[Any]:
        # Keep legacy ``record.__class__ is Finding``-style checks working.
        return type(self)._immutable_view_base

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_selected_closure_sealed", False):
            raise TypeError("selected closure data is immutable")
        object.__setattr__(self, name, value)


_IMMUTABLE_SCHEMA_RECORD_TYPES: dict[type[Any], type[Any]] = {}


def _immutable_schema_record(record: Any) -> Any:
    """Copy a parsed schema record into a sealed view without revalidation."""
    if isinstance(record, _ImmutableSchemaRecord):
        return record
    record_type = type(record)
    view_type = _IMMUTABLE_SCHEMA_RECORD_TYPES.get(record_type)
    if view_type is None:
        view_type = type(
            f"_Immutable{record_type.__name__}",
            (_ImmutableSchemaRecord, record_type),
            {"__module__": record_type.__module__, "_immutable_view_base": record_type},
        )
        _IMMUTABLE_SCHEMA_RECORD_TYPES[record_type] = view_type
    view = object.__new__(view_type)
    for item in fields(record):
        object.__setattr__(view, item.name, _deep_immutable(getattr(record, item.name)))
    object.__setattr__(view, "_selected_closure_sealed", True)
    return view


def _immutable_artifact(record: ArtifactEnvelope) -> ArtifactEnvelope:
    """Copy a validated record into a read-only view without validating again."""
    view = object.__new__(_ImmutableArtifactEnvelope)
    for item in fields(record):
        object.__setattr__(view, item.name, _deep_immutable(getattr(record, item.name)))
    object.__setattr__(view, "_selected_closure_sealed", True)
    return view


def _immutable_ordered_pages(ordered_pages: tuple[Any, ...]) -> tuple[Any, ...]:
    """Copy declared page views and children without reparsing their payloads."""
    from btran.orchestrator_contract import OrderedEffectivePage

    return tuple(
        OrderedEffectivePage(
            _immutable_schema_record(item.page),
            tuple(_immutable_schema_record(segment) for segment in item.segments),
        )
        for item in ordered_pages
    )


def _expected_selected_closure_ids(
    snapshot: RevisionSnapshot,
    records: Mapping[str, ArtifactEnvelope],
    findings: Mapping[str, Finding],
) -> tuple[set[str], set[str]]:
    """Follow selected records/findings to compute the exact archive closure."""
    record_ids = set(snapshot.selected_artifact_ids)
    finding_ids = set(snapshot.selected_finding_ids)
    changed = True
    while changed:
        changed = False
        for record_id in tuple(record_ids):
            record = records.get(record_id)
            if record is None:
                continue
            before = len(finding_ids)
            finding_ids.update(record.finding_ids)
            changed |= len(finding_ids) != before
        for finding_id in tuple(finding_ids):
            finding = findings.get(finding_id)
            if finding is None:
                continue
            before = len(record_ids)
            record_ids.update(finding.dependency_ids)
            changed |= len(record_ids) != before
        for record_id in tuple(record_ids):
            record = records.get(record_id)
            if record is None:
                continue
            before = len(record_ids)
            record_ids.update(record.dependency_ids)
            changed |= len(record_ids) != before
    return record_ids, finding_ids


def _assert_exact_selected_closure(
    snapshot: RevisionSnapshot,
    records: Mapping[str, ArtifactEnvelope],
    findings: Mapping[str, Finding],
    *,
    error_type: type[Exception],
) -> None:
    expected_records, expected_findings = _expected_selected_closure_ids(snapshot, records, findings)
    if set(records) != expected_records:
        raise error_type("selected records are not the exact archive closure")
    if set(findings) != expected_findings:
        raise error_type("selected findings are not the exact archive closure")


@dataclass(frozen=True)
class SelectedClosure:
    """In-memory authority for one selected sealed revision.

    Loading performs the archive validation once.  Every accessor below reads
    these immutable maps and tuples; it never consults the workspace again.
    This is important for both deterministic ordering and for legacy workspaces,
    which are deliberately read-only.
    """

    revision_id: str
    snapshot: RevisionSnapshot
    records: Mapping[str, ArtifactEnvelope]
    findings: Mapping[str, Finding]
    edges: Mapping[str, DependencyGraphEdge]
    attestations: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]
    # Parsed once while loading.  Accessors return these immutable views and
    # never deserialize or validate selected records again.
    _ordered_pages: tuple[Any, ...] = ()
    _effective_content: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, RevisionSnapshot):
            raise TypeError("selected closure snapshot must be a RevisionSnapshot")
        if self.snapshot.revision_id != self.revision_id:
            raise ValueError("selected closure revision ID does not match snapshot")
        # The parsed records are validation results, not mutable state owned by
        # the caller.  Copy each exposed dataclass into a sealed view so a
        # caller cannot mutate the archive authority after load.
        object.__setattr__(self, "snapshot", _immutable_schema_record(self.snapshot))
        ordered_pages = _immutable_ordered_pages(self._ordered_pages)
        object.__setattr__(self, "_ordered_pages", ordered_pages)
        if self._effective_content is not None:
            from btran.orchestrator_contract import SelectedEffectiveContent
            if isinstance(self._effective_content, SelectedEffectiveContent):
                object.__setattr__(
                    self, "_effective_content",
                    SelectedEffectiveContent(
                        ordered_pages,
                        finding_ids=self._effective_content.finding_ids,
                    ),
                )
        maps = {
            "records": self.records, "findings": self.findings,
            "edges": self.edges, "attestations": self.attestations,
            "provenance": self.provenance,
        }
        for name, value in maps.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"selected closure {name} must be a mapping")
            if name == "records":
                value = {
                    key: item if isinstance(item, _ImmutableArtifactEnvelope) else _immutable_artifact(item)
                    for key, item in value.items()
                }
            elif name in {"findings", "edges"}:
                value = {key: _immutable_schema_record(item) for key, item in value.items()}
            elif name in {"attestations", "provenance"}:
                value = _deep_immutable(value)
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        # The selected archive is the authority.  Do not silently broaden it
        # from a mutable index or from historical records.  Finding
        # dependencies are part of the closure too, even when no selected
        # record points at them directly.
        _assert_exact_selected_closure(
            self.snapshot, self.records, self.findings, error_type=ValueError,
        )

    @classmethod
    def empty(cls) -> "SelectedClosure":
        """Return the explicit empty closure used before the first revision."""
        snapshot = RevisionSnapshot(revision_id="unsealed")
        from btran.orchestrator_contract import SelectedEffectiveContent
        content = SelectedEffectiveContent()
        return cls("unsealed", snapshot, {}, {}, {}, {}, {}, (), content)

    @property
    def artifact_map(self) -> Mapping[str, ArtifactEnvelope]:
        return self.records

    @property
    def record_map(self) -> Mapping[str, ArtifactEnvelope]:
        return self.records

    @property
    def finding_map(self) -> Mapping[str, Finding]:
        return self.findings

    @property
    def edge_map(self) -> Mapping[str, DependencyGraphEdge]:
        return self.edges

    @property
    def attestation_map(self) -> Mapping[str, Mapping[str, Any]]:
        return self.attestations

    @property
    def final_finding_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.findings))

    @property
    def finding_ids(self) -> tuple[str, ...]:
        return self.final_finding_ids

    def record(self, record_id: str) -> ArtifactEnvelope:
        try:
            return self.records[record_id]
        except KeyError as exc:
            raise SelectedClosureError(f"selected record is missing: {record_id}") from exc

    def finding(self, finding_id: str) -> Finding:
        try:
            return self.findings[finding_id]
        except KeyError as exc:
            raise SelectedClosureError(f"selected finding is missing: {finding_id}") from exc

    def edge(self, edge_id: str) -> DependencyGraphEdge:
        try:
            return self.edges[edge_id]
        except KeyError as exc:
            raise SelectedClosureError(f"selected edge is missing: {edge_id}") from exc

    @staticmethod
    def _payload_id(record: ArtifactEnvelope, field: str) -> str | None:
        value = record.payload.get(field)
        return value if isinstance(value, str) and value else None

    def _typed_records(self, kinds: set[str]) -> tuple[ArtifactEnvelope, ...]:
        return tuple(record for record in self.records.values() if record.kind in kinds)

    @property
    def ordered_effective_pages(self) -> tuple[Any, ...]:
        """Return the pages in the order declared by the selected revision."""
        return self._ordered_pages

    @property
    def selected_effective_content(self) -> Any:
        return self._effective_content

    @property
    def ordered_effective_segments(self) -> tuple[Any, ...]:
        """Return selected effective segments in declared page order."""
        return tuple(segment for page in self._ordered_pages for segment in page.segments)

    @property
    def source_page_cache_leaves(self) -> tuple[ArtifactEnvelope, ...]:
        return tuple(record for record in self.records.values() if record.kind in _SOURCE_PAGE_CACHE_LEAF_KINDS)

    @property
    def translation_segment_cache_leaves(self) -> tuple[ArtifactEnvelope, ...]:
        return tuple(record for record in self.records.values() if record.kind in _TRANSLATION_SEGMENT_CACHE_LEAF_KINDS)

    @property
    def selected_terminology_entries(self) -> tuple[ArtifactEnvelope, ...]:
        return tuple(record for record in self.records.values() if record.kind in {
            "ConceptProjection", "ConceptSelector", "TerminologyOverlay",
        })

    @property
    def selected_correction_targets(self) -> tuple[ArtifactEnvelope, ...]:
        return tuple(record for record in self.records.values() if record.kind in {
            "SourceTextOverlay", "TargetSegmentOverlay", "TargetOccurrenceOverlay",
            "TerminologyOverlay", "CorrectionRecord",
        })

    # Method spellings keep the contract usable by stages that prefer verbs.
    def source_page_leaves(self) -> tuple[ArtifactEnvelope, ...]:
        return self.source_page_cache_leaves

    def translation_segment_leaves(self) -> tuple[ArtifactEnvelope, ...]:
        return self.translation_segment_cache_leaves

    def terminology_entries(self) -> tuple[ArtifactEnvelope, ...]:
        return self.selected_terminology_entries

    def correction_targets(self) -> tuple[ArtifactEnvelope, ...]:
        return self.selected_correction_targets

    @property
    def source_page_cache_leaf_map(self) -> Mapping[str, ArtifactEnvelope]:
        return _cache_leaf_map(
            self.records, _SOURCE_PAGE_CACHE_LEAF_KINDS, "page_id", "source page",
        )

    @property
    def translation_segment_cache_leaf_map(self) -> Mapping[str, ArtifactEnvelope]:
        return _cache_leaf_map(
            self.records, _TRANSLATION_SEGMENT_CACHE_LEAF_KINDS, "segment_id", "translation segment",
        )

    @property
    def selected_terminology_entry_map(self) -> Mapping[str, ArtifactEnvelope]:
        return MappingProxyType({record.artifact_id: record for record in self.selected_terminology_entries})

    @property
    def selected_correction_target_map(self) -> Mapping[str, ArtifactEnvelope]:
        return MappingProxyType({record.artifact_id: record for record in self.selected_correction_targets})


def _canonical_member(data: bytes, name: str) -> Any:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedClosureError(f"{name} is not UTF-8 canonical JSON") from exc
    if canonical_json_bytes(value) != data:
        raise SelectedClosureError(f"{name} is not canonical JSON")
    return value


def _legacy_active_revision_id(revisions: Any) -> str | None:
    pointer = Path(revisions.root) / "active-revision.json"
    if not pointer.exists():
        return None
    try:
        body = _canonical_member(pointer.read_bytes(), "active-revision.json")
        if set(body) != {"revision_id"} or not isinstance(body["revision_id"], str):
            raise SelectedClosureError("active revision pointer is invalid")
        return body["revision_id"]
    except OSError as exc:
        raise SelectedClosureError("active revision pointer is unreadable") from exc


def _legacy_ids(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise SelectedClosureError(f"legacy {name} is invalid")
    result = tuple(sorted(set(value)))
    if len(result) != len(value) or any(not isinstance(item, str) or not item for item in result):
        raise SelectedClosureError(f"legacy {name} is invalid")
    return result


def _read_legacy_revision_once(revisions: LegacyRevisionStore, revision_id: str) -> tuple[RevisionSnapshot, Mapping[str, bytes], Mapping[str, Any]]:
    """Verify and retain a legacy bundle from one filesystem traversal.

    The old adapter's ``verify_bundle`` validates by opening every member and
    callers then opened those members a second time.  That is both unnecessary
    and observable for legacy read-only workspaces, so this compatibility
    reader keeps the bytes it validates in memory.
    """
    bundle = Path(revisions.revisions_dir) / revision_id
    if not bundle.is_dir() or bundle.is_symlink():
        raise SelectedClosureError("sealed revision bundle is missing or unsafe")
    files: dict[str, bytes] = {}
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise SelectedClosureError("sealed revision bundle contains a symlink")
        if path.is_file():
            files[path.relative_to(bundle).as_posix()] = path.read_bytes()

    def member(name: str) -> bytes:
        try:
            return files[name]
        except KeyError as exc:
            raise SelectedClosureError(f"legacy bundle member is missing: {name}") from exc

    snapshot_bytes = member("snapshot.json")
    try:
        snapshot = RevisionSnapshot.from_json(snapshot_bytes.decode("utf-8"))
        manifest = json.loads(member("bundle-manifest.json").decode("utf-8"))
        provenance = json.loads(member("provenance.json").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, SchemaError, TypeError, ValueError) as exc:
        raise SelectedClosureError("legacy revision metadata is invalid") from exc
    if snapshot.revision_id != revision_id or not isinstance(manifest, Mapping) or not isinstance(provenance, Mapping):
        raise SelectedClosureError("legacy revision metadata is invalid")
    if manifest.get("revision_id") != revision_id:
        raise SelectedClosureError("bundle manifest revision mismatch")
    if manifest.get("snapshot_sha256") != hashlib.sha256(snapshot.to_json().encode("utf-8")).hexdigest():
        raise SelectedClosureError("bundle snapshot hash mismatch")
    if manifest.get("provenance_sha256") != hashlib.sha256(canonical_json_bytes(dict(provenance))).hexdigest():
        raise SelectedClosureError("bundle provenance hash mismatch")

    epub_filename = manifest.get("epub_filename")
    if not isinstance(epub_filename, str) or Path(epub_filename).name != epub_filename:
        raise SelectedClosureError("bundle EPUB filename is invalid")
    epub_bytes = member(epub_filename)
    if manifest.get("epub_sha256") != hashlib.sha256(epub_bytes).hexdigest():
        raise SelectedClosureError("bundle EPUB hash mismatch")
    try:
        LegacyRevisionStore._verify_embedded_provenance(epub_bytes, provenance)
    except ArtifactError as exc:
        raise SelectedClosureError("bundle EPUB provenance is invalid") from exc

    artifact_ids = _legacy_ids(manifest.get("artifact_ids", ()), "artifact_ids")
    finding_ids = _legacy_ids(manifest.get("finding_ids", ()), "finding_ids")
    attestation_ids = _legacy_ids(manifest.get("semantic_attestation_ids", ()), "semantic_attestation_ids")
    edge_ids = _legacy_ids(manifest.get("edge_ids", ()), "edge_ids")
    values: dict[str, bytes] = {"snapshot.json": snapshot_bytes}
    records: dict[str, ArtifactEnvelope] = {}
    findings: dict[str, Finding] = {}
    attestations: dict[str, Mapping[str, Any]] = {}
    for artifact_id in artifact_ids:
        name = f"artifacts/{artifact_id}.json"
        data = member(name)
        try:
            record = ArtifactEnvelope.from_json(data.decode("utf-8"))
        except (UnicodeDecodeError, SchemaError) as exc:
            raise SelectedClosureError("bundle artifact is missing or invalid") from exc
        if record.artifact_id != artifact_id or artifact_id_for(record.kind, record.payload, record.dependency_ids) != artifact_id:
            raise SelectedClosureError("bundle artifact content hash mismatch")
        records[artifact_id] = record
        values[name] = data
    for finding_id in finding_ids:
        name = f"findings/{finding_id}.json"
        data = member(name)
        try:
            finding = Finding.from_json(data.decode("utf-8"))
        except (UnicodeDecodeError, SchemaError) as exc:
            raise SelectedClosureError("bundle finding is missing or invalid") from exc
        if finding.finding_id != finding_id:
            raise SelectedClosureError("bundle finding ID mismatch")
        findings[finding_id] = finding
        values[name] = data
    for attestation_id in attestation_ids:
        name = f"attestations/{attestation_id}.json"
        data = member(name)
        body = _canonical_member(data, name)
        required = {"attestation_id", "artifact_id", "kind", "semantic_key", "dependency_ids"}
        if not isinstance(body, Mapping) or set(body) != required or body["attestation_id"] != attestation_id:
            raise SelectedClosureError("bundle semantic attestation is invalid")
        try:
            expected = LegacyArtifactStore.semantic_attestation_id_for(
                artifact_id=body["artifact_id"], kind=body["kind"],
                semantic_key=body["semantic_key"], dependency_ids=body["dependency_ids"],
            )
        except (ArtifactError, TypeError, KeyError) as exc:
            raise SelectedClosureError("bundle semantic attestation is invalid") from exc
        if expected != attestation_id or body["artifact_id"] not in records:
            raise SelectedClosureError("bundle semantic attestation closure mismatch")
        envelope = records[body["artifact_id"]]
        if body["kind"] != envelope.kind or tuple(body["dependency_ids"]) != envelope.dependency_ids:
            raise SelectedClosureError("bundle semantic attestation closure mismatch")
        attestations[attestation_id] = body
        values[name] = data
    for edge_id in edge_ids:
        name = f"edges/{edge_id}.json"
        data = member(name)
        try:
            edge = DependencyGraphEdge.from_json(data.decode("utf-8"))
        except (UnicodeDecodeError, SchemaError) as exc:
            raise SelectedClosureError("bundle graph edge is missing or invalid") from exc
        if edge.edge_id != edge_id or dependency_edge_id_for(edge.stable_subject_id, edge.parent_artifact_id, edge.child_artifact_id, edge.stage, edge.edge_kind) != edge_id:
            raise SelectedClosureError("bundle graph edge hash mismatch")
        values[name] = data
    # Preserve the old adapter's closure guarantees while still using retained
    # bytes for every subsequent load operation.
    if not set(snapshot.selected_artifact_ids).issubset(records) or not set(snapshot.selected_finding_ids).issubset(findings):
        raise SelectedClosureError("bundle omits selected snapshot IDs")
    for record in records.values():
        if not set(record.dependency_ids).issubset(records) or not set(record.finding_ids).issubset(findings):
            raise SelectedClosureError("bundle artifact closure is incomplete")
    for finding in findings.values():
        if not set(finding.dependency_ids).issubset(records):
            raise SelectedClosureError("bundle finding closure is incomplete")
    render_input_id = manifest.get("render_input_artifact_id")
    render_input_hash = manifest.get("render_input_hash")
    if render_input_id is None:
        if render_input_hash is not None:
            raise SelectedClosureError("bundle render-input hash has no input")
    elif render_input_id not in records or render_input_hash != hashlib.sha256(
            canonical_json_bytes(records[render_input_id].to_dict())).hexdigest():
        raise SelectedClosureError("bundle render-input hash mismatch")
    return snapshot, values, provenance


def _revision_members(revisions: Any, revision_id: str) -> tuple[RevisionSnapshot, Mapping[str, bytes], Mapping[str, Any]]:
    """Validate one revision while retaining the validated member bytes."""
    try:
        if isinstance(revisions, V2RevisionStore):
            values = revisions.storage.verify_revision(revision_id)
            snapshot = RevisionSnapshot.from_json(values["snapshot.json"].decode("utf-8"))
            return snapshot, values, {}
        if isinstance(revisions, LegacyRevisionStore):
            return _read_legacy_revision_once(revisions, revision_id)
        raise TypeError("unsupported revision store")
    except (OSError, UnicodeDecodeError, SchemaError, ArtifactError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SelectedClosureError):
            raise
        raise SelectedClosureError(f"cannot load selected revision {revision_id}") from exc


_STABLE_ID_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "PageRecord": ("page_id",),
    "BookRecord": ("book_id",),
    "Segment": ("segment_id",),
    "RawSourceSegment": ("segment_id",),
    "TermOccurrence": ("occurrence_id",),
    "OccurrenceEvidence": ("occurrence_id",),
    "OccurrenceEvidenceShard": (),  # shard IDs are envelope identities
    "TerminologyConcept": ("concept_id",),
    "ConceptMembership": ("concept_id", "membership_id"),
    "ConceptProjection": ("projection_id",),
    "ConceptSelector": ("selector_id",),
    "TranslationArtifact": ("translation_artifact_id",),
    "DiagnosticTranslationFallback": ("translation_artifact_id",),
    "RawSourceExtraction": ("page_id",),
    "DiagnosticSourceFallback": ("page_id",),
    "OccurrenceTargetMapping": ("mapping_id",),
    "EffectiveSourceSegment": ("effective_segment_id", "segment_id"),
    "DiagnosticEffectiveSourceSegment": ("effective_segment_id", "segment_id"),
    "EffectiveTargetSegment": ("effective_segment_id", "segment_id"),
    "DiagnosticEffectiveTargetSegment": ("effective_segment_id", "segment_id"),
    "EffectiveSourcePage": ("effective_page_id", "page_id"),
    "EffectiveTargetPage": ("effective_page_id", "page_id"),
    "CorrectionRecord": ("correction_id",),
    "CorrectionEvent": ("event_id",),
    "CorrectionImpact": ("projection_plan_id",),
    "ExtractionSemantics": ("page_id",),
}


def _validate_stable_identities(records: Mapping[str, ArtifactEnvelope]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for record in records.values():
        for field in _STABLE_ID_FIELDS_BY_KIND.get(record.kind, ()):
            value = record.payload.get(field)
            if not isinstance(value, str) or not value:
                continue
            key = (record.kind, field, value)
            if key in seen:
                raise SelectedClosureError(f"duplicate selected stable identity: {field}")
            seen.add(key)


def _cache_leaf_map(
    records: Mapping[str, ArtifactEnvelope],
    kinds: frozenset[str],
    field: str,
    label: str,
) -> Mapping[str, ArtifactEnvelope]:
    """Build a cache map without allowing identity collisions to overwrite."""
    result: dict[str, ArtifactEnvelope] = {}
    for record in sorted(records.values(), key=lambda item: (item.kind, item.artifact_id)):
        if record.kind not in kinds:
            continue
        stable_id = record.payload.get(field)
        if not isinstance(stable_id, str) or not stable_id:
            continue
        if stable_id in result:
            raise SelectedClosureError(f"duplicate {label} cache identity: {stable_id}")
        result[stable_id] = record
    return MappingProxyType(result)


def _validate_cache_leaf_identities(records: Mapping[str, ArtifactEnvelope]) -> None:
    """Reject same-key leaves across every source/translation cache kind."""
    _cache_leaf_map(records, _SOURCE_PAGE_CACHE_LEAF_KINDS, "page_id", "source page")
    _cache_leaf_map(
        records, _TRANSLATION_SEGMENT_CACHE_LEAF_KINDS, "segment_id", "translation segment",
    )


def _build_ordered_pages(records: Mapping[str, ArtifactEnvelope], provenance: Mapping[str, Any]) -> tuple[Any, ...]:
    """Deserialize and validate effective pages and their declared children once."""
    all_records = tuple(records.values())
    has_target = any(record.kind == "EffectiveTargetPage" for record in all_records)
    # A translated revision retains its source pages and segments as immutable
    # dependencies of the target materialization.  They are cache/provenance
    # leaves, not the selected render content.  Prefer target pages whenever
    # the closure contains them; source-only revisions still use source pages.
    page_kinds = {"EffectiveTargetPage"} if has_target else {"EffectiveSourcePage"}
    pages_by_artifact: dict[str, tuple[EffectivePage, ArtifactEnvelope]] = {}
    effective_segments: dict[str, tuple[EffectiveSegment, ArtifactEnvelope]] = {}
    for record in all_records:
        if record.kind in page_kinds:
            try:
                page = EffectivePage.from_dict(record.payload)
            except (SchemaError, TypeError, ValueError) as exc:
                raise SelectedClosureError("selected effective page is invalid") from exc
            pages_by_artifact[record.artifact_id] = (page, record)
        elif record.kind in ({"EffectiveTargetSegment", "DiagnosticEffectiveTargetSegment"} if has_target else {"EffectiveSourceSegment", "DiagnosticEffectiveSourceSegment"}):
            try:
                segment = EffectiveSegment.from_dict(record.payload)
            except (SchemaError, TypeError, ValueError) as exc:
                raise SelectedClosureError("selected effective segment is invalid") from exc
            effective_segments[segment.effective_segment_id] = (segment, record)

    declared_segment_ids: set[str] = set()
    parsed: dict[str, Any] = {}
    from btran.orchestrator_contract import OrderedEffectivePage
    for artifact_id, (page, page_record) in pages_by_artifact.items():
        ids = page.effective_segment_ids
        if len(set(ids)) != len(ids):
            raise SelectedClosureError("effective page declares duplicate children")
        declared_segment_ids.update(ids)
        children: list[EffectiveSegment] = []
        child_artifact_ids: set[str] = set()
        for segment_id in ids:
            if segment_id not in effective_segments:
                raise SelectedClosureError("effective page declares a missing child")
            segment, segment_record = effective_segments[segment_id]
            children.append(segment)
            child_artifact_ids.add(segment_record.artifact_id)
        # Dependency IDs are the persisted page→child relationship.  Exact
        # equality rejects both omitted declared children and extra children.
        if set(page_record.dependency_ids) != child_artifact_ids:
            raise SelectedClosureError("effective page child relationship is invalid")
        parsed[artifact_id] = OrderedEffectivePage(page, tuple(children))
    if set(effective_segments) != declared_segment_ids:
        raise SelectedClosureError("selected effective segments are not declared page children")

    if not pages_by_artifact:
        return ()
    placements = provenance.get("placements") if isinstance(provenance, Mapping) else None
    ordered_artifact_ids: tuple[str, ...]
    if placements is not None:
        if not isinstance(placements, (list, tuple)):
            raise SelectedClosureError("revision placements are invalid")
        ordered: list[str] = []
        seen_page_ids: set[str] = set()
        for placement in placements:
            if not isinstance(placement, Mapping):
                raise SelectedClosureError("revision placement is invalid")
            page_id = placement.get("page_id")
            artifact_id = placement.get("effective_page_artifact_id")
            if not isinstance(page_id, str) or not isinstance(artifact_id, str) or artifact_id not in parsed:
                raise SelectedClosureError("revision placement references an unknown page")
            page = pages_by_artifact[artifact_id][0]
            if page.page_id != page_id or page_id in seen_page_ids:
                raise SelectedClosureError("revision placements are duplicate or inconsistent")
            seen_page_ids.add(page_id)
            ordered.append(artifact_id)
        if set(ordered) != set(parsed) or len(ordered) != len(parsed):
            raise SelectedClosureError("revision placements omit or add a page")
        ordered_artifact_ids = tuple(ordered)
    else:
        numbers = [page.display_metadata.get("page_number") for page, _ in pages_by_artifact.values()]
        if len(pages_by_artifact) > 1 and not all(isinstance(number, int) and not isinstance(number, bool) for number in numbers):
            raise SelectedClosureError("selected pages have no declared order")
        if len(pages_by_artifact) == 1:
            ordered_artifact_ids = tuple(parsed)
        else:
            if len(set(numbers)) != len(numbers) or set(numbers) != set(range(1, len(numbers) + 1)):
                raise SelectedClosureError("selected page order is invalid")
            ordered_artifact_ids = tuple(artifact_id for artifact_id, _ in sorted(
                pages_by_artifact.items(), key=lambda item: item[1][0].display_metadata["page_number"]))
    return tuple(parsed[artifact_id] for artifact_id in ordered_artifact_ids)


def load_selected_closure(
    revisions: RevisionStore | Path | str,
    revision_id: str | None = None,
) -> SelectedClosure:
    """Load and validate the selected revision exactly once.

    ``revision_id`` is explicit whenever a revision is selected.  Omitting it
    reads only the active pointer; it never searches revision history.  A
    missing active revision is the explicit empty pre-first-run closure.
    """
    if isinstance(revisions, (str, Path)):
        revisions = RevisionStore(revisions)
    if not isinstance(revisions, (V2RevisionStore,)) and not hasattr(revisions, "verify_bundle"):
        raise TypeError("revisions must be a RevisionStore or workspace path")
    if revision_id is None:
        if isinstance(revisions, V2RevisionStore):
            revision_id = revisions.storage.active_revision_id()
        else:
            revision_id = _legacy_active_revision_id(revisions)
    if revision_id is None or revision_id == "unsealed":
        return SelectedClosure.empty()

    snapshot, values, provenance = _revision_members(revisions, revision_id)
    records: dict[str, ArtifactEnvelope] = {}
    findings: dict[str, Finding] = {}
    edges: dict[str, DependencyGraphEdge] = {}
    attestations: dict[str, Mapping[str, Any]] = {}
    for name, data in values.items():
        if name.startswith("records/") or name.startswith("artifacts/"):
            try:
                record = ArtifactEnvelope.from_json(data.decode("utf-8"))
            except (UnicodeDecodeError, SchemaError) as exc:
                raise SelectedClosureError(f"invalid selected record: {name}") from exc
            if record.artifact_id in records:
                raise SelectedClosureError("duplicate selected record identity")
            records[record.artifact_id] = record
        elif name.startswith("findings/"):
            try:
                finding = Finding.from_json(data.decode("utf-8"))
            except (UnicodeDecodeError, SchemaError) as exc:
                raise SelectedClosureError(f"invalid selected finding: {name}") from exc
            if finding.finding_id in findings:
                raise SelectedClosureError("duplicate selected finding identity")
            findings[finding.finding_id] = finding
        elif name.startswith("edges/") or name.startswith("graph/"):
            try:
                edge = DependencyGraphEdge.from_json(data.decode("utf-8"))
            except (UnicodeDecodeError, SchemaError) as exc:
                raise SelectedClosureError(f"invalid selected edge: {name}") from exc
            if edge.edge_id in edges:
                raise SelectedClosureError("duplicate selected edge identity")
            edges[edge.edge_id] = edge
        elif name.startswith("attestations/"):
            body = _canonical_member(data, name)
            if not isinstance(body, Mapping) or body.get("attestation_id") in attestations:
                raise SelectedClosureError("duplicate selected attestation identity")
            attestations[body["attestation_id"]] = MappingProxyType(dict(body))

    if snapshot.revision_id != revision_id:
        raise SelectedClosureError("selected revision ID does not match snapshot")
    _assert_exact_selected_closure(
        snapshot, records, findings, error_type=SelectedClosureError,
    )
    if not set(snapshot.selected_cache_attestation_ids).issubset(attestations):
        raise SelectedClosureError("selected attestations are missing")
    for record in records.values():
        if not set(record.dependency_ids).issubset(records) or not set(record.finding_ids).issubset(findings):
            raise SelectedClosureError("selected record relationship escapes closure")
    for finding in findings.values():
        if not set(finding.dependency_ids).issubset(records):
            raise SelectedClosureError("selected finding relationship escapes closure")
    for body in attestations.values():
        artifact_id = body.get("artifact_id")
        if artifact_id not in records:
            raise SelectedClosureError("selected attestation relationship escapes closure")
        if body.get("kind") != records[artifact_id].kind or tuple(body.get("dependency_ids", ())) != records[artifact_id].dependency_ids:
            raise SelectedClosureError("selected attestation does not bind its record")
    for edge in edges.values():
        if edge.parent_artifact_id not in records or edge.child_artifact_id not in records:
            raise SelectedClosureError("selected edge relationship escapes closure")

    _validate_stable_identities(records)
    _validate_cache_leaf_identities(records)
    ordered_pages = _build_ordered_pages(records, provenance)
    from btran.orchestrator_contract import SelectedEffectiveContent
    effective_content = SelectedEffectiveContent(ordered_pages, finding_ids=tuple(sorted(findings)))
    return SelectedClosure(
        revision_id, snapshot, records, findings, edges, attestations, provenance,
        ordered_pages, effective_content,
    )


# Compatibility spelling used by callers that treat this as a store method.
selected_closure = load_selected_closure


def _input_failure(path: Path | str, error: BaseException) -> BookDiscovery:
    return BookDiscovery(None, (), (), InvocationFailure.input_access(path, error))


def _finding_page_missing(page: PageRecord) -> Finding:
    return Finding(
        kind="page_missing", severity="warning", stage="discovery",
        subject_refs=(page.page_id,),
        evidence={"raw_file_sha256": page.raw_file_sha256},
        message="Previously discovered page is absent from current input.",
    )


def _read_discovery_history(workspace: Path) -> tuple[PageRecord, ...]:
    """Read only retained identities; old artifacts/corrections are never removed."""
    path = workspace / DISCOVERY_FILENAME
    if not path.exists():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {"schema_version", "book", "known_pages", "placements", "finding_ids"}:
            raise ManifestValidationError("discovery history has invalid fields")
        if raw["schema_version"] != DISCOVERY_VERSION:
            raise ManifestValidationError("unsupported discovery history version")
        BookRecord.from_dict(raw["book"])
        records = tuple(PageRecord.from_dict(item) for item in raw["known_pages"])
    except (OSError, TypeError, KeyError, json.JSONDecodeError, SchemaError) as exc:
        raise ManifestValidationError("discovery history is invalid") from exc
    if tuple(sorted({record.page_id for record in records})) != tuple(record.page_id for record in records):
        raise ManifestValidationError("discovery known pages must be sorted and unique")
    if any(record.page_id != page_id_for_raw_sha256(record.raw_file_sha256) for record in records):
        raise ManifestValidationError("discovery known page identity does not match raw hash")
    return records


def _is_legacy_discovery_workspace(workspace: Path) -> bool:
    """Return whether discovery must use the legacy read-only contract.

    ``ArtifactStore`` cannot be constructed first: its v2 adapter creates the
    SQLite state file.  An old discovery snapshot is itself legacy state, even
    when the workspace predates the old artifact directories.  A v2 workspace
    is unambiguously identified by its state database and remains writable.
    """
    if (workspace / "state-v2.sqlite3").exists():
        return False
    if any((workspace / name).exists() for name in (
        DISCOVERY_FILENAME, MANIFEST_FILENAME, "artifacts", "findings", "index",
        "attestations", "graph", "active-revision.json",
    )):
        return True
    revisions = workspace / "revisions"
    return revisions.is_dir() and any(path.is_dir() for path in revisions.iterdir())


def _persist_discovery(
    workspace: Path,
    book: BookRecord,
    pages: tuple[DiscoveredPage, ...],
    historical_pages: tuple[PageRecord, ...],
    findings: tuple[Finding, ...],
) -> None:
    # A legacy workspace is read/verify-only.  Check this before constructing
    # ArtifactStore: constructing the v2 adapter creates state-v2.sqlite3.
    if _is_legacy_discovery_workspace(workspace):
        return
    store = ArtifactStore(workspace)
    if isinstance(store, LegacyReadOnlyArtifactStore):
        return
    workspace.mkdir(parents=True, exist_ok=True)
    # Persist findings independently through Task 3 immutable storage, then
    # retain IDs in the small discovery snapshot for inspectability.
    for finding in findings:
        store.put_finding(finding)
    known = {page.page_id: page for page in historical_pages}
    known.update({item.page.page_id: item.page for item in pages})
    body: dict[str, Any] = {
        "schema_version": DISCOVERY_VERSION,
        "book": book.to_dict(),
        "known_pages": [known[page_id].to_dict() for page_id in sorted(known)],
        "placements": [
            {
                "page_id": item.page.page_id,
                "raw_file_sha256": item.page.raw_file_sha256,
                "relative_path": item.placement.relative_path,
                "placement_id": item.placement.placement_id,
                "reconciliation": item.reconciliation,
            }
            for item in pages
        ],
        "finding_ids": sorted(finding.finding_id for finding in findings),
    }
    target = workspace / DISCOVERY_FILENAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_bytes(canonical_json(body).encode("utf-8"))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def discover_book(input_dir: Path | str, workspace: Path | str | None = None) -> BookDiscovery:
    """Discover supported files by raw bytes, never filename/page number/decode.

    Every input access error is converted to ``InvocationFailure(input_access)``.
    Decoding is intentionally not attempted: an unreadable-but-supported image is
    accepted here and becomes a typed page fallback in the extraction task.
    """
    requested = Path(input_dir)
    try:
        directory = requested.resolve()
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))
        entries = list(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
        return _input_failure(requested, exc)

    raw_pages: list[tuple[Path, str]] = []
    for entry in sorted(entries, key=lambda candidate: candidate.name):
        try:
            # Do not use ``Path.is_file`` here: it converts a failed stat into
            # False, silently omitting a requested input page.
            entry_stat = entry.stat()
            if not stat.S_ISREG(entry_stat.st_mode) or entry.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            # Read arbitrary bytes, including files image libraries later
            # cannot decode.
            raw_pages.append((entry, raw_file_sha256(entry.read_bytes())))
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
            return _input_failure(entry, exc)

    root = Path(workspace) if workspace is not None else None
    try:
        historical = _read_discovery_history(root) if root is not None else ()
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
        # A workspace error is not an input error.  Invocation boundary owns it.
        raise ManifestValidationError("discovery workspace is inaccessible") from exc

    reconciliations = reconcile_book_pages([digest for _, digest in raw_pages], historical)
    pages: list[DiscoveredPage] = []
    findings: list[Finding] = []
    for (path, digest), reconciliation in zip(raw_pages, reconciliations, strict=True):
        if reconciliation.finding is not None:
            findings.append(reconciliation.finding)
        page = (
            next(item for item in historical if item.page_id == reconciliation.page_id)
            if reconciliation.status == "reused" else page_record_for_raw_sha256(digest)
        )
        relative = path.relative_to(directory).as_posix()
        pages.append(DiscoveredPage(page, PagePlacement.create(page.page_id, digest, relative), reconciliation.status))

    current_ids = {item.page.page_id for item in pages}
    for page in historical:
        if page.page_id not in current_ids:
            findings.append(_finding_page_missing(page))
    book = book_record_for_pages([item.page for item in pages])
    result = BookDiscovery(book, tuple(pages), tuple(findings))
    if root is not None:
        _persist_discovery(root, book, result.pages, historical, result.findings)
    return result


# --- Legacy migration-only Manifest API.  New pipeline code must use
# ``discover_book``; these routines preserve existing Task 1 serialization.

def generate_manifest(input_dir: Path | str) -> Manifest:
    """Create a deterministic legacy manifest for supported images in *input_dir*."""
    directory = Path(input_dir).resolve()
    if not directory.is_dir():
        raise ManifestValidationError(f"input_dir is not a directory: {directory}")
    filenames: list[str] = []
    for path in directory.iterdir():
        # Unlike ``Path.is_file``, this propagates a stat error to the
        # invocation boundary, which emits ``InvocationFailure(input_access)``.
        if stat.S_ISREG(path.stat().st_mode) and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            filenames.append(path.name)
    filenames.sort()
    pages = [{"filename": filename, "page_number": number, "status": "pending"}
             for number, filename in enumerate(filenames, start=1)]
    return Manifest(input_dir=str(directory), pages=pages, total_pages=len(pages))


def write_manifest(manifest: Manifest, path: Path | str) -> None:
    """Preserve Task 1's narrowly migrated legacy manifest serialization."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json({
        "input_dir": manifest.input_dir, "pages": manifest.pages,
        "total_pages": manifest.total_pages,
    }).encode("utf-8"))


def read_manifest(path: Path | str) -> Manifest:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"manifest is not valid JSON: {manifest_path}") from exc
    try:
        manifest = Manifest.from_dict(data)
    except (TypeError, KeyError) as exc:
        raise ManifestValidationError(f"manifest has an invalid shape: {manifest_path}") from exc
    validate_manifest(manifest)
    return manifest


def load_or_generate_manifest(input_dir: Path | str, manifest_path: Path | str | None = None) -> Manifest:
    directory = Path(input_dir).resolve()
    path = Path(manifest_path) if manifest_path is not None else directory / MANIFEST_FILENAME
    if path.exists():
        return read_manifest(path)
    manifest = generate_manifest(directory)
    write_manifest(manifest, path)
    return manifest


def validate_manifest(manifest: Manifest) -> None:
    if not isinstance(manifest.input_dir, str) or not manifest.input_dir:
        raise ManifestValidationError("input_dir must be a non-empty string")
    input_dir = Path(manifest.input_dir).resolve()
    if not input_dir.is_dir():
        raise ManifestValidationError(f"input_dir is not a directory: {input_dir}")
    if manifest.total_pages != len(manifest.pages):
        raise ManifestValidationError("total_pages does not match pages")
    page_numbers: set[int] = set()
    filenames: set[str] = set()
    for index, page in enumerate(manifest.pages, start=1):
        if not isinstance(page, dict):
            raise ManifestValidationError(f"page {index} must be an object")
        required = {"filename", "page_number", "status"}
        if not required.issubset(page):
            raise ManifestValidationError(f"page {index} is missing required fields")
        filename, page_number = page["filename"], page["page_number"]
        if not isinstance(filename, str) or not filename:
            raise ManifestValidationError(f"page {index} filename must be a non-empty string")
        filename_path = Path(filename)
        if filename_path.is_absolute() or filename_path.name != filename or "\\" in filename:
            raise ManifestValidationError(f"page {index} filename must be a bare filename: {filename}")
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise ManifestValidationError(f"page {index} page_number must be a positive integer")
        if page_number != index:
            raise ManifestValidationError("manifest pages must be ordered sequentially")
        if page_number in page_numbers or filename in filenames:
            raise ManifestValidationError("manifest contains duplicate page identity")
        page_path = (input_dir / filename).resolve()
        try:
            page_path.relative_to(input_dir)
        except ValueError as exc:
            raise ManifestValidationError(f"page {index} must be inside input_dir: {filename}") from exc
        # ``Path.is_file`` suppresses several access errors.  Let callers at
        # the invocation boundary classify a legacy page stat failure as
        # ``InvocationFailure(input_access)`` instead of mistaking it for a
        # missing/degraded page.
        if not stat.S_ISREG(page_path.stat().st_mode):
            raise ManifestValidationError(f"referenced page does not exist: {filename}")
        if page_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ManifestValidationError(f"unsupported page image type: {filename}")
        page_numbers.add(page_number)
        filenames.add(filename)


def manifest_page_paths(manifest: Manifest) -> list[tuple[int, Path]]:
    validate_manifest(manifest)
    input_dir = Path(manifest.input_dir).resolve()
    return [(page["page_number"], (input_dir / page["filename"]).resolve()) for page in manifest.pages]
