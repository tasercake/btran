"""Legacy manifest migration helpers plus raw-hash book discovery."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from btran.artifacts import (
    ArtifactEnvelope,
    ArtifactError,
    ArtifactStore,
    DependencyGraphEdge,
    LegacyReadOnlyArtifactStore,
    RevisionStore,
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

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, RevisionSnapshot):
            raise TypeError("selected closure snapshot must be a RevisionSnapshot")
        if self.snapshot.revision_id != self.revision_id:
            raise ValueError("selected closure revision ID does not match snapshot")
        maps = {
            "records": self.records, "findings": self.findings,
            "edges": self.edges, "attestations": self.attestations,
            "provenance": self.provenance,
        }
        for name, value in maps.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"selected closure {name} must be a mapping")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        # The selected archive is the authority.  Do not silently broaden it
        # from a mutable index or from historical records.
        if set(self.records) != set(self.snapshot.selected_artifact_ids) | {
            dependency_id
            for record in self.records.values()
            for dependency_id in record.dependency_ids
        }:
            raise ValueError("selected closure record map is not closed")
        if set(self.findings) != set(self.snapshot.selected_finding_ids) | {
            finding_id
            for record in self.records.values()
            for finding_id in record.finding_ids
        } | {
            finding_id
            for finding in self.findings.values()
            for finding_id in finding.dependency_ids
        }:
            raise ValueError("selected closure finding map is not closed")

    @classmethod
    def empty(cls) -> "SelectedClosure":
        """Return the explicit empty closure used before the first revision."""
        snapshot = RevisionSnapshot(revision_id="unsealed")
        return cls("unsealed", snapshot, {}, {}, {}, {}, {})

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
        """Return selected effective pages with declared segment order."""
        pages: list[tuple[tuple[Any, ...], Any]] = []
        segment_records = {
            self._payload_id(record, "effective_segment_id"): record
            for record in self.records.values()
            if self._payload_id(record, "effective_segment_id") is not None
        }
        all_pages = tuple(self.records.values())
        has_target_pages = any(record.kind == "EffectiveTargetPage" for record in all_pages)
        page_kinds = {"EffectiveTargetPage"} if has_target_pages else {"EffectiveSourcePage"}
        for record in self._typed_records(page_kinds):
            try:
                page = EffectivePage.from_dict(record.payload)
                segments = tuple(
                    EffectiveSegment.from_dict(segment_records[segment_id].payload)
                    for segment_id in page.effective_segment_ids
                )
            except (KeyError, SchemaError, TypeError, ValueError) as exc:
                raise SelectedClosureError("selected effective page has invalid children") from exc
            # Page number is optional.  When present it is the declared page
            # order; otherwise retain the stable page identity order.
            number = page.display_metadata.get("page_number")
            order = (0, number, page.page_id) if isinstance(number, int) and not isinstance(number, bool) else (1, page.page_id)
            pages.append((order, (page, segments)))
        pages.sort(key=lambda item: item[0])
        from btran.orchestrator_contract import OrderedEffectivePage
        return tuple(OrderedEffectivePage(page, segments) for _, (page, segments) in pages)

    @property
    def selected_effective_content(self) -> Any:
        from btran.orchestrator_contract import SelectedEffectiveContent
        return SelectedEffectiveContent(self.ordered_effective_pages, finding_ids=self.final_finding_ids)

    @property
    def source_page_cache_leaves(self) -> tuple[ArtifactEnvelope, ...]:
        return tuple(record for record in self.records.values() if record.kind in {
            "RawSourceExtraction", "DiagnosticSourceFallback", "EffectiveSourcePage",
            "DiagnosticEffectiveSourcePage",
        })

    @property
    def translation_segment_cache_leaves(self) -> tuple[ArtifactEnvelope, ...]:
        return tuple(record for record in self.records.values() if record.kind in {
            "TranslationArtifact", "DiagnosticTranslationFallback", "EffectiveTargetSegment",
            "DiagnosticEffectiveTargetSegment",
        })

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
        return MappingProxyType({
            stable_id: record
            for record in self.source_page_cache_leaves
            for stable_id in (self._payload_id(record, "page_id"),)
            if stable_id is not None
        })

    @property
    def translation_segment_cache_leaf_map(self) -> Mapping[str, ArtifactEnvelope]:
        return MappingProxyType({
            stable_id: record
            for record in self.translation_segment_cache_leaves
            for stable_id in (self._payload_id(record, "segment_id"),)
            if stable_id is not None
        })

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


def _revision_members(revisions: Any, revision_id: str) -> tuple[RevisionSnapshot, Mapping[str, bytes], Mapping[str, Any]]:
    """Validate one revision through its store, then retain its bytes in memory."""
    try:
        if isinstance(revisions, V2RevisionStore):
            values = revisions.storage.verify_revision(revision_id)
            snapshot = RevisionSnapshot.from_json(values["snapshot.json"].decode("utf-8"))
            provenance: Mapping[str, Any] = {}
            return snapshot, values, provenance

        # Legacy verification is intentionally routed through the read-only
        # adapter.  It does not quarantine invalid files or create directories.
        revisions.verify_bundle(revision_id)
        bundle = Path(revisions.revisions_dir) / revision_id
        snapshot_bytes = (bundle / "snapshot.json").read_bytes()
        snapshot = RevisionSnapshot.from_json(snapshot_bytes.decode("utf-8"))
        manifest = _canonical_member((bundle / "bundle-manifest.json").read_bytes(), "bundle-manifest.json")
        values: dict[str, bytes] = {"snapshot.json": snapshot_bytes}
        for artifact_id in manifest.get("artifact_ids", ()):
            values[f"artifacts/{artifact_id}.json"] = (bundle / "artifacts" / f"{artifact_id}.json").read_bytes()
        for finding_id in manifest.get("finding_ids", ()):
            values[f"findings/{finding_id}.json"] = (bundle / "findings" / f"{finding_id}.json").read_bytes()
        for attestation_id in manifest.get("semantic_attestation_ids", ()):
            values[f"attestations/{attestation_id}.json"] = (bundle / "attestations" / f"{attestation_id}.json").read_bytes()
        for edge_id in manifest.get("edge_ids", ()):
            values[f"edges/{edge_id}.json"] = (bundle / "graph" / f"{edge_id}.json").read_bytes()
        provenance_path = bundle / "provenance.json"
        provenance = {} if not provenance_path.exists() else _canonical_member(provenance_path.read_bytes(), "provenance.json")
        if not isinstance(provenance, Mapping):
            raise SelectedClosureError("revision provenance must be an object")
        return snapshot, values, provenance
    except (OSError, UnicodeDecodeError, SchemaError, ArtifactError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SelectedClosureError):
            raise
        raise SelectedClosureError(f"cannot load selected revision {revision_id}") from exc


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
    if set(records) != set(snapshot.selected_artifact_ids) | {
        dependency_id for record in records.values() for dependency_id in record.dependency_ids
    }:
        raise SelectedClosureError("selected records are not the exact archive closure")
    if not set(snapshot.selected_finding_ids).issubset(findings):
        raise SelectedClosureError("selected findings are missing")
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

    # Detect duplicate stable identities within one persisted kind.  Artifact
    # IDs alone are not enough: two valid envelopes can still claim one page or
    # one effective segment.
    seen_stable: set[tuple[str, str, str]] = set()
    for record in records.values():
        for field in ("page_id", "segment_id", "effective_page_id", "effective_segment_id", "concept_id", "projection_id", "translation_artifact_id"):
            stable_id = record.payload.get(field)
            if isinstance(stable_id, str) and stable_id:
                key = (record.kind, field, stable_id)
                if key in seen_stable:
                    raise SelectedClosureError("duplicate selected stable identity")
                seen_stable.add(key)

    return SelectedClosure(revision_id, snapshot, records, findings, edges, attestations, provenance)


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


def _persist_discovery(
    workspace: Path,
    book: BookRecord,
    pages: tuple[DiscoveredPage, ...],
    historical_pages: tuple[PageRecord, ...],
    findings: tuple[Finding, ...],
) -> None:
    # A legacy workspace is read/verify-only.  In particular, do not let the
    # discovery compatibility path create a v2 DB or call the legacy adapter's
    # mutation guard while merely reading old state.
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
