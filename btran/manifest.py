"""Legacy manifest migration helpers plus raw-hash book discovery."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from btran.artifacts import ArtifactStore
from btran.identity import (
    PagePlacement,
    book_record_for_pages,
    page_id_for_raw_sha256,
    page_record_for_raw_sha256,
    raw_file_sha256,
    reconcile_book_pages,
)
from btran.schema import BookRecord, Finding, Manifest, PageRecord, SchemaError, canonical_json


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
    workspace.mkdir(parents=True, exist_ok=True)
    # Persist findings independently through Task 3 immutable storage, then
    # retain IDs in the small discovery snapshot for inspectability.
    store = ArtifactStore(workspace)
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
