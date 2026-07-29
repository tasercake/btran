"""Image safety checks run before any page is sent to a model.

Resolution and unreadable inputs block a run. Orientation, blur, and duplicate
findings are warnings for the operator to review. Preflight never modifies a
source image.
"""

from __future__ import annotations

import hashlib
import tempfile
from io import BytesIO
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError, __version__ as PILLOW_VERSION

from btran.artifacts import ArtifactStore, preflight_semantic_key
from btran.hasher import compute_phash, compute_sha256, hamming_distance
from btran.manifest import ManifestValidationError, manifest_page_paths
from btran.schema import (
    ConfidenceAssessment,
    Finding,
    Manifest,
    stage_summary_finding,
    uncertainty_finding,
)


MINIMUM_IMAGE_DIMENSION = 500
BLUR_VARIANCE_THRESHOLD = 100.0
PERCEPTUAL_DUPLICATE_THRESHOLD = 5
PREFLIGHT_ALGORITHM_VERSION = "preflight-v1"
PREFLIGHT_ARTIFACT_KIND = "PagePreflight"
PREFLIGHT_ASSESSMENT_ARTIFACT_KIND = "ConfidenceAssessment"
_DECODER_FAILURES = (OSError, UnidentifiedImageError, Image.DecompressionBombError)


@dataclass(frozen=True)
class PreflightIssue:
    """One preflight finding, with an explicit run policy."""

    page_number: int
    image_path: str
    check: str
    severity: str
    message: str


@dataclass
class PreflightResult:
    """Complete preflight findings for a manifest."""

    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "blocking"]

    @property
    def warnings(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.blocking_issues


def _issue(
    page_number: int, image_path: Path, check: str, severity: str, message: str
) -> PreflightIssue:
    return PreflightIssue(page_number, str(image_path), check, severity, message)


def check_resolution(image_path: Path | str, page_number: int) -> PreflightIssue | None:
    """Block images whose smallest dimension is under 500 pixels."""
    path = Path(image_path)
    with Image.open(path) as image:
        width, height = image.size
    smallest_dimension = min(width, height)
    if smallest_dimension >= MINIMUM_IMAGE_DIMENSION:
        return None
    return _issue(
        page_number,
        path,
        "resolution",
        "blocking",
        f"smallest dimension is {smallest_dimension}px; minimum is {MINIMUM_IMAGE_DIMENSION}px",
    )


def check_exif_orientation(
    image_path: Path | str, page_number: int
) -> PreflightIssue | None:
    """Warn about a non-normal EXIF orientation without changing the source."""
    path = Path(image_path)
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
    if orientation == 1:
        return None
    return _issue(
        page_number,
        path,
        "orientation",
        "warning",
        f"EXIF orientation {orientation} detected; source image was not modified",
    )


def normalize_exif_orientation_copy(
    image_path: Path | str, output_path: Path | str
) -> Path:
    """Write an EXIF-normalized copy without modifying ``image_path``.

    The caller must choose a different destination; using the source as the
    destination is rejected to preserve preflight's non-destructive contract.
    """
    source = Path(image_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("output_path must differ from image_path")

    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image)
        normalized.load()
        image_format = image.format

    exif = normalized.getexif()
    exif[274] = 1
    normalized.save(destination, format=image_format, exif=exif.tobytes())
    return destination


def laplacian_variance(image_path: Path | str) -> float:
    """Calculate grayscale Laplacian variance without an OpenCV dependency."""
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        width, height = grayscale.size
        pixels = grayscale.load()

        if width < 3 or height < 3:
            return 0.0
        total = 0.0
        total_squared = 0.0
        count = 0
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                value = (
                    4 * pixels[x, y]
                    - pixels[x - 1, y]
                    - pixels[x + 1, y]
                    - pixels[x, y - 1]
                    - pixels[x, y + 1]
                )
                total += value
                total_squared += value * value
                count += 1
    mean = total / count
    return total_squared / count - mean * mean


def check_blur(image_path: Path | str, page_number: int) -> PreflightIssue | None:
    """Warn when Laplacian variance indicates a likely blurry page."""
    path = Path(image_path)
    variance = laplacian_variance(path)
    if variance >= BLUR_VARIANCE_THRESHOLD:
        return None
    return _issue(
        page_number,
        path,
        "blur",
        "warning",
        f"Laplacian variance is {variance:.1f}; warning threshold is {BLUR_VARIANCE_THRESHOLD}",
    )


def detect_duplicates(
    pages: Iterable[tuple[int, Path | str]],
    perceptual_threshold: int = PERCEPTUAL_DUPLICATE_THRESHOLD,
) -> list[PreflightIssue]:
    """Warn once for each later page matching an earlier SHA256 or phash."""
    seen: list[tuple[int, Path, str, str]] = []
    issues: list[PreflightIssue] = []
    for page_number, image_path in pages:
        path = Path(image_path)
        sha256 = compute_sha256(path)
        exact_match = next((prior for prior in seen if prior[2] == sha256), None)
        if exact_match is not None:
            issues.append(
                _issue(
                    page_number,
                    path,
                    "duplicate",
                    "warning",
                    f"exact SHA256 duplicate of page {exact_match[0]}",
                )
            )
            seen.append((page_number, path, sha256, exact_match[3]))
            continue

        phash = compute_phash(path)
        perceptual_match = next(
            (
                prior
                for prior in seen
                if hamming_distance(phash, prior[3]) <= perceptual_threshold
            ),
            None,
        )
        if perceptual_match is not None:
            issues.append(
                _issue(
                    page_number,
                    path,
                    "duplicate",
                    "warning",
                    f"perceptual hash duplicate of page {perceptual_match[0]}",
                )
            )
        seen.append((page_number, path, sha256, phash))
    return issues


def preflight_manifest(
    manifest: Manifest, *, defer_undecodable: bool = False,
) -> PreflightResult:
    """Validate every legacy-manifest page before model calls.

    Task-4 raw-byte discovery has already accepted supported files without
    decoding them.  Its bridge path therefore defers decode failures to source
    extraction, where they receive page diagnostics, instead of excluding
    those files at preflight.
    """
    try:
        pages = manifest_page_paths(manifest)
    except ManifestValidationError as exc:
        return PreflightResult(
            [_issue(0, Path(manifest.input_dir), "manifest", "blocking", str(exc))]
        )

    result = PreflightResult()
    readable_pages: list[tuple[int, Path]] = []
    for page_number, path in pages:
        try:
            orientation_issue = check_exif_orientation(path, page_number)
            if orientation_issue is not None:
                result.issues.append(orientation_issue)
            resolution_issue = check_resolution(path, page_number)
            if resolution_issue is not None:
                result.issues.append(resolution_issue)
            blur_issue = check_blur(path, page_number)
            if blur_issue is not None:
                result.issues.append(blur_issue)
            readable_pages.append((page_number, path))
        except _DECODER_FAILURES as exc:
            # Pillow uses OSError both for unknown formats and late decoder
            # failures such as a truncated but otherwise valid PNG.  Discovery
            # has already admitted supported files by raw bytes, so its bridge
            # must defer every decode failure to source extraction.
            severity = "warning" if defer_undecodable else "blocking"
            message = f"cannot read image: {exc}"
            if severity == "warning":
                message += "; accepted by discovery and deferred to source extraction"
            result.issues.append(_issue(page_number, path, "readability", severity, message))

    try:
        result.issues.extend(detect_duplicates(readable_pages))
    except _DECODER_FAILURES as exc:
        severity = "warning" if defer_undecodable else "blocking"
        message = f"cannot hash image: {exc}"
        if severity == "warning":
            message += "; accepted by discovery and deferred to source extraction"
        result.issues.append(_issue(0, Path(manifest.input_dir), "duplicate", severity, message))
    return result


@dataclass(frozen=True)
class PreflightPageInput:
    """Accepted page, optionally carrying discovery's immutable raw-byte copy."""

    page_id: str
    image_path: Path
    raw_file_sha256: str
    page_number: int = 1
    raw_bytes: bytes | None = None


@dataclass(frozen=True)
class PersistedPreflightPage:
    page_id: str
    artifact_id: str
    assessment_artifact_id: str | None
    finding_ids: tuple[str, ...]
    degraded: bool


@dataclass(frozen=True)
class PersistedPreflightRun:
    pages: tuple[PersistedPreflightPage, ...]
    stage_summary_finding_id: str
    status: str


def _preflight_configuration(configuration: Mapping[str, Any] | None) -> dict[str, Any]:
    value = {
        "minimum_image_dimension": MINIMUM_IMAGE_DIMENSION,
        "blur_variance_threshold": BLUR_VARIANCE_THRESHOLD,
        "perceptual_duplicate_threshold": PERCEPTUAL_DUPLICATE_THRESHOLD,
    }
    if configuration is not None:
        value.update(dict(configuration))
    return value


def _normalized_image_bytes(raw_bytes: bytes) -> bytes:
    """Return normalized bytes from exact accepted raw bytes, never a second read."""
    with Image.open(BytesIO(raw_bytes)) as image:
        normalized = ImageOps.exif_transpose(image)
        normalized.load()
        with tempfile.SpooledTemporaryFile() as output:
            normalized.save(output, format="PNG")
            output.seek(0)
            return output.read()


def _raw_bytes(page: PreflightPageInput) -> bytes:
    """Use discovery-owned bytes when available; otherwise make one source read."""
    raw_bytes = page.raw_bytes if page.raw_bytes is not None else page.image_path.read_bytes()
    if not isinstance(raw_bytes, bytes):
        raise TypeError("accepted raw bytes must be bytes")
    return raw_bytes


def _validate_raw_identity(page: PreflightPageInput, raw_bytes: bytes) -> None:
    if hashlib.sha256(raw_bytes).hexdigest() != page.raw_file_sha256:
        raise OSError("raw bytes do not match accepted page identity")


def _accepted_raw_bytes(page: PreflightPageInput) -> bytes:
    """Return actual bytes only after validating their accepted identity."""
    raw_bytes = _raw_bytes(page)
    _validate_raw_identity(page, raw_bytes)
    return raw_bytes


def _laplacian_variance_image(image: Image.Image) -> float:
    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()
    if width < 3 or height < 3:
        return 0.0
    total = total_squared = 0.0
    count = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            value = (
                4 * pixels[x, y]
                - pixels[x - 1, y]
                - pixels[x + 1, y]
                - pixels[x, y - 1]
                - pixels[x, y + 1]
            )
            total += value
            total_squared += value * value
            count += 1
    mean = total / count
    return total_squared / count - mean * mean


def _raw_preflight_checks(page: PreflightPageInput, raw_bytes: bytes) -> tuple[list[PreflightIssue], str]:
    """Check/hash one immutable raw copy, never mutable path after acceptance."""
    with Image.open(BytesIO(raw_bytes)) as image:
        orientation = image.getexif().get(274, 1)
        width, height = image.size
        image.load()
        variance = _laplacian_variance_image(image)
        phash = str(imagehash.phash(image))
    issues: list[PreflightIssue] = []
    if orientation != 1:
        issues.append(_issue(page.page_number, page.image_path, "orientation", "warning",
                             f"EXIF orientation {orientation} detected; source image was not modified"))
    smallest_dimension = min(width, height)
    if smallest_dimension < MINIMUM_IMAGE_DIMENSION:
        issues.append(_issue(page.page_number, page.image_path, "resolution", "blocking",
                             f"smallest dimension is {smallest_dimension}px; minimum is {MINIMUM_IMAGE_DIMENSION}px"))
    if variance < BLUR_VARIANCE_THRESHOLD:
        issues.append(_issue(page.page_number, page.image_path, "blur", "warning",
                             f"Laplacian variance is {variance:.1f}; warning threshold is {BLUR_VARIANCE_THRESHOLD}"))
    return issues, phash


def _preflight_finding(page_id: str, issue: PreflightIssue) -> Finding:
    # Legacy "blocking" describes old UI severity only. New pipeline findings
    # are informational and cannot encode a gate.
    severity = "error" if issue.severity == "blocking" else "warning"
    return Finding(kind=f"preflight_{issue.check}", severity=severity, stage="preflight",
                   subject_refs=(page_id,), evidence={"check": issue.check,
                   "legacy_severity": issue.severity, "message": issue.message},
                   message="Preflight finding recorded; accepted pages continue.")


def _persist_preflight_assessment(
    store: ArtifactStore, *, page_id: str, artifact_id: str, base_revision_id: str,
) -> tuple[str, tuple[str, ...]]:
    assessment = ConfidenceAssessment(subject_id=page_id, producing_stage="preflight",
                                      producing_artifact_id=artifact_id, score=None,
                                      signals=("decode_failure", "degraded", "fallback"))
    uncertainty = uncertainty_finding(assessment)
    store.put_finding(uncertainty)
    # Preflight has only a page identity.  A segment-scoped request must name
    # an exact segment, so defer this degraded review request to extraction's
    # diagnostic segment instead of publishing an invalid page-as-segment one.
    finding_ids = (uncertainty.finding_id,)
    envelope = store.put(PREFLIGHT_ASSESSMENT_ARTIFACT_KIND, assessment.to_dict(),
                         dependency_ids=(artifact_id,), finding_ids=finding_ids,
                         semantic_key=f"confidence:{artifact_id}")
    return envelope.artifact_id, finding_ids


def persist_preflight_pages(
    pages: Iterable[PreflightPageInput], *, store: ArtifactStore,
    configuration: Mapping[str, Any] | None = None, base_revision_id: str = "unsealed",
) -> PersistedPreflightRun:
    """Persist semantic-keyed preflight leaves while isolating every page failure.

    Raw and normalized image bytes are key material.  File paths, ordering, and
    mtimes are deliberately absent.  Decode failure is retained for extraction
    rather than dropping an accepted page or creating effective content here.
    """
    inputs = tuple(pages)
    if not all(isinstance(page, PreflightPageInput) for page in inputs):
        raise TypeError("pages must contain PreflightPageInput")
    config = _preflight_configuration(configuration)
    results: list[PersistedPreflightPage] = []
    seen: list[tuple[int, str, str]] = []
    for page in inputs:
        # Keep this value even if decode/normalization fails: fallback semantic
        # identity must use accepted raw bytes, never a shared empty sentinel.
        raw_bytes: bytes | None = None
        normalized_bytes = b""
        issues: list[PreflightIssue] = []
        degraded = False
        try:
            # Retain actual buffer before identity validation.  A mismatch is
            # a diagnostic leaf keyed by these bytes, never digest bytes.
            raw_bytes = _raw_bytes(page)
            _validate_raw_identity(page, raw_bytes)
            normalized_bytes = _normalized_image_bytes(raw_bytes)
            issues, phash = _raw_preflight_checks(page, raw_bytes)
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            # Duplicate check is isolated too: a malformed peer never erases a
            # readable page's own preflight artifact.
            exact = next((prior for prior in seen if prior[1] == sha256), None)
            perceptual = None if exact is not None else next(
                (prior for prior in seen if hamming_distance(phash, prior[2]) <= PERCEPTUAL_DUPLICATE_THRESHOLD), None
            )
            if exact is not None:
                issues.append(_issue(page.page_number, page.image_path, "duplicate", "warning",
                                     f"exact SHA256 duplicate of page {exact[0]}"))
            elif perceptual is not None:
                issues.append(_issue(page.page_number, page.image_path, "duplicate", "warning",
                                     f"perceptual hash duplicate of page {perceptual[0]}"))
            seen.append((page.page_number, sha256, phash))
        except _DECODER_FAILURES as exc:
            degraded = True
            issues.append(_issue(page.page_number, page.image_path, "readability", "warning",
                                 f"cannot decode accepted image: {type(exc).__name__}"))
        except Exception as exc:
            # Preflight is a page leaf: a library/hash/config surprise is a
            # diagnostic, never a reason to discard other accepted pages.
            degraded = True
            issues.append(_issue(page.page_number, page.image_path, "readability", "warning",
                                 f"cannot preflight accepted image: {type(exc).__name__}"))
        key = preflight_semantic_key(algorithm_version=PREFLIGHT_ALGORITHM_VERSION,
                                     image_library_version=PILLOW_VERSION, configuration=config,
                                     raw_bytes=(raw_bytes if raw_bytes is not None else bytes.fromhex(page.raw_file_sha256)),
                                     normalized_image_bytes=normalized_bytes)
        findings = [_preflight_finding(page.page_id, issue) for issue in issues]
        for finding in findings:
            store.put_finding(finding)
        artifact = store.put(PREFLIGHT_ARTIFACT_KIND, {
            "page_id": page.page_id, "raw_file_sha256": page.raw_file_sha256,
            "raw_image_sha256": hashlib.sha256(raw_bytes if raw_bytes is not None else bytes.fromhex(page.raw_file_sha256)).hexdigest(),
            "normalized_image_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
            "issues": [{"check": issue.check, "severity": issue.severity, "message": issue.message}
                       for issue in issues],
        }, finding_ids=tuple(sorted(finding.finding_id for finding in findings)), semantic_key=key)
        assessment_id: str | None = None
        assessment_findings: tuple[str, ...] = ()
        if degraded:
            assessment_id, assessment_findings = _persist_preflight_assessment(
                store, page_id=page.page_id, artifact_id=artifact.artifact_id,
                base_revision_id=base_revision_id)
        results.append(PersistedPreflightPage(
            page.page_id, artifact.artifact_id, assessment_id,
            tuple(sorted((*[finding.finding_id for finding in findings], *assessment_findings))), degraded,
        ))
    status = "degraded" if any(result.degraded for result in results) else "completed"
    summary = stage_summary_finding("preflight", status, {
        "accepted_pages": len(results), "degraded_pages": sum(result.degraded for result in results),
        "findings": sum(len(result.finding_ids) for result in results),
    }, subject_refs=tuple(sorted(result.page_id for result in results)))
    store.put_finding(summary)
    return PersistedPreflightRun(tuple(results), summary.finding_id, status)
