"""Image safety checks run before any page is sent to a model.

Resolution and unreadable inputs block a run. Orientation, blur, and duplicate
findings are warnings for the operator to review. Preflight never modifies a
source image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

from btran.hasher import compute_phash, compute_sha256, hamming_distance
from btran.manifest import ManifestValidationError, manifest_page_paths
from btran.schema import Manifest


MINIMUM_IMAGE_DIMENSION = 500
BLUR_VARIANCE_THRESHOLD = 100.0
PERCEPTUAL_DUPLICATE_THRESHOLD = 5


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


def preflight_manifest(manifest: Manifest) -> PreflightResult:
    """Validate and inspect every page before a model call can be made."""
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
        except (OSError, UnidentifiedImageError) as exc:
            result.issues.append(
                _issue(page_number, path, "readability", "blocking", f"cannot read image: {exc}")
            )

    try:
        result.issues.extend(detect_duplicates(readable_pages))
    except (OSError, UnidentifiedImageError) as exc:
        result.issues.append(
            _issue(0, Path(manifest.input_dir), "duplicate", "blocking", f"cannot hash image: {exc}")
        )
    return result
