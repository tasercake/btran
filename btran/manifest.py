"""Persistent, validated input-page manifests."""

from __future__ import annotations

import json
from pathlib import Path

from btran.schema import Manifest


MANIFEST_FILENAME = "manifest.json"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic"})


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot safely identify its input pages."""


def generate_manifest(input_dir: Path | str) -> Manifest:
    """Create a deterministic manifest for supported images in *input_dir*."""
    directory = Path(input_dir).resolve()
    if not directory.is_dir():
        raise ManifestValidationError(f"input_dir is not a directory: {directory}")

    filenames = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    pages = [
        {"filename": filename, "page_number": number, "status": "pending"}
        for number, filename in enumerate(filenames, start=1)
    ]
    return Manifest(input_dir=str(directory), pages=pages, total_pages=len(pages))


def write_manifest(manifest: Manifest, path: Path | str) -> None:
    """Write a schema ``Manifest`` as readable JSON."""
    manifest.to_file(Path(path))


def read_manifest(path: Path | str) -> Manifest:
    """Read and validate a manifest JSON file."""
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


def load_or_generate_manifest(
    input_dir: Path | str, manifest_path: Path | str | None = None
) -> Manifest:
    """Load an existing manifest, or generate and persist one if it is absent."""
    directory = Path(input_dir).resolve()
    path = Path(manifest_path) if manifest_path is not None else directory / MANIFEST_FILENAME
    if path.exists():
        return read_manifest(path)

    manifest = generate_manifest(directory)
    write_manifest(manifest, path)
    return manifest


def validate_manifest(manifest: Manifest) -> None:
    """Ensure every page entry is a safe, existing page before model calls."""
    if not isinstance(manifest.input_dir, str):
        raise ManifestValidationError("input_dir must be a string")
    if not manifest.input_dir:
        raise ManifestValidationError("input_dir must be non-empty")
    input_dir = Path(manifest.input_dir).resolve()
    if not input_dir.is_dir():
        raise ManifestValidationError(f"input_dir is not a directory: {input_dir}")
    if manifest.total_pages != len(manifest.pages):
        raise ManifestValidationError(
            f"total_pages ({manifest.total_pages}) does not match pages ({len(manifest.pages)})"
        )

    page_numbers: set[int] = set()
    filenames: set[str] = set()
    for index, page in enumerate(manifest.pages, start=1):
        if not isinstance(page, dict):
            raise ManifestValidationError(f"page {index} must be an object")
        required = {"filename", "page_number", "status"}
        if not required.issubset(page):
            missing = ", ".join(sorted(required - set(page)))
            raise ManifestValidationError(f"page {index} is missing required fields: {missing}")

        filename = page["filename"]
        page_number = page["page_number"]
        if not isinstance(filename, str) or not filename:
            raise ManifestValidationError(f"page {index} filename must be a non-empty string")
        filename_path = Path(filename)
        if filename_path.is_absolute() or filename_path.name != filename or "\\" in filename:
            raise ManifestValidationError(f"page {index} filename must be a bare filename: {filename}")
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise ManifestValidationError(f"page {index} page_number must be a positive integer")
        if page_number in page_numbers:
            raise ManifestValidationError(f"duplicate page_number: {page_number}")
        if filename in filenames:
            raise ManifestValidationError(f"duplicate filename: {filename}")

        page_path = (input_dir / filename).resolve()
        try:
            page_path.relative_to(input_dir)
        except ValueError as exc:
            raise ManifestValidationError(
                f"page {index} must be inside input_dir: {filename}"
            ) from exc
        if not page_path.is_file():
            raise ManifestValidationError(f"referenced page does not exist: {filename}")
        if page_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ManifestValidationError(f"unsupported page image type: {filename}")

        page_numbers.add(page_number)
        filenames.add(filename)


def manifest_page_paths(manifest: Manifest) -> list[tuple[int, Path]]:
    """Return manifest pages as ``(page_number, absolute_path)`` pairs.

    Validation is deliberately repeated at this boundary so callers can use it
    immediately before model work without trusting an earlier preflight step.
    """
    validate_manifest(manifest)
    input_dir = Path(manifest.input_dir).resolve()
    return [
        (page["page_number"], (input_dir / page["filename"]).resolve())
        for page in manifest.pages
    ]
