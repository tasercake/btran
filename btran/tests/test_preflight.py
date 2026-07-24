"""Tests for image input preflight checks."""

import os
from pathlib import Path

from PIL import Image
import pytest

from btran.preflight import (
    BLUR_VARIANCE_THRESHOLD,
    PreflightResult,
    check_blur,
    check_resolution,
    detect_duplicates,
    normalize_exif_orientation_copy,
    preflight_manifest,
)
from btran.schema import Manifest


FIXTURES = Path(__file__).parent / "fixtures"


def _copy_image(name: str, destination: Path) -> None:
    destination.write_bytes((FIXTURES / name).read_bytes())


def test_resolution_under_500px_on_smallest_side_is_blocking():
    issue = check_resolution(FIXTURES / "low_res_page.png", page_number=1)

    assert issue is not None
    assert issue.severity == "blocking"
    assert issue.check == "resolution"
    assert "500px" in issue.message


def test_resolution_at_500px_on_smallest_side_passes():
    assert check_resolution(FIXTURES / "hi_res_page.png", page_number=1) is None


def test_preflight_reports_exif_orientation_without_rewriting_source(tmp_path):
    path = tmp_path / "rotated_page.jpg"
    _copy_image("rotated_page.jpg", path)
    source_bytes = path.read_bytes()
    manifest = Manifest(
        input_dir=str(tmp_path),
        pages=[{"filename": path.name, "page_number": 1, "status": "pending"}],
        total_pages=1,
    )

    result = preflight_manifest(manifest)

    assert any(issue.check == "orientation" for issue in result.warnings)
    assert path.read_bytes() == source_bytes


def test_normalize_exif_orientation_copy_writes_normalized_copy_without_changing_source(tmp_path):
    source = tmp_path / "rotated_page.jpg"
    output = tmp_path / "normalized_page.jpg"
    _copy_image("rotated_page.jpg", source)
    source_bytes = source.read_bytes()

    written_path = normalize_exif_orientation_copy(source, output)

    normalized = Image.open(output)
    assert written_path == output
    assert source.read_bytes() == source_bytes
    assert normalized.size == (600, 800)
    assert normalized.getexif().get(274, 1) == 1


def test_normalize_exif_orientation_copy_rejects_source_as_destination(tmp_path):
    source = tmp_path / "rotated_page.jpg"
    _copy_image("rotated_page.jpg", source)

    with pytest.raises(ValueError, match="must differ"):
        normalize_exif_orientation_copy(source, source)


def test_normalize_exif_orientation_copy_rejects_hardlinked_source_destination(tmp_path):
    source = tmp_path / "rotated_page.jpg"
    output = tmp_path / "hardlinked_normalized_page.jpg"
    _copy_image("rotated_page.jpg", source)
    source_bytes = source.read_bytes()
    os.link(source, output)

    with pytest.raises(ValueError, match="must differ"):
        normalize_exif_orientation_copy(source, output)

    assert source.read_bytes() == source_bytes


def test_sharp_image_passes_laplacian_blur_check():
    issue = check_blur(FIXTURES / "hi_res_page.png", page_number=1)

    assert issue is None


def test_blurry_image_is_a_warning_from_laplacian_variance():
    issue = check_blur(FIXTURES / "blurry_page.png", page_number=1)

    assert issue is not None
    assert issue.severity == "warning"
    assert issue.check == "blur"
    assert str(BLUR_VARIANCE_THRESHOLD) in issue.message


def test_exact_sha256_duplicate_is_warned(tmp_path):
    first = tmp_path / "page_001.png"
    second = tmp_path / "page_002.png"
    _copy_image("hi_res_page.png", first)
    _copy_image("duplicate_page.png", second)

    issues = detect_duplicates([(1, first), (2, second)])

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].check == "duplicate"
    assert "SHA256" in issues[0].message


def test_perceptually_identical_image_with_different_bytes_is_warned(tmp_path):
    first = tmp_path / "page_001.png"
    second = tmp_path / "page_002.jpg"
    _copy_image("hi_res_page.png", first)
    _copy_image("perceptual_duplicate.jpg", second)

    issues = detect_duplicates([(1, first), (2, second)])

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "perceptual hash" in issues[0].message


def test_preflight_manifest_checks_every_page_and_blocks_invalid_input(tmp_path):
    _copy_image("hi_res_page.png", tmp_path / "page_001.png")
    _copy_image("low_res_page.png", tmp_path / "page_002.png")
    manifest = Manifest(
        input_dir=str(tmp_path),
        pages=[
            {"filename": "page_001.png", "page_number": 1, "status": "pending"},
            {"filename": "page_002.png", "page_number": 2, "status": "pending"},
        ],
        total_pages=2,
    )

    result = preflight_manifest(manifest)

    assert isinstance(result, PreflightResult)
    assert not result.ok
    assert result.blocking_issues[0].page_number == 2
