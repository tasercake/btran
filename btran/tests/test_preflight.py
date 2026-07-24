"""Tests for image input preflight checks."""

from pathlib import Path

from PIL import Image

from btran.preflight import (
    BLUR_VARIANCE_THRESHOLD,
    PreflightResult,
    check_blur,
    check_resolution,
    correct_exif_orientation,
    detect_duplicates,
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


def test_exif_orientation_is_corrected_in_place_and_warned(tmp_path):
    path = tmp_path / "rotated_page.jpg"
    _copy_image("rotated_page.jpg", path)
    before = Image.open(path)
    assert before.size == (800, 600)
    assert before.getexif().get(274) == 6

    issue = correct_exif_orientation(path, page_number=1)

    corrected = Image.open(path)
    assert issue is not None
    assert issue.severity == "warning"
    assert issue.check == "orientation"
    assert corrected.size == (600, 800)
    assert corrected.getexif().get(274, 1) == 1


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
