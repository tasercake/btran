"""Tests for image input preflight checks."""

import os
from pathlib import Path
from unittest.mock import patch

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


def test_preflight_defers_truncated_valid_png_oserror_after_discovery(tmp_path):
    path = tmp_path / "truncated.png"
    Image.new("RGB", (600, 600), "white").save(path)
    path.write_bytes(path.read_bytes()[:-30])
    manifest = Manifest(
        input_dir=str(tmp_path),
        pages=[{"filename": path.name, "page_number": 1, "status": "pending"}],
        total_pages=1,
    )

    # This is a valid PNG header whose decoder raises OSError when loaded.
    with pytest.raises(OSError, match="truncated"):
        check_blur(path, page_number=1)

    result = preflight_manifest(manifest, defer_undecodable=True)

    assert result.ok
    assert [(issue.check, issue.severity) for issue in result.issues] == [("readability", "warning")]
    assert "deferred to source extraction" in result.issues[0].message


def test_preflight_defers_decompression_bomb_after_discovery(tmp_path, monkeypatch):
    path = tmp_path / "bomb.png"
    Image.new("RGB", (600, 600), "white").save(path)
    manifest = Manifest(
        input_dir=str(tmp_path),
        pages=[{"filename": path.name, "page_number": 1, "status": "pending"}],
        total_pages=1,
    )
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(Image.DecompressionBombError):
        check_resolution(path, page_number=1)

    result = preflight_manifest(manifest, defer_undecodable=True)

    assert result.ok
    assert [(issue.check, issue.severity) for issue in result.issues] == [("readability", "warning")]
    assert "deferred to source extraction" in result.issues[0].message


def test_preflight_defers_decompression_bomb_from_phash(tmp_path):
    path = tmp_path / "page.png"
    _copy_image("hi_res_page.png", path)
    manifest = Manifest(
        input_dir=str(tmp_path),
        pages=[{"filename": path.name, "page_number": 1, "status": "pending"}],
        total_pages=1,
    )

    with patch(
        "btran.preflight.compute_phash",
        side_effect=Image.DecompressionBombError("too many pixels"),
    ):
        result = preflight_manifest(manifest, defer_undecodable=True)

    assert result.ok
    assert [(issue.check, issue.severity) for issue in result.issues] == [("duplicate", "warning")]
    assert "deferred to source extraction" in result.issues[0].message


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


def test_persisted_preflight_is_semantic_keyed_and_decode_failure_keeps_page(tmp_path):
    from hashlib import sha256

    from btran.artifacts import ArtifactStore
    from btran.identity import page_id_for_raw_sha256
    from btran.preflight import PreflightPageInput, persist_preflight_pages

    good = tmp_path / "good.png"
    bad = tmp_path / "accepted-but-bad.png"
    _copy_image("hi_res_page.png", good)
    bad.write_bytes(b"not a decoded image")
    inputs = []
    for number, path in enumerate((good, bad), 1):
        digest = sha256(path.read_bytes()).hexdigest()
        inputs.append(PreflightPageInput(page_id_for_raw_sha256(digest), path, digest, number))
    store = ArtifactStore(tmp_path / "state")

    result = persist_preflight_pages(inputs, store=store, base_revision_id="base-rev")

    assert len(result.pages) == 2
    assert result.status == "degraded"
    valid, failed = result.pages
    assert store.get(valid.artifact_id).kind == "PagePreflight"
    assert failed.assessment_artifact_id is not None
    findings = [store.get_finding(item) for item in failed.finding_ids]
    # Preflight knows only page identity. It must defer a segment-scoped review
    # request until extraction creates its diagnostic segment.
    assert not [item for item in findings if item.kind == "review_request"]
    uncertainty = next(item for item in findings if item.kind == "uncertainty")
    assert uncertainty.subject_refs == (failed.page_id,)


def test_persisted_preflight_hash_mismatch_keys_and_records_actual_raw_bytes(tmp_path):
    from hashlib import sha256

    from btran.artifacts import ArtifactStore, preflight_semantic_key
    from btran.identity import page_id_for_raw_sha256
    from btran.preflight import (
        PREFLIGHT_ALGORITHM_VERSION,
        PreflightPageInput,
        persist_preflight_pages,
    )
    from PIL import __version__ as pillow_version

    actual_raw = b"accepted bytes that fail identity validation"
    claimed_digest = sha256(b"different accepted bytes").hexdigest()
    page_id = page_id_for_raw_sha256(claimed_digest)
    store = ArtifactStore(tmp_path / "state")

    result = persist_preflight_pages([
        PreflightPageInput(page_id, tmp_path / "unread.png", claimed_digest, raw_bytes=actual_raw)
    ], store=store)

    artifact = store.get(result.pages[0].artifact_id)
    assert result.status == "degraded"
    assert artifact.payload["raw_image_sha256"] == sha256(actual_raw).hexdigest()
    assert artifact.semantic_key == preflight_semantic_key(
        algorithm_version=PREFLIGHT_ALGORITHM_VERSION,
        image_library_version=pillow_version,
        configuration={
            "minimum_image_dimension": 500,
            "blur_variance_threshold": 100.0,
            "perceptual_duplicate_threshold": 5,
        },
        raw_bytes=actual_raw,
        normalized_image_bytes=b"",
    )


def test_persisted_preflight_uses_discovery_raw_copy_after_source_mutates(tmp_path):
    from hashlib import sha256

    from btran.artifacts import ArtifactStore
    from btran.identity import page_id_for_raw_sha256
    from btran.preflight import PreflightPageInput, persist_preflight_pages

    image = tmp_path / "page.png"
    _copy_image("hi_res_page.png", image)
    accepted_raw = image.read_bytes()
    digest = sha256(accepted_raw).hexdigest()
    # Simulate source replacement after discovery. Preflight must not reopen it.
    image.write_bytes(b"replaced mutable source")
    store = ArtifactStore(tmp_path / "state")

    result = persist_preflight_pages([
        PreflightPageInput(page_id_for_raw_sha256(digest), image, digest, raw_bytes=accepted_raw)
    ], store=store)

    artifact = store.get(result.pages[0].artifact_id)
    assert result.status == "completed"
    assert artifact.payload["raw_image_sha256"] == digest
