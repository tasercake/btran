"""Tests for the input-page manifest contract."""

import json
from pathlib import Path

import pytest

from btran.manifest import (
    ManifestValidationError,
    generate_manifest,
    load_or_generate_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _copy_image(name: str, destination: Path) -> None:
    destination.write_bytes((FIXTURES / name).read_bytes())


class TestManifestPersistence:
    def test_write_then_read_round_trips_schema_manifest(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        _copy_image("hi_res_page.png", images / "page_002.png")
        _copy_image("hi_res_page.png", images / "page_001.png")

        manifest = generate_manifest(images)
        manifest_path = tmp_path / "manifest.json"
        write_manifest(manifest, manifest_path)

        assert read_manifest(manifest_path) == manifest
        assert json.loads(manifest_path.read_text())["total_pages"] == 2

    def test_read_rejects_invalid_manifest_json(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("not json")

        with pytest.raises(ManifestValidationError, match="valid JSON"):
            read_manifest(path)

    def test_load_or_generate_writes_manifest_when_missing(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        _copy_image("hi_res_page.png", images / "cover.png")

        manifest_path = tmp_path / "manifest.json"
        manifest = load_or_generate_manifest(images, manifest_path)

        assert manifest_path.exists()
        assert manifest.pages == [
            {"filename": "cover.png", "page_number": 1, "status": "pending"}
        ]


class TestManifestValidation:
    def test_generate_manifest_orders_supported_images_and_ignores_non_images(self, tmp_path):
        _copy_image("hi_res_page.png", tmp_path / "page_010.png")
        _copy_image("hi_res_page.png", tmp_path / "page_002.png")
        (tmp_path / "notes.txt").write_text("not an image")

        manifest = generate_manifest(tmp_path)

        assert [page["filename"] for page in manifest.pages] == [
            "page_002.png",
            "page_010.png",
        ]
        assert [page["page_number"] for page in manifest.pages] == [1, 2]
        assert manifest.total_pages == 2

    def test_validate_rejects_nonsequential_page_numbers_in_manifest_order(self, tmp_path):
        _copy_image("hi_res_page.png", tmp_path / "page_001.png")
        _copy_image("hi_res_page.png", tmp_path / "page_002.png")
        from btran.schema import Manifest

        manifest = Manifest(
            input_dir=str(tmp_path),
            pages=[
                {"filename": "page_001.png", "page_number": 2, "status": "pending"},
                {"filename": "page_002.png", "page_number": 1, "status": "pending"},
            ],
            total_pages=2,
        )

        with pytest.raises(ManifestValidationError, match="ordered sequentially"):
            validate_manifest(manifest)

    def test_validate_rejects_missing_referenced_page(self, tmp_path):
        from btran.schema import Manifest

        manifest = Manifest(
            input_dir=str(tmp_path),
            pages=[{"filename": "missing.png", "page_number": 1, "status": "pending"}],
            total_pages=1,
        )

        with pytest.raises(ManifestValidationError, match="missing.png"):
            validate_manifest(manifest)

    def test_validate_rejects_invalid_page_shape_and_total(self, tmp_path):
        from btran.schema import Manifest

        manifest = Manifest(
            input_dir=str(tmp_path),
            pages=[{"filename": "page.png", "page_number": 1}],
            total_pages=2,
        )

        with pytest.raises(ManifestValidationError, match="total_pages"):
            validate_manifest(manifest)

    def test_validate_rejects_page_path_outside_input_directory(self, tmp_path):
        from btran.schema import Manifest

        outside = tmp_path.parent / "outside.png"
        _copy_image("hi_res_page.png", outside)
        manifest = Manifest(
            input_dir=str(tmp_path),
            pages=[{"filename": "../outside.png", "page_number": 1, "status": "pending"}],
            total_pages=1,
        )

        with pytest.raises(ManifestValidationError, match="bare filename"):
            validate_manifest(manifest)

    def test_read_rejects_non_string_input_directory(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"input_dir": 42, "pages": [], "total_pages": 0}))

        with pytest.raises(ManifestValidationError, match="input_dir must be a string"):
            read_manifest(path)

    @pytest.mark.parametrize("filename", ["subdir/../page.png", r"..\\page.png", "{absolute_path}"])
    def test_validate_rejects_non_filename_references(self, tmp_path, filename):
        from btran.schema import Manifest

        page = tmp_path / "page.png"
        _copy_image("hi_res_page.png", page)
        if filename == "{absolute_path}":
            filename = str(page)
        manifest = Manifest(
            input_dir=str(tmp_path),
            pages=[{"filename": filename, "page_number": 1, "status": "pending"}],
            total_pages=1,
        )

        with pytest.raises(ManifestValidationError, match="bare filename"):
            validate_manifest(manifest)
