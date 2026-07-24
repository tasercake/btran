"""Tests for Issue #5: per-page failure handling, EPUB gating, streaming error reporting.

TDD: tests-first, watch them fail, then implement in orchestrator.py + cli.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran.config import Config
from btran.hasher import ImageCache
from btran.orchestrator import run, RunResult
from btran.schema import ErrorResult, PageResult
from btran.translator import TranslationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(path: Path, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Write a tiny 8x8 PNG for testing."""
    from PIL import Image
    img = Image.new("RGB", (8, 8), color=color)
    img.save(path, format="PNG")


def _make_page_result(page_number: int = 1) -> PageResult:
    """Create a minimal PageResult for mocking."""
    return PageResult(
        page_number=page_number,
        sha256="a" * 64,
        phash="b" * 16,
        source_lang="ja",
        target_lang="en",
        page_text="こんにちは",
        translated_text="Hello",
    )


def _make_config(input_dir: Path, intermediate_dir: Path, cache_db: Path, **kwargs) -> Config:
    defaults: dict = {
        "input_dir": input_dir,
        "output_epub": input_dir / "out.epub",
        "intermediate_dir": intermediate_dir,
        "cache_db": cache_db,
        "target_lang": "en",
        "source_lang": "ja",
        "concurrency": 4,
        "max_retries": 2,
        "no_resume": False,
    }
    defaults.update(kwargs)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# 1. One page failing among successes → run returns errors, no EPUB
# ---------------------------------------------------------------------------

class TestOneFailureAmongSuccesses:
    @pytest.mark.asyncio
    async def test_run_result_contains_error_and_no_epub_created(self, tmp_path: Path):
        """2 images: one succeeds, one fails → RunResult has 1 error, no EPUB."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        epub_path = input_dir / "out.epub"
        config = _make_config(input_dir, intermediate_dir, cache_db, output_epub=epub_path)

        img1 = input_dir / "a.png"
        img2 = input_dir / "b.png"
        _make_test_image(img1, color=(255, 0, 0))
        _make_test_image(img2, color=(0, 255, 0))

        mock_translate = AsyncMock(
            side_effect=[
                _make_page_result(1),         # page 1 succeeds
                TranslationError("fail 1"),   # page 2 attempt 1
                TranslationError("fail 2"),   # page 2 attempt 2 (exhausted)
            ]
        )

        with patch("btran.orchestrator.translate_image", mock_translate):
            result = await run(config)

        # Result contains error
        assert isinstance(result, RunResult)
        assert len(result.errors) == 1
        assert "page 2" in result.errors[0].lower() or "2" in result.errors[0]

        # EPUB must NOT be created
        assert not epub_path.exists(), "EPUB should not be created when any page fails"


# ---------------------------------------------------------------------------
# 2. Complete success → EPUB created
# ---------------------------------------------------------------------------

class TestAllPagesSucceed:
    @pytest.mark.asyncio
    async def test_run_result_empty_errors_and_epub_created(self, tmp_path: Path):
        """All pages succeed → RunResult has no errors, EPUB is created."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        epub_path = input_dir / "out.epub"
        config = _make_config(input_dir, intermediate_dir, cache_db, output_epub=epub_path)

        for i, name in enumerate(["a.png", "b.png"]):
            _make_test_image(input_dir / name, color=(i * 100, 200, 150))

        mock_translate = AsyncMock(
            side_effect=[
                _make_page_result(1),
                _make_page_result(2),
            ]
        )

        with patch("btran.orchestrator.translate_image", mock_translate):
            result = await run(config)

        assert len(result.errors) == 0
        assert epub_path.exists(), "EPUB should be created when all pages succeed"


# ---------------------------------------------------------------------------
# 3. Per-page failure stderr callback
# ---------------------------------------------------------------------------

class TestStreamingErrorCallback:
    @pytest.mark.asyncio
    async def test_on_page_error_called_for_each_failed_page(self, tmp_path: Path):
        """on_page_error callback receives page_number and message for each failure."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        epub_path = input_dir / "out.epub"
        config = _make_config(input_dir, intermediate_dir, cache_db, output_epub=epub_path)

        img1 = input_dir / "a.png"
        img2 = input_dir / "b.png"
        img3 = input_dir / "c.png"
        _make_test_image(img1, color=(255, 0, 0))
        _make_test_image(img2, color=(0, 255, 0))
        _make_test_image(img3, color=(0, 0, 255))

        mock_translate = AsyncMock(
            side_effect=[
                _make_page_result(1),
                TranslationError("fail a"),
                TranslationError("fail a"),
                TranslationError("fail b"),
                TranslationError("fail b"),
            ]
        )

        errors_reported: list[tuple[int, str]] = []

        def on_error(page_num: int, msg: str) -> None:
            errors_reported.append((page_num, msg))

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config, on_page_error=on_error)

        # Page 2 failed, page 3 failed → 2 error callbacks
        assert len(errors_reported) == 2, f"Expected 2 error callbacks, got {errors_reported}"
        page_nums = {pn for pn, _ in errors_reported}
        assert page_nums == {2, 3}


# ---------------------------------------------------------------------------
# 4. Stale intermediate files excluded
# ---------------------------------------------------------------------------

class TestStaleFilesExcluded:
    @pytest.mark.asyncio
    async def test_stale_page_json_from_prior_run_not_used(self, tmp_path: Path):
        """Pre-existing page_*.json from a prior run must not count as success."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        intermediate_dir.mkdir(parents=True)
        cache_db = tmp_path / "cache.db"
        epub_path = input_dir / "out.epub"
        config = _make_config(input_dir, intermediate_dir, cache_db, output_epub=epub_path)

        # Create only one image (page 1)
        img1 = input_dir / "a.png"
        _make_test_image(img1, color=(255, 0, 0))

        # Plant a stale page_0002.json from a "prior run"
        stale_result = PageResult(
            page_number=2,
            sha256="d" * 64,
            phash="e" * 16,
            source_lang="ja",
            target_lang="en",
            page_text="stale",
            translated_text="stale translation",
        )
        stale_result.to_file(intermediate_dir / "page_0002.json")

        mock_translate = AsyncMock(return_value=_make_page_result(1))

        with patch("btran.orchestrator.translate_image", mock_translate):
            result = await run(config)

        # The stale file should be cleaned up; only page 1 matters
        assert len(result.errors) == 0
        assert epub_path.exists()

        # The stale page_0002.json should have been removed (not in expected pages)
        assert not (intermediate_dir / "page_0002.json").exists()


# ---------------------------------------------------------------------------
# 5. Malformed intermediate file excluded
# ---------------------------------------------------------------------------

class TestMalformedFileExcluded:
    @pytest.mark.asyncio
    async def test_corrupted_json_is_treated_as_missing_page(self, tmp_path: Path):
        """A page_*.json with invalid JSON counts as missing → no EPUB."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        intermediate_dir.mkdir(parents=True)
        cache_db = tmp_path / "cache.db"
        epub_path = input_dir / "out.epub"
        config = _make_config(input_dir, intermediate_dir, cache_db, output_epub=epub_path)

        # Create two images
        img1 = input_dir / "a.png"
        img2 = input_dir / "b.png"
        _make_test_image(img1, color=(255, 0, 0))
        _make_test_image(img2, color=(0, 255, 0))

        # Mock: page 1 succeeds, page 2 also "succeeds" from mock but
        # we'll corrupt it after the run writes it
        mock_translate = AsyncMock(
            side_effect=[_make_page_result(1), _make_page_result(2)]
        )

        # Intercept _atomic_write to corrupt page 2's output
        from btran.orchestrator import _atomic_write as original_atomic_write

        def corrupting_atomic_write(path: Path, content: str) -> None:
            if "page_0002" in str(path):
                path.write_text("this is not json {{{")
            else:
                original_atomic_write(path, content)

        with patch("btran.orchestrator._atomic_write", corrupting_atomic_write):
            with patch("btran.orchestrator.translate_image", mock_translate):
                result = await run(config)

        # Should detect page 2 is malformed → error, no EPUB
        assert len(result.errors) >= 1
        assert not epub_path.exists()


# ---------------------------------------------------------------------------
# 6. Unexpected task exception capture
# ---------------------------------------------------------------------------

class TestUnexpectedTaskException:
    @pytest.mark.asyncio
    async def test_runtime_error_in_translate_captured_as_failure(self, tmp_path: Path):
        """An unexpected RuntimeError inside translate_image is captured, not crash."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        epub_path = input_dir / "out.epub"
        config = _make_config(input_dir, intermediate_dir, cache_db, output_epub=epub_path)

        img1 = input_dir / "a.png"
        _make_test_image(img1, color=(255, 0, 0))

        mock_translate = AsyncMock(side_effect=RuntimeError("unexpected crash!"))

        with patch("btran.orchestrator.translate_image", mock_translate):
            result = await run(config)

        assert len(result.errors) == 1
        assert "RuntimeError" in result.errors[0] or "unexpected" in result.errors[0].lower()
        assert not epub_path.exists()


# ---------------------------------------------------------------------------
# 7. Run manifest: cached pages still count as this-run successes, stale excluded
# ---------------------------------------------------------------------------

class TestRunManifest:
    @pytest.mark.asyncio
    async def test_cached_page_counts_as_current_success(self, tmp_path: Path):
        """Cached page should be written fresh and counted as current-run success."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        epub_path = input_dir / "out.epub"
        config = _make_config(input_dir, intermediate_dir, cache_db, output_epub=epub_path)

        img1 = input_dir / "a.png"
        _make_test_image(img1, color=(255, 0, 0))

        from btran.hasher import compute_phash, compute_prompt_fingerprint, compute_sha256
        from btran.translator import TRANSLATION_PROMPT

        prompt_v = compute_prompt_fingerprint(TRANSLATION_PROMPT)
        sha1 = compute_sha256(img1)
        ph1 = compute_phash(img1)
        cached = PageResult(
            page_number=99, sha256=sha1, phash=ph1,
            source_lang="ja", target_lang="en",
            page_text="cached", translated_text="cached",
        )
        icache = ImageCache(cache_db)
        icache.store(
            sha1, ph1, str(img1), cached,
            source_lang="ja", target_lang="en", model=config.model,
            prompt_version=prompt_v,
        )
        icache.close()

        mock_translate = AsyncMock()
        with patch("btran.orchestrator.translate_image", mock_translate):
            result = await run(config)

        mock_translate.assert_not_called()
        assert len(result.errors) == 0
        assert epub_path.exists()
