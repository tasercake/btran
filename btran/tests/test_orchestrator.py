"""Tests for btran.orchestrator — TDD: tests first, then implementation."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.config import Config
from btran.hasher import ImageCache
from btran.orchestrator import run
from btran.schema import ErrorResult, PageResult
from btran.translator import TranslationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(path: Path, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Write a tiny 8x8 PNG for testing."""
    img = Image.new("RGB", (8, 8), color=color)
    img.save(path, format="PNG")


def _make_diagonal(path: Path, size: int = 64) -> None:
    """Write a PNG with a red diagonal on white — phash very different from solids."""
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    pix = img.load()
    for i in range(size):
        pix[i, i] = (255, 0, 0)
    img.save(path, format="PNG")


def _make_config(input_dir: Path, intermediate_dir: Path, cache_db: Path, **kwargs) -> Config:
    """Create a Config for testing with sensible defaults."""
    defaults: dict = {
        "input_dir": input_dir,
        "output_epub": input_dir / "out.epub",
        "intermediate_dir": intermediate_dir,
        "cache_db": cache_db,
        "target_lang": "en",
        "source_lang": "ja",
        "concurrency": 4,
        "max_retries": 3,
        "no_resume": False,
    }
    defaults.update(kwargs)
    return Config(**defaults)


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


# ---------------------------------------------------------------------------
# Empty input dir
# ---------------------------------------------------------------------------

class TestEmptyInputDir:
    @pytest.mark.asyncio
    async def test_empty_dir_prints_message_and_skips_translation(self, tmp_path: Path):
        """Empty input dir → no-op, prints message, no translation calls."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db)

        mock_translate = AsyncMock()

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        mock_translate.assert_not_called()


# ---------------------------------------------------------------------------
# Two images, no cache
# ---------------------------------------------------------------------------

class TestTwoImagesNoCache:
    @pytest.mark.asyncio
    async def test_translates_both_and_writes_intermediate_files(self, tmp_path: Path):
        """2 images, no cache → calls translate_image twice, writes 2 JSON files."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db)

        # Create two test images
        img1 = input_dir / "page_01.png"
        img2 = input_dir / "page_02.jpg"
        _make_test_image(img1, color=(255, 0, 0))
        _make_test_image(img2, color=(0, 255, 0))

        mock_translate = AsyncMock(
            side_effect=[
                _make_page_result(1),
                _make_page_result(2),
            ]
        )

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        # translate_image called twice
        assert mock_translate.call_count == 2

        # Two intermediate files written
        f1 = intermediate_dir / "page_0001.json"
        f2 = intermediate_dir / "page_0002.json"
        assert f1.exists()
        assert f2.exists()

        # Verify file content
        data1 = json.loads(f1.read_text())
        assert data1["page_number"] == 1
        data2 = json.loads(f2.read_text())
        assert data2["page_number"] == 2


# ---------------------------------------------------------------------------
# Cached images
# ---------------------------------------------------------------------------

class TestCachedImages:
    @pytest.mark.asyncio
    async def test_skips_translation_for_exact_cache_hit(self, tmp_path: Path):
        """Image with SHA256 in cache → skips translation, writes from cache."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db)

        img1 = input_dir / "a.png"
        img2 = input_dir / "b.png"
        _make_test_image(img1, color=(255, 0, 0))
        _make_diagonal(img2)  # very different phash from solid red → no match

        # Pre-populate cache for img1
        from btran.hasher import compute_phash, compute_prompt_fingerprint, compute_sha256
        from btran.translator import TRANSLATION_PROMPT

        prompt_v = compute_prompt_fingerprint(TRANSLATION_PROMPT)
        sha1 = compute_sha256(img1)
        ph1 = compute_phash(img1)
        cached_result = PageResult(
            page_number=99,
            sha256=sha1,
            phash=ph1,
            source_lang="ja",
            target_lang="en",
            page_text="cached text",
            translated_text="cached translation",
        )
        icache = ImageCache(cache_db)
        icache.store(
            sha1, ph1, str(img1), cached_result,
            source_lang="ja", target_lang="en", model=config.model,
            prompt_version=prompt_v,
        )
        icache.close()

        mock_translate = AsyncMock(return_value=_make_page_result(2))

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        # Only img2 was translated (1 call)
        assert mock_translate.call_count == 1

        # Both intermediate files exist
        assert (intermediate_dir / "page_0001.json").exists()
        assert (intermediate_dir / "page_0002.json").exists()

        # The cached file uses the original page_number after renumbering
        data1 = json.loads((intermediate_dir / "page_0001.json").read_text())
        assert data1["page_number"] == 1
        assert data1["page_text"] == "cached text"


# ---------------------------------------------------------------------------
# no_resume = True
# ---------------------------------------------------------------------------

class TestNoResume:
    @pytest.mark.asyncio
    async def test_forces_retranslation_when_no_resume_is_true(self, tmp_path: Path):
        """no_resume=True → forces re-translation even when cached."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db, no_resume=True)

        img1 = input_dir / "a.png"
        _make_test_image(img1, color=(255, 0, 0))

        # Pre-populate cache
        from btran.hasher import compute_phash, compute_prompt_fingerprint, compute_sha256
        from btran.translator import TRANSLATION_PROMPT

        prompt_v = compute_prompt_fingerprint(TRANSLATION_PROMPT)
        sha1 = compute_sha256(img1)
        ph1 = compute_phash(img1)
        icache = ImageCache(cache_db)
        icache.store(
            sha1, ph1, str(img1), _make_page_result(99),
            source_lang="ja", target_lang="en", model=config.model,
            prompt_version=prompt_v,
        )
        icache.close()

        mock_translate = AsyncMock(return_value=_make_page_result(1))

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        # Still translated despite cache hit
        mock_translate.assert_called_once()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_serial_execution_order_with_concurrency_one(self, tmp_path: Path):
        """concurrency=1 → serial execution (no overlap)."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db, concurrency=1)

        img1 = input_dir / "a.png"
        img2 = input_dir / "b.png"
        _make_test_image(img1, color=(255, 0, 0))
        _make_test_image(img2, color=(0, 255, 0))

        call_order: list[int] = []
        running: list[int] = []

        async def serial_mock(**kwargs) -> PageResult:
            page_number = kwargs["page_number"]
            call_order.append(page_number)
            running.append(page_number)
            # If concurrency > 1, running would have more than 1 element
            assert len(running) == 1, f"Expected serial, but {running} running"
            await asyncio.sleep(0.01)
            running.remove(page_number)
            return _make_page_result(page_number)

        with patch("btran.orchestrator.translate_image", side_effect=serial_mock):
            await run(config)

        assert len(call_order) == 2


# ---------------------------------------------------------------------------
# Retry: fails once then succeeds
# ---------------------------------------------------------------------------

class TestRetrySuccess:
    @pytest.mark.asyncio
    async def test_fails_once_then_succeeds_with_retry_count_one(self, tmp_path: Path):
        """translate_image fails once (TranslationError), then succeeds → retry_count=1."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db, max_retries=3)

        img = input_dir / "a.png"
        _make_test_image(img, color=(255, 0, 0))

        mock_translate = AsyncMock(
            side_effect=[
                TranslationError("temporary failure"),
                _make_page_result(1),
            ]
        )

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        assert mock_translate.call_count == 2

        # Result saved with retry_count=1
        data = json.loads((intermediate_dir / "page_0001.json").read_text())
        assert data["retry_count"] == 1


# ---------------------------------------------------------------------------
# Retry exhaustion
# ---------------------------------------------------------------------------

class TestRetryExhaustion:
    @pytest.mark.asyncio
    async def test_all_attempts_fail_saves_error_result(self, tmp_path: Path):
        """translate_image fails all max_retries attempts → ErrorResult saved."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db, max_retries=3)

        img = input_dir / "a.png"
        _make_test_image(img, color=(255, 0, 0))

        mock_translate = AsyncMock(
            side_effect=TranslationError("persistent failure")
        )

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        # All 3 attempts were made
        assert mock_translate.call_count == 3

        # ErrorResult written, not PageResult
        data = json.loads((intermediate_dir / "page_0001.json").read_text())
        assert "error" in data
        assert data["page_number"] == 1
        assert "persistent failure" in data["error"]


# ---------------------------------------------------------------------------
# Intermediate file naming
# ---------------------------------------------------------------------------

class TestIntermediateFileNaming:
    @pytest.mark.asyncio
    async def test_files_named_page_0001_page_0002(self, tmp_path: Path):
        """Intermediate files are named page_0001.json, page_0002.json, etc."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db)

        for i, name in enumerate(["z_first.png", "a_second.png", "m_third.png"]):
            _make_test_image(input_dir / name, color=(i * 80, 100, 200))

        mock_translate = AsyncMock(
            side_effect=[_make_page_result(i) for i in range(1, 4)]
        )

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        assert (intermediate_dir / "page_0001.json").exists()
        assert (intermediate_dir / "page_0002.json").exists()
        assert (intermediate_dir / "page_0003.json").exists()


# ---------------------------------------------------------------------------
# Partial failure summary
# ---------------------------------------------------------------------------

class TestPartialFailureSummary:
    @pytest.mark.asyncio
    async def test_summary_shows_failure_count(self, tmp_path: Path, capsys):
        """One success + one failure → summary shows 1 failed."""
        input_dir = tmp_path / "scans"
        input_dir.mkdir()
        intermediate_dir = tmp_path / "intermediate"
        cache_db = tmp_path / "cache.db"
        config = _make_config(input_dir, intermediate_dir, cache_db, max_retries=2)

        img1 = input_dir / "a.png"
        img2 = input_dir / "b.png"
        _make_test_image(img1, color=(255, 0, 0))
        _make_test_image(img2, color=(0, 255, 0))

        mock_translate = AsyncMock(
            side_effect=[
                _make_page_result(1),  # succeeds
                TranslationError("fail"),  # attempt 1 fails
                TranslationError("fail"),  # attempt 2 fails (exhausted)
            ]
        )

        with patch("btran.orchestrator.translate_image", mock_translate):
            await run(config)

        captured = capsys.readouterr()
        assert "1/2 pages translated" in captured.out
        assert "1 failed" in captured.out
