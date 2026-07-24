"""Issue #5 failure semantics retained by the typed two-pass workflow."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.config import Config
from btran.orchestrator import run
from btran.schema import PageExtraction, SourceBlock, TranslatedBlock


def _config(tmp_path: Path) -> Config:
    return Config(input_dir=tmp_path / "in", output_epub=tmp_path / "out.epub", intermediate_dir=tmp_path / "work", cache_db=tmp_path / "cache.sqlite", source_lang="ja", target_lang="en", max_retries=1)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (600, 600), color).save(path)


def _source(number: int, path: Path) -> PageExtraction:
    return PageExtraction(number, str(path), "a" * 64, f"{number:016x}", "ja", "vision", blocks=[SourceBlock(f"page_{number}_block_0", "paragraph", "source", 0)])


@pytest.mark.asyncio
async def test_terminal_translation_failure_is_streamed_and_blocks_epub(tmp_path: Path):
    cfg = _config(tmp_path); cfg.input_dir.mkdir(); _image(cfg.input_dir / "a.png", (1, 2, 3))
    reported: list[tuple[int, str]] = []
    source = _source(1, cfg.input_dir / "a.png")
    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), patch("btran.orchestrator.translate_blocks", AsyncMock(side_effect=RuntimeError("model broke"))), patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg, on_page_error=lambda page, message: reported.append((page, message)))
    assert result.errors and reported == [(1, "RuntimeError: model broke")]
    epub.assert_not_called()


@pytest.mark.asyncio
async def test_all_pages_must_have_exact_translated_ids_before_epub(tmp_path: Path):
    cfg = _config(tmp_path); cfg.input_dir.mkdir(); _image(cfg.input_dir / "a.png", (1, 2, 3)); _image(cfg.input_dir / "b.png", (4, 5, 6))
    sources = [_source(1, cfg.input_dir / "a.png"), _source(2, cfg.input_dir / "b.png")]
    async def translate(source, glossary, **kwargs):
        if source.page_number == 1:
            return [TranslatedBlock("wrong", "translation")]
        return [TranslatedBlock(source.blocks[0].id, "translation")]
    with patch("btran.orchestrator.extract_page", AsyncMock(side_effect=sources)), patch("btran.orchestrator.translate_blocks", side_effect=translate), patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg)
    assert result.errors and "page 1" in result.errors[0]
    epub.assert_not_called()
